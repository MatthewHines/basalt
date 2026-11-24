// src/core/elastic_array.cpp - see elastic_array.hpp.
#include "basalt/core/elastic_array.hpp"

#include "basalt/core/device.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <set>
#include <string>

namespace basalt::core {

// The ledger sink (null -> engine log::info). Tests install a capturing sink.
void (*ElasticArray::ledger_sink)(const char* line) = nullptr;

namespace {

// Read the engine-measured free/total device memory (the vram_free_mib quantity, R7-B1). Never mutates.
bool read_free(uint64_t& free_b, uint64_t& total_b, std::string& why) {
    return device_memory(-1, free_b, total_b, why);
}

// Release-with-detection: the shim's release is void (unobservable, arm-only), so the caller peeks at the arm
// before the call — the same peek the fail-shim tests use — and records a release-pending stuck entry when the
// arm is pending. An armless release runs and is assumed to succeed (a real driver failure of a void release is
// not detectable and not handled here).
bool release_ok(unsigned long long handle) {
    if (handle == 0) return true;    if (vmm::fail_armed(vmm::FailOp::Release)) {
        vmm::release(handle);
        return false;
    }
    vmm::release(handle);
    return true;
}

// The retry-capable unmap of a scratch or MAIN VA. One retry: attempt 1 armed -> attempt 2 (real unmap if not
// armed); attempt 1 armed + attempt 2 armed -> both fail -> false. No arm: single real unmap.
bool unmap_with_retry(uint8_t* va, uint64_t bytes) {
    // One attempt, one bounded retry — decided by the RESULT, never by peeking the fail-shim
    // (a pending positional skip is invisible to peeking and would phantom-fail the step).
    return vmm::unmap_only(va, bytes) || vmm::unmap_only(va, bytes);
}


std::string reason_piece(const char* tag, uint64_t index, uint64_t moved) {
    return std::string(tag) + " at piece " + std::to_string(index + 1) + "; moved " + std::to_string(moved);
}

}  // namespace

// ---------------------------------------------------------------- ElasticArray

void ElasticArray::emit_ledger(const char* op, bool failed, const char* step, uint64_t moved, uint64_t free_mib) {
    uint64_t device_bytes = 0;
    for (const auto& [off, e] : extents_) {
        if (e.host) continue;
        if (e.state != ExtentState::S0 && e.state != ExtentState::S4b) continue;
        device_bytes += e.bytes;
    }
    std::string line = op;
    line += ": ";
    if (failed) {
        line += step;
        line += " failed; ";
    }
    line += "moved=" + std::to_string(moved);
    line += " device_bytes=" + std::to_string(device_bytes >> 20);
    line += " free_mib=" + std::to_string(free_mib >> 20);
    if (ledger_sink) ledger_sink(line.c_str());
    else basalt::log::info("basalt memory", "%s", line.c_str());
}

void ElasticArray::recompute_host_start() {
    host_start_ = kNoHost;
    for (const auto& [off, e] : extents_) {
        if (e.host) {
            host_start_ = off;
            break;
        }
    }
}

bool ElasticArray::front_door(std::string& err) {
    if (sync_hook_ && !sync_hook_(sync_hook_ctx_, err)) {
        if (err.empty()) err = "sync failed";
        return false;
    }
    repair_sweep();
    recompute_host_start();
    return true;
}

bool ElasticArray::i1_ok() const {
    uint64_t covered = 0;
    for (const auto& [off, e] : extents_) {
        if (off != covered) return false;
        covered += e.bytes;
    }
    return covered == backed_;
}

bool ElasticArray::i2_ok() const {
    uint64_t device_extent_bytes = 0;
    std::map<uint64_t, uint64_t> by_handle;
    for (const auto& [off, e] : extents_) {
        if (e.host) continue;
        if (e.state != ExtentState::S0 && e.state != ExtentState::S4b) continue;
        device_extent_bytes += e.bytes;
        if (e.handle) by_handle[e.handle] = e.bytes;
    }
    uint64_t live = 0;
    for (const auto& [h, size] : by_handle) live += size;
    return live == device_extent_bytes;
}

void ElasticArray::repair_sweep() {
    if (plain_ || extents_.empty()) return;

    // ---- Stage 1: stuck entries (unmap-retry -> release-retry), insertion order
    for (size_t i = 0; i < stuck_.size();) {
        StuckRecord& s = stuck_[i];
        if (s.kind == 0) {  // unmap-failed: retry the unmap; on success release (L3)
            if (unmap_with_retry(s.va, s.bytes)) {
                if (!release_ok(s.handle)) {
                    stuck_.push_back({s.handle, nullptr, s.bytes, 1});
                }
                stuck_.erase(stuck_.begin() + i);
                continue;
            }
            ++i;  // still stuck — keep and move on
            continue;
        }
        // release-pending: retry the release; on failure keep
        if (release_ok(s.handle)) {
            stuck_.erase(stuck_.begin() + i);
            continue;
        }
        ++i;
    }

    // ---- Stage 2: S3 extents (offset order) — retry the scratch unmap, then the release (L3)
    for (auto it = extents_.begin(); it != extents_.end(); ++it) {
        if (it->second.state != ExtentState::S3) continue;
        const uint8_t staging = it->second.staging;
        uint8_t* scratch_va = staging == 1 ? reinterpret_cast<uint8_t*>(scratch_d_va_)
                                           : reinterpret_cast<uint8_t*>(scratch_h_va_);
        if (unmap_with_retry(scratch_va, it->second.bytes)) {
            if (staging == 1) w_d_live_ = false; else w_h_live_ = false;
            if (!release_ok(it->second.pending)) {
                stuck_.push_back({it->second.pending, nullptr, it->second.bytes, 1});
            }
            it->second.pending = 0;
            it->second.staging = 0;
            it->second.state = ExtentState::S0;
        } else {
            basalt::log::warn("basalt memory", "repair: S3 extent at %llu still stuck",
                              (unsigned long long) it->second.off);
        }
    }

    // ---- Stage 3: S4b extents (offset order) — re-run the canonical SetAccess
    for (auto& [off, e] : extents_) {
        if (e.state != ExtentState::S4b) continue;
        const bool ok = e.host ? vmm::set_access_host(base_ + e.off, e.bytes)
                               : vmm::set_access_device(base_ + e.off, e.bytes);
        if (ok) e.state = ExtentState::S0;
        else basalt::log::warn("basalt memory", "repair: S4b extent at %llu access still missing",
                               (unsigned long long) e.off);
    }

    // ---- Stage 4: S4 extents (offset order) — map bytes-holder back onto MAIN
    for (auto it = extents_.begin(); it != extents_.end(); ++it) {
        Extent& e = it->second;
        if (e.state != ExtentState::S4) continue;
        uint8_t* va = base_ + e.off;
        const vmm::MapOutcome outcome = e.host ? vmm::map_host_handle(va, e.bytes, e.handle)
                                               : vmm::map_handle(va, e.bytes, e.handle);
        if (outcome == vmm::MapOutcome::AccessFail) {
            // Mapped without access is exactly S4b: stage 3 of the NEXT sweep re-runs SetAccess.
            e.state = ExtentState::S4b;
            basalt::log::warn("basalt memory", "repair: S4 extent at %llu mapped, access pending (S4b)",
                              (unsigned long long) e.off);
            continue;
        }
        if (outcome == vmm::MapOutcome::MapFail) {
            basalt::log::warn("basalt memory", "repair: S4 extent at %llu still broken",
                              (unsigned long long) e.off);
            continue;
        }
        // FLIP extent backing to match the bytes; the record already carries the bytes-holder.
        e.state = ExtentState::S0;
        e.pending = 0;
        e.staging = 0;

        // No other-extent cleanup here: a live pending is, by definition, an S3 scratch mapping
        // (stage 2 owns its unmap->release chain) — releasing it here would break L3 and mark it
        // S3-as-S0. Stage-4 repair touches exactly the extent being healed.
    }

    if (!i2_ok()) {
        basalt::log::warn("basalt memory", "repair: I2 violated after the sweep");
    }
}

// ---------------------------------------------------------------- open / ensure / back_host / trim

ElasticArray::~ElasticArray() {
    if (!base_) return;
    if (plain_) {
        cudaFree(base_);
        return;
    }
    cudaDeviceSynchronize();
    // Teardown order (L3): every mapping goes first, then every handle — never release mapped.
    for (const auto& [off, e] : extents_) {
        if (e.handle) vmm::unmap_only(base_ + e.off, e.bytes);
    }
    for (const auto& [off, e] : extents_) {
        if (e.pending) vmm::unmap_only(reinterpret_cast<uint8_t*>(e.staging == 1 ? scratch_d_va_ : scratch_h_va_), e.bytes);
    }
    if (w_d_live_) vmm::unmap_only(reinterpret_cast<uint8_t*>(scratch_d_va_), gran_);
    if (w_h_live_) vmm::unmap_only(reinterpret_cast<uint8_t*>(scratch_h_va_), gran_);
    for (const auto& s : stuck_) {
        if (s.va && s.kind == 0) vmm::unmap_only(s.va, s.bytes);
    }
    for (const auto& [off, e] : extents_) {
        if (e.handle) vmm::release(e.handle);
        if (e.pending) vmm::release(e.pending);
    }
    for (const auto& s : stuck_) vmm::release(s.handle);
    vmm::release_range(base_, reserved_);
}

bool ElasticArray::open(uint64_t bytes, std::string& err) {
    if (bytes == 0) bytes = 1;
    if (vmm::supported()) {
        gran_ = vmm::granularity();
        if (gran_) {
            extents_limit_ = (bytes + gran_ - 1) / gran_ * gran_;
            reserved_ = extents_limit_ + 2 * gran_;   // W_d/W_h in the tail
            base_ = vmm::reserve(reserved_);
            if (base_) {
                scratch_d_va_ = reinterpret_cast<uint64_t>(base_) + extents_limit_;
                scratch_h_va_ = reinterpret_cast<uint64_t>(base_) + extents_limit_ + gran_;
                return true;
            }
        }
    }
    err = "";
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        err = "out of device memory for the array";
        return false;
    }
    cudaMemset(p, 0, bytes);
    base_ = (uint8_t*) p;
    reserved_ = backed_ = bytes;
    extents_limit_ = bytes;
    plain_ = true;
    return true;
}

// map only to the requested bound, rounded UP to the granule the driver requires: never more
// than one granule past the bound (the plan charges the raw slice, so keep the overshoot small)
uint64_t ElasticArray::growth(uint64_t bytes) const {
    if (bytes <= backed_ || bytes > extents_limit_) return 0;
    return std::min((bytes + gran_ - 1) / gran_ * gran_, extents_limit_) - backed_;
}

bool ElasticArray::ensure(uint64_t bytes, uint64_t keep_free, std::string& err) {
    if (!front_door(err)) {
        emit_ledger("ensure", true, "sync", 0, 0);
        return false;
    }
    if (plain_ || !base_) {
        if (bytes <= backed_) return true;
        err = "plain array cannot grow";
        emit_ledger("ensure", true, "preflight", 0, 0);
        return false;
    }
    if (bytes <= backed_) return true;
    if (bytes > extents_limit_) {
        err = "the array is full";
        emit_ledger("ensure", true, "preflight", 0, 0);
        return false;
    }
    // past the host boundary the bytes live in RAM: nothing left for the device to map there
    if (host_start_ != kNoHost) {
        if (bytes > host_start_) bytes = host_start_;
        if (bytes <= backed_) return true;
    }
    uint64_t free_b = 0, total_b = 0;
    std::string why;
    if (!read_free(free_b, total_b, why)) {
        err = "cannot read the free device memory: " + why;
        emit_ledger("ensure", true, "floor", 0, 0);
        return false;
    }
    const uint64_t want = std::min((bytes + gran_ - 1) / gran_ * gran_, extents_limit_);
    uint64_t at = backed_;
    if (at % gran_ != 0) at = (at + gran_ - 1) / gran_ * gran_;
    if (at >= want) return true;
    uint64_t moved = 0;
    while (at < want) {
        const uint64_t n = std::min(gran_, want - at);
        if (keep_free > 0 && free_b < n + keep_free) {
            err = "the VRAM reserve of " + std::to_string(keep_free >> 20) + " MiB stays free";
            emit_ledger("ensure", true, "floor", moved, free_b);
            return false;
        }
        const unsigned long long h = vmm::map(base_ + at, n);
        if (!h) {
            err = "out of device memory for the array";
            emit_ledger("ensure", true, "map", moved, free_b);
            return false;
        }
        cudaMemset(base_ + at, 0, n);
        extents_[at] = {at, n, h, false, ExtentState::S0, 0, 0};
        backed_ = at + n;
        at = backed_;
        ++moved;
        if (host_start_ != kNoHost && at >= host_start_) break;
    }
    recompute_host_start();
    emit_ledger("ensure", false, "", moved, free_b);
    return true;
}

bool ElasticArray::back_host(uint64_t bytes, std::string& err) {
    if (!front_door(err)) {
        emit_ledger("back_host", true, "sync", 0, 0);
        return false;
    }
    if (plain_ || !base_) {
        err = "host backing needs virtual memory management";
        emit_ledger("back_host", true, "preflight", 0, 0);
        return false;
    }
    if (bytes <= backed_) return true;
    if (bytes > extents_limit_) {
        err = "the array is full";
        emit_ledger("back_host", true, "preflight", 0, 0);
        return false;
    }
    const uint64_t hg = vmm::host_granularity();
    if (hg == 0) {
        err = "host backing needs virtual memory management";
        emit_ledger("back_host", true, "preflight", 0, 0);
        return false;
    }
    // the host tail must be host-granularity aligned: pad the end up (into the reservation) so map_host fits
    if (bytes % hg != 0) {
        const uint64_t padded = (bytes + hg - 1) / hg * hg;
        if (padded <= extents_limit_) {
            bytes = padded;
        } else if (extents_limit_ % hg == 0) {
            bytes = extents_limit_;
        } else {
            err = "the host tail cannot be padded to the host granularity inside the reservation";
            emit_ledger("back_host", true, "preflight", 0, 0);
            return false;
        }
    }
    if (host_start_ == kNoHost) host_start_ = backed_;
    uint64_t at = backed_;
    uint64_t moved = 0;
    while (at < bytes) {
        const uint64_t n = std::min(gran_, bytes - at);   // one granule piece per chunk (step-0)
        const unsigned long long h = vmm::map_host(base_ + at, n);
        if (!h) {
            err = "out of host memory for the streamed range";   // mapped pieces stay host-backed
            emit_ledger("back_host", true, "map", moved, 0);
            return false;
        }
        cudaMemset(base_ + at, 0, n);
        extents_[at] = {at, n, h, true, ExtentState::S0, 0, 0};
        backed_ = at + n;
        at = backed_;
        ++moved;
    }
    recompute_host_start();
    emit_ledger("back_host", false, "", moved, 0);
    return true;
}

void ElasticArray::trim(uint64_t bytes) {
    std::string err;
    if (!front_door(err)) return;
    if (plain_ || !base_ || extents_.empty() || extents_.begin()->second.off < bytes) {
        emit_ledger("trim", false, "", 0, 0);
        return;
    }
    cudaDeviceSynchronize();
    uint64_t moved = 0;
    while (!extents_.empty() && !extents_.rbegin()->second.host && extents_.rbegin()->second.off >= bytes) {
        const auto& e = extents_.rbegin()->second;
        vmm::unmap(base_ + e.off, e.bytes, e.handle);
        extents_.erase(extents_.rbegin()->first);
        ++moved;
    }
    backed_ = extents_.empty() ? 0 : extents_.rbegin()->second.off + extents_.rbegin()->second.bytes;
    recompute_host_start();
    emit_ledger("trim", false, "", moved, 0);
}

// ---------------------------------------------------------------- readouts

uint64_t ElasticArray::live_handle_count() const {
    std::set<uint64_t> s;
    for (const auto& [off, e] : extents_) {
        if (e.handle) s.insert(e.handle);
        if (e.state == ExtentState::S3 && e.pending) s.insert(e.pending);
    }
    for (const auto& st : stuck_) {
        if (st.handle) s.insert(st.handle);
    }
    return s.size();
}

ExtentState ElasticArray::extent_state(uint64_t off) const {
    for (const auto& [o, e] : extents_) {
        if (off >= e.off && off < e.off + e.bytes) return e.state;
    }
    return ExtentState::S0;
}

uint64_t ElasticArray::bytes_holder(uint64_t off) const {
    for (const auto& [o, e] : extents_) {
        if (off >= e.off && off < e.off + e.bytes) {
            if (e.state == ExtentState::S4 || e.state == ExtentState::S4b) return e.handle;
            return 0;
        }
    }
    return 0;
}

std::vector<StuckEntry> ElasticArray::stuck_entries() const {
    std::vector<StuckEntry> out;
    for (const auto& s : stuck_) out.push_back({s.handle, s.va, s.kind});
    return out;
}

uint64_t ElasticArray::scratch_live_maps() const {
    return (w_d_live_ ? 1ULL : 0ULL) + (w_h_live_ ? 1ULL : 0ULL);
}

// ---------------------------------------------------------------- rehome (host -> device)

MoveResult ElasticArray::rehome(uint64_t off, uint64_t bytes, uint64_t keep_free) {
    MoveResult r{};
    std::string err;
    const uint64_t gran = gran_;
    const char* fail_step = "preflight";

    if (plain_ || !base_) {
        r.reason = "virtual memory management required";
        emit_ledger("rehome", true, "preflight", 0, 0);
        return r;
    }
    if (!front_door(err)) {
        r.reason = err;
        emit_ledger("rehome", true, "sync", 0, 0);
        return r;
    }

    if (bytes == 0 || off % gran != 0 || bytes % gran != 0) {
        r.reason = "not granule-aligned or empty";
        fail_step = "preflight";
        emit_ledger("rehome", true, "preflight", 0, 0);
        return r;
    }
    if (off + bytes > extents_limit_) {
        r.reason = "beyond the array";
        fail_step = "preflight";
        emit_ledger("rehome", true, "preflight", 0, 0);
        return r;
    }

    // Pre-flight: every byte covered + host-backed, and every covering extent clean (S0).
    for (uint64_t at = off; at < off + bytes; at += gran) {
        Extent* found = nullptr;
        for (auto& [o, e] : extents_) {
            if (at >= e.off && at < e.off + e.bytes) { found = &e; break; }
        }
        if (!found || !found->host || found->state != ExtentState::S0) {
            r.reason = "not host-backed";
            fail_step = "preflight";
            emit_ledger("rehome", true, "preflight", 0, 0);
            return r;
        }
    }

    // Floor: engine-measured free must cover the moved bytes + keep_free.
    uint64_t free_b = 0, total_b = 0;
    std::string why;
    if (!read_free(free_b, total_b, why)) {
        r.reason = "cannot read the free device memory: " + why;
        fail_step = "floor";
        emit_ledger("rehome", true, "floor", 0, 0);
        return r;
    }
    if (free_b < bytes + keep_free) {
        r.reason = "free " + std::to_string(free_b >> 20) + " < need " + std::to_string(bytes >> 20) +
                   " (+keep " + std::to_string(keep_free >> 20) + ")";
        fail_step = "floor";
        emit_ledger("rehome", true, "floor", 0, free_b);
        return r;
    }

    uint64_t moved = 0;
    for (uint64_t at = off; at < off + bytes; at += gran) {
        const uint64_t chunk_off = at;
        const uint64_t chunk_idx = (at - off) / gran;

        Extent* e = nullptr;
        for (auto& [o, ee] : extents_) {
            if (chunk_off >= ee.off && chunk_off < ee.off + ee.bytes) { e = &ee; break; }
        }
        if (!e) {
            r.moved = moved;
            r.reason = "not host-backed";
            fail_step = "preflight";
            break;
        }    const uint64_t host_handle = e->handle;

        // (a) cuMemCreate DEVICE handle
        const uint64_t dev_h = vmm::create(gran);
        if (!dev_h) {
            r.moved = moved;
            r.reason = reason_piece("device oom", chunk_idx, moved);
            fail_step = "a";
            break;
        }

        // (b) Map device handle -> W_d; SetAccess {DEVICE rw}
        const vmm::MapOutcome ok_b = vmm::map_handle(reinterpret_cast<uint8_t*>(scratch_d_va_), gran, dev_h);
        if (ok_b == vmm::MapOutcome::MapFail) {
            if (!release_ok(dev_h)) stuck_.push_back({dev_h, nullptr, gran, 1});
            r.moved = moved;
            r.reason = reason_piece("map failed", chunk_idx, moved);
            fail_step = "b";
            break;
        }
        if (ok_b == vmm::MapOutcome::AccessFail) {
            if (unmap_with_retry(reinterpret_cast<uint8_t*>(scratch_d_va_), gran)) {
                w_d_live_ = false;
                if (!release_ok(dev_h)) stuck_.push_back({dev_h, nullptr, gran, 1});
            } else {
                w_d_live_ = true;
                stuck_.push_back({dev_h, reinterpret_cast<uint8_t*>(scratch_d_va_), gran, 0});
            }
            r.moved = moved;
            r.reason = reason_piece("access failed", chunk_idx, moved);
            fail_step = "b";
            break;
        }
        w_d_live_ = true;

        // (c) cuMemcpyHtoDAsync(MAIN host VA -> W_d); sync
        if (!vmm::copy_h2d(reinterpret_cast<uint8_t*>(scratch_d_va_), base_ + chunk_off, gran)) {
            if (unmap_with_retry(reinterpret_cast<uint8_t*>(scratch_d_va_), gran)) {
                w_d_live_ = false;
                if (!release_ok(dev_h)) stuck_.push_back({dev_h, nullptr, gran, 1});
            } else {
                stuck_.push_back({dev_h, reinterpret_cast<uint8_t*>(scratch_d_va_), gran, 0});
            }
            r.moved = moved;
            r.reason = reason_piece("copy failed", chunk_idx, moved);
            fail_step = "c";
            break;
        }

        // (d) Unmap device from W_d
        if (!unmap_with_retry(reinterpret_cast<uint8_t*>(scratch_d_va_), gran)) {
            // S3: pending handle recorded; w_d_live stays; extent backing UNCHANGED (host).
            e->state = ExtentState::S3;
            e->pending = dev_h;
            e->staging = 1;
            r.moved = moved;
            r.reason = reason_piece("unmap failed", chunk_idx, moved);
            fail_step = "d";
            break;
        }
        w_d_live_ = false;

        // (e) Unmap host from MAIN
        if (!unmap_with_retry(base_ + chunk_off, gran)) {
            // NOTHING CHANGED (L2b): host stays live; device released (unmapped — L3). No stuck entry.
            if (!release_ok(dev_h)) stuck_.push_back({dev_h, nullptr, gran, 1});
            r.moved = moved;
            r.reason = "unmap refused; nothing moved beyond chunk " + std::to_string(chunk_idx + 1);
            fail_step = "e";
            break;
        }

        // (f) Map device handle onto MAIN; SetAccess {DEVICE rw}
        const vmm::MapOutcome fo = vmm::map_handle(base_ + chunk_off, gran, dev_h);
        if (fo == vmm::MapOutcome::MapFail) {
            // map fail: RESTORE FIRST — map host back onto MAIN + SetAccess {HOST_NUMA rw, DEVICE rw}
            const vmm::MapOutcome ro = vmm::map_host_handle(base_ + chunk_off, gran, host_handle);
            // release device (unmapped since d — L3; release anyway even if restore map failed)
            if (!release_ok(dev_h)) stuck_.push_back({dev_h, nullptr, gran, 1});
            if (ro == vmm::MapOutcome::MapFail) {
                // double-fault: S4 — bytes-holder = HOST handle (named); device released
                e->state = ExtentState::S4;
                e->host = true;
                e->handle = host_handle;
                e->pending = 0;
                e->staging = 0;
            } else if (ro == vmm::MapOutcome::AccessFail) {
                // restore SetAccess fail: S4b — extent backing = HOST
                e->state = ExtentState::S4b;
                e->host = true;
                e->handle = host_handle;
                e->pending = 0;
                e->staging = 0;
            } else {
                // restore ok: extent backing = HOST (S0)
                e->state = ExtentState::S0;
                e->host = true;
                e->handle = host_handle;
                e->pending = 0;
                e->staging = 0;
            }
            r.moved = moved;
            r.reason = reason_piece("map failed", chunk_idx, moved);
            fail_step = "f";
            break;
        }
        if (fo == vmm::MapOutcome::AccessFail) {
            // SetAccess fail: device at MAIN, no access — unmap it.
            if (unmap_with_retry(base_ + chunk_off, gran)) {
                // unmap ok: restore host (map + canonical host desc)
                const vmm::MapOutcome ro = vmm::map_host_handle(base_ + chunk_off, gran, host_handle);
                if (!release_ok(dev_h)) stuck_.push_back({dev_h, nullptr, gran, 1});
                if (ro == vmm::MapOutcome::MapFail) {
                    // restore map fail: S4 — bytes-holder = HOST handle
                    e->state = ExtentState::S4;
                    e->host = true;
                    e->handle = host_handle;                } else if (ro == vmm::MapOutcome::AccessFail) {
                    // restore SetAccess fail: S4b — extent backing = HOST
                    e->state = ExtentState::S4b;
                    e->host = true;
                    e->handle = host_handle;
                } else {
                    e->state = ExtentState::S0;
                    e->host = true;
                    e->handle = host_handle;
                }
                e->pending = 0;
                e->staging = 0;
                r.moved = moved;
                r.reason = reason_piece("access failed", chunk_idx, moved);
                fail_step = "f";
                break;
            } else {
                // unmap fails: MAIN stuck-held by device; host unmapped — S4 (relaxed)
                stuck_.push_back({dev_h, base_ + chunk_off, gran, 0});
                e->state = ExtentState::S4;
                e->host = true;             // bytes-holder = HOST handle (named, unmapped)
                e->handle = host_handle;
                e->pending = 0;
                e->staging = 0;
                r.moved = moved;
                r.reason = reason_piece("access failed", chunk_idx, moved);
                fail_step = "f";
                break;
            }
        }

        // (g) Release host handle (mapped nowhere — L3); flip extent to device; I2
        if (!release_ok(host_handle)) {
            stuck_.push_back({host_handle, nullptr, gran, 1});
        }
        e->host = false;
        e->handle = dev_h;
        e->state = ExtentState::S0;
        e->pending = 0;
        e->staging = 0;
        recompute_host_start();
        ++moved;
    }

    r.moved = moved;
    if (r.reason.empty()) r.reason = "moved";
    const bool failed = r.reason != "moved";
    emit_ledger("rehome", failed, failed ? fail_step : "", r.moved, free_b);
    return r;
}

// ---------------------------------------------------------------- rehomes_down (device -> host)

MoveResult ElasticArray::rehomes_down(uint64_t off, uint64_t bytes) {
    MoveResult r{};
    std::string err;
    const uint64_t gran = gran_;
    const char* fail_step = "preflight";

    if (plain_ || !base_) {
        r.reason = "virtual memory management required";
        emit_ledger("rehomes_down", true, "preflight", 0, 0);
        return r;
    }
    if (!front_door(err)) {
        r.reason = err;
        emit_ledger("rehomes_down", true, "sync", 0, 0);
        return r;
    }

    if (bytes == 0 || off % gran != 0 || bytes % gran != 0) {
        r.reason = "not granule-aligned or empty";
        fail_step = "preflight";
        emit_ledger("rehomes_down", true, "preflight", 0, 0);
        return r;
    }
    if (off + bytes > extents_limit_) {
        r.reason = "beyond the array";
        fail_step = "preflight";
        emit_ledger("rehomes_down", true, "preflight", 0, 0);
        return r;
    }

    // Pre-flight: every byte covered + device-backed, and every covering extent clean (S0).
    for (uint64_t at = off; at < off + bytes; at += gran) {
        Extent* found = nullptr;
        for (auto& [o, e] : extents_) {
            if (at >= e.off && at < e.off + e.bytes) { found = &e; break; }
        }
        if (!found || found->host || found->state != ExtentState::S0) {
            r.reason = "not device-backed";
            fail_step = "preflight";
            emit_ledger("rehomes_down", true, "preflight", 0, 0);
            return r;
        }
    }

    uint64_t free_b = 0, total_b = 0;
    std::string why;
    if (!read_free(free_b, total_b, why)) {
        r.reason = "cannot read the free device memory: " + why;
        fail_step = "floor";
        emit_ledger("rehomes_down", true, "floor", 0, 0);
        return r;
    }
    const uint64_t free_mib_for_ledger = free_b;

    uint64_t moved = 0;
    for (uint64_t at = off; at < off + bytes; at += gran) {
        const uint64_t chunk_off = at;
        const uint64_t chunk_idx = (at - off) / gran;

        Extent* e = nullptr;
        for (auto& [o, ee] : extents_) {
            if (chunk_off >= ee.off && chunk_off < ee.off + ee.bytes) { e = &ee; break; }
        }
        if (!e) {
            r.moved = moved;
            r.reason = "not device-backed";
            fail_step = "preflight";
            break;
        }
        const uint64_t device_handle = e->handle;

        // (a) cuMemCreate HOST_NUMA handle
        const uint64_t host_h = vmm::create_host(gran);
        if (!host_h) {
            r.moved = moved;
            r.reason = reason_piece("host oom", chunk_idx, moved);
            fail_step = "a";
            break;
        }

        // (b) Map host handle -> W_h; SetAccess {HOST_NUMA rw, DEVICE rw}
        const vmm::MapOutcome ok_b = vmm::map_host_handle(reinterpret_cast<uint8_t*>(scratch_h_va_), gran, host_h);
        if (ok_b == vmm::MapOutcome::MapFail) {
            if (!release_ok(host_h)) stuck_.push_back({host_h, nullptr, gran, 1});
            r.moved = moved;
            r.reason = reason_piece("map failed", chunk_idx, moved);
            fail_step = "b";
            break;
        }
        if (ok_b == vmm::MapOutcome::AccessFail) {
            if (unmap_with_retry(reinterpret_cast<uint8_t*>(scratch_h_va_), gran)) {
                w_h_live_ = false;
                if (!release_ok(host_h)) stuck_.push_back({host_h, nullptr, gran, 1});
            } else {
                w_h_live_ = true;
                stuck_.push_back({host_h, reinterpret_cast<uint8_t*>(scratch_h_va_), gran, 0});
            }
            r.moved = moved;
            r.reason = reason_piece("access failed", chunk_idx, moved);
            fail_step = "b";
            break;
        }
        w_h_live_ = true;

        // (c) cuMemcpyDtoHAsync(MAIN device VA -> W_h); sync
        if (!vmm::copy_d2h(reinterpret_cast<uint8_t*>(scratch_h_va_), base_ + chunk_off, gran)) {
            if (unmap_with_retry(reinterpret_cast<uint8_t*>(scratch_h_va_), gran)) {
                w_h_live_ = false;
                if (!release_ok(host_h)) stuck_.push_back({host_h, nullptr, gran, 1});
            } else {
                stuck_.push_back({host_h, reinterpret_cast<uint8_t*>(scratch_h_va_), gran, 0});
            }
            r.moved = moved;
            r.reason = reason_piece("copy failed", chunk_idx, moved);
            fail_step = "c";
            break;
        }

        // (d) Unmap host from W_h
        if (!unmap_with_retry(reinterpret_cast<uint8_t*>(scratch_h_va_), gran)) {
            // S3: pending handle recorded; w_h_live stays; extent backing UNCHANGED (device).
            e->state = ExtentState::S3;
            e->pending = host_h;
            e->staging = 2;
            r.moved = moved;
            r.reason = reason_piece("unmap failed", chunk_idx, moved);
            fail_step = "d";
            break;
        }
        w_h_live_ = false;

        // (e) Unmap device from MAIN
        if (!unmap_with_retry(base_ + chunk_off, gran)) {
            // NOTHING CHANGED (L2b): device stays live; host released (unmapped — L3). No stuck entry.
            if (!release_ok(host_h)) stuck_.push_back({host_h, nullptr, gran, 1});
            r.moved = moved;
            r.reason = "unmap refused; nothing moved beyond chunk " + std::to_string(chunk_idx + 1);
            fail_step = "e";
            break;
        }

        // (f) Map host onto MAIN; SetAccess {HOST_NUMA rw, DEVICE rw}
        const vmm::MapOutcome fo = vmm::map_host_handle(base_ + chunk_off, gran, host_h);
        if (fo == vmm::MapOutcome::MapFail) {
            // map fail: RESTORE FIRST — map device onto MAIN + SetAccess {DEVICE rw}
            const vmm::MapOutcome ro = vmm::map_handle(base_ + chunk_off, gran, device_handle);
            if (ro == vmm::MapOutcome::MapFail) {
                // restore map fail: S4 — bytes-holder = FRESH HOST handle; DEVICE released (unmapped since e)
                if (!release_ok(device_handle)) {
                    stuck_.push_back({device_handle, nullptr, gran, 1});
                }
                e->state = ExtentState::S4;
                e->host = true;      // bytes-holder = fresh host handle
                e->handle = host_h;
                e->pending = 0;
                e->staging = 0;
            } else if (ro == vmm::MapOutcome::AccessFail) {
                // restore SetAccess fail: S4b — extent backing = DEVICE (mapped at MAIN)
                if (!release_ok(host_h)) {
                    stuck_.push_back({host_h, nullptr, gran, 1});
                }
                e->state = ExtentState::S4b;
                e->host = false;
                e->handle = device_handle;
                e->pending = 0;
                e->staging = 0;
            } else {
                // restore ok: extent backing = DEVICE (S0)
                if (!release_ok(host_h)) {
                    stuck_.push_back({host_h, nullptr, gran, 1});
                }
                e->state = ExtentState::S0;
                e->host = false;
                e->handle = device_handle;
                e->pending = 0;
                e->staging = 0;
            }
            r.moved = moved;
            r.reason = reason_piece("map failed", chunk_idx, moved);
            fail_step = "f";
            break;
        }
        if (fo == vmm::MapOutcome::AccessFail) {
            // SetAccess fail: host mapped at MAIN, no access — unmap it.
            if (unmap_with_retry(base_ + chunk_off, gran)) {
                // unmap ok: restore device
                const vmm::MapOutcome ro = vmm::map_handle(base_ + chunk_off, gran, device_handle);
                if (ro == vmm::MapOutcome::MapFail) {
                    // restore map fail: S4 — bytes-holder = FRESH HOST handle (named; copy (c)
                    // succeeded, so the bytes live ONLY in it — it must be kept). The device is
                    // the old backing, unmapped since e: released (L3).
                    if (!release_ok(device_handle)) {
                        stuck_.push_back({device_handle, nullptr, gran, 1});
                    }
                    e->state = ExtentState::S4;
                    e->host = true;
                    e->handle = host_h;
                } else if (ro == vmm::MapOutcome::AccessFail) {
                    // restore SetAccess fail: S4b — extent backing = DEVICE (mapped at MAIN)
                    if (!release_ok(host_h)) {
                        stuck_.push_back({host_h, nullptr, gran, 1});
                    }
                    e->state = ExtentState::S4b;
                    e->host = false;
                    e->handle = device_handle;
                } else {
                    if (!release_ok(host_h)) {
                        stuck_.push_back({host_h, nullptr, gran, 1});
                    }
                    e->state = ExtentState::S0;
                    e->host = false;
                    e->handle = device_handle;
                }
                e->pending = 0;
                e->staging = 0;
                r.moved = moved;
                r.reason = reason_piece("access failed", chunk_idx, moved);
                fail_step = "f";
                break;
            } else {
                // unmap fails: host HOLDS THE BYTES and sits at MAIN — FLIP extent to host NOW, S4b
                e->state = ExtentState::S4b;
                e->host = true;   // bytes-holder = fresh host handle (this extent's own backing)
                e->handle = host_h;
                e->pending = 0;
                e->staging = 0;
                // the device handle is released (unmapped since e — L3)
                if (!release_ok(device_handle)) {
                    stuck_.push_back({device_handle, nullptr, gran, 1});
                }
                r.moved = moved;
                r.reason = reason_piece("access failed", chunk_idx, moved);
                fail_step = "f";
                break;
            }
        }

        // (g) Release device handle (mapped nowhere — L3); flip extent to host; I2
        if (!release_ok(device_handle)) {
            stuck_.push_back({device_handle, nullptr, gran, 1});
        }
        e->host = true;
        e->handle = host_h;
        e->state = ExtentState::S0;
        e->pending = 0;
        e->staging = 0;
        recompute_host_start();
        ++moved;
    }

    r.moved = moved;
    if (r.reason.empty()) r.reason = "moved";
    const bool failed = r.reason != "moved";
    emit_ledger("rehomes_down", failed, failed ? fail_step : "", r.moved, free_mib_for_ledger);
    return r;
}

}  // namespace basalt::core
