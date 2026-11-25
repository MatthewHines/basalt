// src/program/serve/commands.cpp - scheduling and protocol command dispatch.
#include "state.hpp"
#include "../cli/tokens.hpp"

namespace basalt::program::serve {

namespace {
// the busy clock runs while a request or a batch window does
struct BusyClock {
    double& total;
    bool& on;
    Clock::time_point t0;
    BusyClock(double& total_, bool& on_, Clock::time_point& from) : total(total_), on(on_), t0(Clock::now()) {
        from = t0;
        on = true;
    }
    ~BusyClock() {
        total += std::chrono::duration<double>(Clock::now() - t0).count();
        on = false;
    }
};
}  // namespace

uint64_t fnv1a(const void* data, size_t n, uint64_t h) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

std::vector<ImgKey> Runtime::imgs_below(const std::vector<ImgKey>& all, int64_t L) {
    std::vector<ImgKey> v;
    for (const ImgKey& k : all) if (k.start < L) v.push_back(k);
    return v;
}

bool Runtime::batch_on() { for (const BSlot& b : bs) if (b.active) return true; return false; }

void Runtime::stats_tick(bool force) {
    if (!stats.due(Clock::now(), force)) return;
    std::string out = stats.slot_lines();
    int64_t total = 0;
    std::vector<uint8_t> homes;
    std::vector<uint32_t> routed;
    if (expert_map(total, homes, routed)) out += ConsoleStats::map_line(total, homes, routed);
    std::string cards;
    for (const CardFacts& c : card_facts()) {
        char bus[32] = {}, gbps[32] = "-";
        if (cudaDeviceGetPCIBusId(bus, sizeof bus, c.device) != cudaSuccess) continue;
        if (c.h2d_gbps > 0) std::snprintf(gbps, sizeof gbps, "%.1f", c.h2d_gbps);
        cards += (cards.empty() ? "" : ";") + std::string(bus) + "/" + std::to_string(c.expert_slots) + "/" + gbps + "/" +
                 std::to_string(c.expert_bytes >> 20) + "/" + (c.kv_bytes >= 0 ? std::to_string(c.kv_bytes >> 20) : "-");
    }
    if (!cards.empty() && cards != cards_sent_) {
        cards_sent_ = cards;
        out += "INFO cards=" + cards + "\n";
    }
    if (out.empty()) return;
    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
}

Flow Runtime::dispatch() {
    if (batch_on()) {
        if (!try_next_line(line)) {
            const BusyClock busy(busy_s, busy_on, busy_from);
            const size_t active = (size_t) std::count_if(bs.begin(), bs.end(), [](const BSlot& b) { return b.active; });
            if (!next_batch_window()) return Flow::error;
            // a slot that ended its request has its last figures out with its BDONE
            stats_tick(active != (size_t) std::count_if(bs.begin(), bs.end(), [](const BSlot& b) { return b.active; }));
            return Flow::next;
        }
    } else {
        kv_ceiling_note();   // between requests too: the free VRAM also moves without the KV
        // idle: once no line has come for a second (a client's next turn usually comes sooner), what a crash would
        // lose goes to disk, then the idle conversations whose time is up leave VRAM, one at a time while no line
        // waits
        bool got = false, parking = true;
        std::string pe;
        while (!got && (disk.enabled() || parking)) {
            const int r = next_line_within(line, 1000);
            if (r < 0) { basalt::log::info("basalt serve", "input pipe closed"); return Flow::quit; }
            got = r == 1;
            while (!got && !(got = try_next_line(line)) && disk.enabled() && spill_to_disk(false, 1) > 0) {}
            if (got) break;
            parking = parking && park_idle_engine(pe);
            retire_idle_lanes();   // S2: lanes quiet past the threshold hand their caches to host backing (never fatal)
            if (!pe.empty()) { std::printf("ERR %s\n", pe.c_str()); return Flow::error; }
            if (!parking) break;
        }
        // The wait for the next line wakes once a second: an engine that parks nothing (inkling2) or runs batch slots
        // reaches no other idle moment while the pipe is quiet, and the wall clock needs waking to fire the S2 retire.
        // EOF unwinds exactly as the plain next_line did.
        if (!got) {
            for (;;) {
                const int r = next_line_within(line, 1000);
                if (r == 1) break;
                if (r < 0) { basalt::log::info("basalt serve", "input pipe closed"); return Flow::quit; }
                retire_idle_lanes();   // S2: lanes quiet past the threshold hand their caches to host backing (never fatal)
            }
        }
    }
    // --batch: BSTOP <slot> ends that slot at its next batch window; BGEN <slot> <max_new> ... reads the
    // request's prompt and first token as a GEN 1, then continues it in that slot
    // S2: a generation line is its lane's activity edge. The gate and kv_note_activity share ONE prefix set
    // ("GEN", "SGEN", "BGEN" — the I image variants included): the reader rewrites GENT/SGENT/BGENT to those
    // forms, so the *T literals here were dead and batch wakes were lost.
    if (line.rfind("GEN", 0) == 0 || line.rfind("SGEN", 0) == 0 || line.rfind("BGEN", 0) == 0)
        kv_note_activity(line);
    if (line.rfind("BSTOP ", 0) == 0) {
        const int b = std::atoi(line.c_str() + 6);
        if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
        return Flow::next;
    }
    if (line.rfind("BYIELD", 0) == 0) return Flow::next;   // for a prompt read that has ended meanwhile
    if (line.rfind("ENC ", 0) == 0 || line.rfind("ENCV ", 0) == 0) {   // the in-engine encoder, between requests and windows
        enc_command(line);
        return Flow::next;
    }
    if (line.rfind("PROBE", 0) == 0) {   // between requests, not a request (basalt-probe's commands)
        std::string perr;
        if (batch_on()) perr = "PROBE: not while batch slots are decoding";
        else probe_command(line, perr);
        if (!perr.empty()) std::printf("ERR %s\n", perr.c_str());
        std::fflush(stdout);
        return Flow::next;
    }
    if (line.rfind("VRAM", 0) == 0) {   // between requests, not a request
        std::string verr;
        if (batch_on()) verr = "VRAM: not while batch slots are decoding";
        else vram_command(line, verr, false);
        if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
        std::fflush(stdout);
        stats_tick(true);   // the cards' expert slots moved
        return Flow::next;
    }
    // SLEEP / WAKE, between requests: the expert cache gives its VRAM back down to its smallest size (every
    // expert keeps a RAM copy and the slot it held is remembered), so the process can be suspended with little
    // VRAM left to save; WAKE maps the cache again and puts each expert back in its slot.  Reply: a VRAM line.
    if (line == "SLEEP" || line == "WAKE") {
        std::string verr;
        const bool down = line == "SLEEP";
        if (batch_on()) {
            verr = line + ": not while batch slots are decoding";
        } else {
            sleep_command(down, verr);
        }
        // asleep, the driver holds the engine's GPU state in RAM until the wake: the RAM guard judges no reading
        // then (no shed, no admission wait from it), and judges a fresh one at the wake
        if (down && verr.empty()) basalt::platform::ram_guard_pause(true);
        if (!down) basalt::platform::ram_guard_pause(false);
        if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
        std::fflush(stdout);
        stats_tick(true);
        return Flow::next;
    }
    if (line == "STATS") {   // the server's console: XSLOT and XMAP lines from now on (console_stats.hpp)
        stats.on = true;
        stats_tick(true);
        return Flow::next;
    }
    if (line.rfind("PMATCH ", 0) == 0 || line.rfind("PFREE ", 0) == 0) {   // a prompt held by handle, between requests and windows
        prompt_command(line);
        return Flow::next;
    }
    if (line.rfind("ERR_GENT ", 0) == 0) {
        std::printf("ERR %s\n", line.c_str() + 9);
        std::fflush(stdout);
        return Flow::next;
    }
    basalt::platform::ram_guard_pause(false);   // a request without a WAKE first: the guard judges again
    admit_slot = -1;
    solo_slot = -1;
    if (line.rfind("SGEN ", 0) == 0 || line.rfind("SGENI ", 0) == 0) {   // SGEN <slot> <max_new> ...: a GEN in that slot
        const bool simg = line.rfind("SGENI ", 0) == 0;
        char* e1 = nullptr;
        const long b = std::strtol(line.c_str() + (simg ? 6 : 5), &e1, 10);
        if (!slot0_solo() || bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || e1 == line.c_str() + (simg ? 6 : 5)) {
            std::printf("ERR SGEN: no such free slot or slot 0 is not the solo session\n");
            std::fflush(stdout);
            return Flow::next;
        }
        solo_slot = (int) b;
        line = std::string(simg ? "GENI" : "GEN") + e1;
    }
    if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
        const bool bimg = line.rfind("BGENI ", 0) == 0;
        char* e1 = nullptr;
        const long b = std::strtol(line.c_str() + (bimg ? 6 : 5), &e1, 10);
        char* e2 = nullptr;
        const long long mn = std::strtoll(e1, &e2, 10);
        if (bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || mn < 1 || e2 == e1) {
            std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", o.batch);
            std::fflush(stdout);
            return Flow::next;
        }
        admit_slot = (int) b;
        admit_max_new = mn;
        line = std::string(bimg ? "GENI 1" : "GEN 1") + e2;
    }
    between_requests(line);
    if (line == "QUIT") return Flow::quit;
    // the watchdog watches a request from here until this iteration ends, whichever way it ends
    struct BusyScope {
        BusyScope() { basalt::core::progress().busy.store(true); basalt::core::progress_at("request"); }
        ~BusyScope() { basalt::core::progress().busy.store(false); basalt::core::progress_at("idle"); }
    } busy_scope;
    const BusyClock busy(busy_s, busy_on, busy_from);
    return run_request();
}

// ---- S2 idle-lane KV retire (S2-DESIGN R1/R2). The loop's idle moment calls retire_idle_lanes(); the wall clock
// makes every decision and the moment only wakes it. Retiring is a speed bonus for the rest of the card, never
// fatal: a refusal logs and the next tick tries again (the arrays repair their own stuck entries at the gate, S1).
namespace {
uint64_t kv_secs(Clock::time_point t) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
}
basalt::core::KvPolicyConf kv_conf(const basalt::program::CommonOptions& o) {
    return {(uint64_t) o.kv_idle_demote_seconds, (uint64_t) o.kv_keep_warm_cells};
}
}  // namespace

void Runtime::retire_idle_lanes() {
    if (o.kv_idle_demote_seconds <= 0) return;   // R7: off is exactly today's loop
    const size_t n = (size_t) std::max(o.batch, 1);
    if (kv_pol.size() != n) {   // boot (or the batch size's first use): every lane has been idle since now
        kv_pol.clear(); kv_pol.resize(n);
        kv_idle_at.assign(n, Clock::now());
    }
    const basalt::core::KvPolicyConf conf = kv_conf(o);
    const Clock::time_point now = Clock::now();
    const uint64_t now_s = kv_secs(now);
    if (!batch_on() && kv_line_busy) { kv_line_busy = false; kv_idle_at[0] = now; }   // a solo run ended between lines; this tick closes its clock
    for (size_t i = 0; i < n; ++i) {
        const bool busy = batch_on() && i < bs.size() && bs[i].active;
        if (basalt::core::kv_policy_decide(now_s, busy, kv_secs(kv_idle_at[i]), conf, kv_pol[i]) != basalt::core::KvAction::Retire)
            continue;
        uint64_t freed = 0;
        std::string err;
        if (!slot_session((int) i).kv_retire(o.kv_keep_warm_cells, freed, err)) {
            basalt::log::info("basalt serve", "kv retire: lane %zu refused (%s); the next tick tries again", i, err.c_str());
            continue;
        }
        basalt::core::kv_policy_note_retired(now_s, kv_pol[i]);
        if (freed > 0)
            basalt::log::info("basalt serve", "kv retire: lane %zu handed %.2f GiB of cache to host backing (%lld warm cells kept)",
                              i, (double) freed / (double) (1ull << 30), (long long) o.kv_keep_warm_cells);
    }
}

void Runtime::kv_note_activity(const std::string& line) {
    if (o.kv_idle_demote_seconds <= 0) return;
    // The line carries the reader-rewritten forms: GEN names the solo lane; SGEN/BGEN (and their I variants)
    // name their slot. A retired lane's wake is the ARRIVAL of its request: the lane that gets work starts its
    // flap guard and clock — never lane 0 by default.
    size_t lane;
    if (line.rfind("GEN", 0) == 0) lane = 0;   // GEN and GENI are the solo lane, whatever follows the verb
    else if (line.rfind("SGEN", 0) == 0 || line.rfind("BGEN", 0) == 0) {
        const char* p = line.c_str() + 4;
        if (*p == 'I') ++p;   // SGENI/BGENI: the interleaved variants carry the same slot field
        lane = (size_t) std::atoi(p);
    } else return;
    if (lane >= kv_pol.size()) return;
    if (lane == 0 && !batch_on()) kv_line_busy = true;   // the solo clock closes its busy edge at the next idle tick
    if (kv_pol[lane].retired)
        basalt::core::kv_policy_note_wake(kv_secs(Clock::now()), kv_conf(o), kv_pol[lane]);
}

void Runtime::kv_stamp_idle(size_t lane) {
    if (o.kv_idle_demote_seconds <= 0 || lane >= kv_idle_at.size()) return;
    kv_idle_at[lane] = Clock::now();
}

}  // namespace basalt::program::serve
