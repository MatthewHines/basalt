// tests/core/vmm_fail_test.cpp - the vmm fail-shim self-tests and the T9c two-whole-maps probe (S1 re-home slice).
//
// The shim arms a one-shot failure per op; each check asserts the arm is consumed and that the next call
// succeeds (L2b no-op-on-fail).  Exit 77 without VMM.  The T9c probe records (does not verify) whether two
// whole maps of one handle into two separate reservations are allowed: the design leaves L4 unverified.
#include "basalt/core/vmm.hpp"

#include <cuda_runtime.h>
#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}
void note(const char* what) { std::printf("note %s\n", what); }

// ---- T9c probe: raw driver lookups (same idiom as src/core/vmm.cpp) ------------------
template <class F>
bool entry(const char* name, F& f) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    f = reinterpret_cast<F>(p);
    return true;
}

struct Probe {
    CUresult (CUDAAPI* device_get)(CUdevice*, int) = nullptr;
    CUresult (CUDAAPI* attribute)(int*, CUdevice_attribute, CUdevice) = nullptr;
    CUresult (CUDAAPI* reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (CUDAAPI* address_free)(CUdeviceptr, size_t) = nullptr;
    CUresult (CUDAAPI* create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) = nullptr;
    CUresult (CUDAAPI* release)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (CUDAAPI* map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    CUresult (CUDAAPI* unmap)(CUdeviceptr, size_t) = nullptr;

    bool ok = false;
};

const Probe& probe() {
    static const Probe v = [] {
        Probe x;
        x.ok = entry("cuDeviceGet", x.device_get) && entry("cuDeviceGetAttribute", x.attribute) &&
               entry("cuMemAddressReserve", x.reserve) && entry("cuMemAddressFree", x.address_free) &&
               entry("cuMemCreate", x.create) && entry("cuMemRelease", x.release) &&
               entry("cuMemMap", x.map) && entry("cuMemUnmap", x.unmap);
        return x;
    }();
    return v;
}

void t9c_probe(uint64_t g) {
    const Probe& p = probe();
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    CUdevice cu = 0;
    if (!p.ok || p.device_get == nullptr || p.attribute == nullptr || p.reserve == nullptr ||
        p.create == nullptr || p.map == nullptr || p.release == nullptr || p.unmap == nullptr ||
        p.device_get(&cu, dev) != CUDA_SUCCESS) {
        std::printf("t9c step1 rc=-1 va=0\nt9c step2 rc=-1 va=0\nt9c verdict refused second_rc=-1\n");
        return;
    }
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;

    CUdeviceptr R1 = 0, R2 = 0;
    CUresult rc1 = p.reserve(&R1, (size_t) g, 0, 0, 0);
    CUresult rc2 = p.reserve(&R2, (size_t) g, 0, 0, 0);
    if (rc1 != CUDA_SUCCESS || rc2 != CUDA_SUCCESS) {
        std::printf("t9c step1 rc=%d va=%llu\nt9c step2 rc=%d va=%llu\nt9c verdict refused second_rc=%d\n",
                    (int) rc1, (unsigned long long) R1, (int) rc2, (unsigned long long) R2, (int) rc2);
        return;
    }
    CUmemGenericAllocationHandle h = 0;
    CUresult rc3 = p.create(&h, (size_t) g, &prop, 0);
    if (rc3 != CUDA_SUCCESS) {
        std::printf("t9c step1 rc=%d va=%llu\nt9c step2 rc=%d va=%llu\nt9c verdict refused second_rc=%d\n",
                    (int) rc3, (unsigned long long) R1, (int) rc3, (unsigned long long) R2, (int) rc3);
        p.address_free(R1, (size_t) g);
        p.address_free(R2, (size_t) g);
        return;
    }
    CUresult rc_map1 = p.map(R1, (size_t) g, 0, h, 0);
    CUresult rc_map2 = p.map(R2, (size_t) g, 0, h, 0);
    std::printf("t9c step1 rc=%d va=%llu\n", (int) rc_map1, (unsigned long long) R1);
    std::printf("t9c step2 rc=%d va=%llu\n", (int) rc_map2, (unsigned long long) R2);
    if (rc_map1 == CUDA_SUCCESS && rc_map2 == CUDA_SUCCESS) {
        std::printf("t9c verdict mapped_twice\n");
        p.unmap(R1, (size_t) g);
        p.unmap(R2, (size_t) g);
    } else {
        std::printf("t9c verdict refused second_rc=%d\n", (int) rc_map2);
        if (rc_map1 == CUDA_SUCCESS) p.unmap(R1, (size_t) g);
        if (rc_map2 == CUDA_SUCCESS) p.unmap(R2, (size_t) g);
    }
    p.release(h);
    p.address_free(R1, (size_t) g);
    p.address_free(R2, (size_t) g);
}

}  // namespace

int main() {
    using namespace basalt::core::vmm;
    reset_arms();
    if (!supported()) {
        std::puts("vmm not supported: skipped");
        return 77;
    }
    const uint64_t g = granularity();
    if (g == 0) {
        std::puts("vmm granularity unavailable: skipped");
        return 77;
    }

    // --- shim: create ---------------------------------------------------
    fail_next(FailOp::Create);
    check(fail_armed(FailOp::Create), "create arm pending");
    check(create(g) == 0, "armed create returns 0");
    check(!fail_armed(FailOp::Create), "create arm consumed");
    unsigned long long handle = create(g);
    check(handle != 0, "next create succeeds");
    release(handle);

    // --- shim: atomic map, Map-fail (nothing left mapped) ---------------
    uint8_t* va = reserve(g);
    check(va != nullptr, "reserve");
    fail_next(FailOp::Map);
    check(map(va, g) == 0, "armed atomic map returns 0");
    check(!fail_armed(FailOp::Map), "map arm consumed");
    auto mapped = map(va, g);
    check(mapped != 0, "real atomic map succeeds (nothing left mapped)");
    check(unmap(va, g, mapped), "real unmap succeeds");
    release_range(va, g);

    // --- shim: atomic map, SetAccess-fail (unwinds fully) ---------------
    va = reserve(g);
    check(va != nullptr, "reserve");
    fail_next(FailOp::SetAccess);
    check(map(va, g) == 0, "armed atomic-map set-access returns 0");
    check(!fail_armed(FailOp::SetAccess), "set-access arm consumed");
    mapped = map(va, g);
    check(mapped != 0, "real atomic map succeeds (nothing left mapped)");
    check(unmap(va, g, mapped), "real unmap succeeds");
    release_range(va, g);

    // --- shim: composed map_handle, Map-fail ------------------------------
    va = reserve(g);
    check(va != nullptr, "reserve");
    handle = create(g);
    check(handle != 0, "create handle");
    fail_next(FailOp::Map);
    check(map_handle(va, g, handle) == MapOutcome::MapFail, "armed composed map yields MapFail");
    check(!fail_armed(FailOp::Map), "composed map arm consumed");
    check(map_handle(va, g, handle) == MapOutcome::Ok, "real composed map succeeds (nothing left)");
    check(unmap(va, g, handle), "composed map unmap succeeds");
    release_range(va, g);

    // --- shim: composed map_handle, SetAccess-fail (S4b generator) ---------
    va = reserve(g);
    check(va != nullptr, "reserve");
    handle = create(g);
    check(handle != 0, "create handle");
    fail_next(FailOp::SetAccess);
    check(map_handle(va, g, handle) == MapOutcome::AccessFail, "armed composed set-access yields AccessFail");
    check(!fail_armed(FailOp::SetAccess), "composed set-access arm consumed");
    check(unmap(va, g, handle), "the S4b-generator handle stays mapped (unmap succeeds)");
    release_range(va, g);

    // --- shim: positional fail_after + skip-aware peek (review F4) ------------
    va = reserve(g);
    check(va != nullptr, "reserve");
    handle = create(g);
    check(handle != 0, "create handle");
    fail_after(FailOp::Map, 1, 1);
    check(!fail_armed(FailOp::Map), "peek is skip-aware: next call will pass");
    check(map_handle(va, g, handle) == MapOutcome::Ok, "fail_after: the skipped call really maps");
    check(unmap_only(va, g), "cleanup the skipped mapping");
    check(fail_armed(FailOp::Map), "arm is now the next call");
    check(map_handle(va, g, handle) == MapOutcome::MapFail, "fail_after: the armed call fails");
    check(!fail_armed(FailOp::Map), "arm consumed exactly once");
    release(handle);
    release_range(va, g);

    // --- shim: unmap, no-op on fail ----------------------------------------
    va = reserve(g);
    check(va != nullptr, "reserve");
    mapped = map(va, g);
    check(mapped != 0, "real atomic map");
    fail_next(FailOp::Unmap);
    check(unmap(va, g, mapped) == false, "armed unmap returns false");
    check(!fail_armed(FailOp::Unmap), "unmap arm consumed");
    check(set_access_device(va, g) == true, "mapping retained after armed-unmap fail");
    check(unmap(va, g, mapped) == true, "real unmap succeeds");
    release_range(va, g);

    // --- shim: unmap_only, no-op on fail -----------------------------------
    va = reserve(g);
    check(va != nullptr, "reserve");
    mapped = map(va, g);
    check(mapped != 0, "real atomic map");
    fail_next(FailOp::UnmapOnly);
    unmap_only(va, g);  // armed: returns false, driver untouched (arm consumed)
    check(!fail_armed(FailOp::UnmapOnly), "unmap-only arm consumed");
    check(set_access_device(va, g) == true, "mapping retained after armed unmap-only");
    unmap_only(va, g);  // real: the mapping is consumed (and the handle's mapping with it)
    release(mapped);    // L3-valid: the handle is mapped nowhere now
    release_range(va, g);

    // --- shim: release, no-op on fail ---------------------------------------
    handle = create(g);
    check(handle != 0, "create handle");
    fail_next(FailOp::Release);
    release(handle);     // armed: no-op (void)
    check(!fail_armed(FailOp::Release), "release arm consumed");
    release(handle);     // real release (the handle is still owned)

    // --- shim: set_access_device gate ---------------------------------------
    va = reserve(g);
    check(va != nullptr, "reserve");
    mapped = map(va, g);
    check(mapped != 0, "real atomic map");
    fail_next(FailOp::SetAccess);
    check(set_access_device(va, g) == false, "armed set_access_device returns false");
    check(!fail_armed(FailOp::SetAccess), "set_access_device arm consumed");
    check(set_access_device(va, g) == true, "real set_access_device succeeds");
    check(unmap(va, g, mapped), "cleanup unmap");
    release_range(va, g);

    // --- shim: copies + gates (round trip) ----------------------------------
    va = reserve(g);
    check(va != nullptr, "reserve");
    mapped = map(va, g);
    check(mapped != 0, "real atomic map");
    std::vector<uint8_t> src(g);
    for (uint64_t i = 0; i < g; ++i) src[i] = (uint8_t) (i * 131u >> 4);
    fail_next(FailOp::CopyH2D);
    check(copy_h2d(va, src.data(), g) == false, "armed copy_h2d returns false");
    check(!fail_armed(FailOp::CopyH2D), "copy-h2d arm consumed");
    check(copy_h2d(va, src.data(), g) == true, "real copy_h2d succeeds");
    std::vector<uint8_t> dst(g, 0);
    check(copy_d2h(dst.data(), va, g) == true, "copy_d2h succeeds");
    check(dst == src, "copy round trip keeps the bytes");
    fail_next(FailOp::CopyD2H);
    check(copy_d2h(dst.data(), va, g) == false, "armed copy_d2h returns false");
    check(!fail_armed(FailOp::CopyD2H), "copy-d2h arm consumed");
    check(copy_d2h(dst.data(), va, g) == true, "real copy_d2h succeeds");
    check(unmap(va, g, mapped), "cleanup unmap");
    release_range(va, g);

    // --- host side: create_host / map_host_handle / set_access_host ----------
    const uint64_t g_host = host_granularity();
    if (g_host == 0) {
        std::puts("host VMM unavailable: skipped");
    } else {
        const uint64_t bytes = g_host;
        fail_next(FailOp::CreateHost);
        check(create_host(bytes) == 0, "armed create_host returns 0");
        check(!fail_armed(FailOp::CreateHost), "create_host arm consumed");
        handle = create_host(bytes);
        check(handle != 0, "next create_host succeeds");
        if (handle != 0) {
            uint8_t* h_va = reserve(bytes);
            if (h_va == nullptr) {
                note("host reservation failed: skipped map_host_handle tests");
                release(handle);
            } else if ((reinterpret_cast<uintptr_t>(h_va) % g_host) != 0) {
                note("host reservation VA not host-aligned: skipped map_host_handle tests");
                release_range(h_va, bytes);
                release(handle);
            } else {
                const auto h1 = handle;
                fail_next(FailOp::Map);
                check(map_host_handle(h_va, bytes, h1) == MapOutcome::MapFail, "armed map_host_handle yields MapFail");
                check(!fail_armed(FailOp::Map), "host map arm consumed");
                check(map_host_handle(h_va, bytes, h1) == MapOutcome::Ok, "real map_host_handle succeeds");
                check(set_access_host(h_va, bytes) == true, "host set_access succeeds");
                check(unmap(h_va, bytes, h1), "cleanup unmap (host handle)");  // h1 released by unmap

                // composed set-access fail leaves the handle mapped (the S4b generator)
                handle = create_host(bytes);
                check(handle != 0, "create host handle again");
                if (handle != 0) {
                    const auto h2 = handle;
                    fail_next(FailOp::SetAccess);
                    check(map_host_handle(h_va, bytes, h2) == MapOutcome::AccessFail, "armed composed host set-access yields AccessFail");
                    check(!fail_armed(FailOp::SetAccess), "host composed set-access arm consumed");
                    check(unmap(h_va, bytes, h2), "the S4b-generator host handle stays mapped (unmap succeeds)");
                }
                release_range(h_va, bytes);
                handle = 0;
            }
        }
    }

    // --- T9c probe: two whole maps of one handle ----------------------------
    t9c_probe(g);

    std::printf(fails == 0 ? "vmm_fail_test: ok\n" : "vmm_fail_test: %d failures\n", fails);
    return fails == 0 ? 0 : 1;
}
