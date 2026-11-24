// src/core/vmm.cpp - the driver's virtual memory functions through the runtime's entry points (no link against the
// driver library).  Read the header first.
#include "basalt/core/vmm.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <array>

namespace basalt::core::vmm {
namespace {
struct Fns {
    CUresult (CUDAAPI* device_get)(CUdevice*, int) = nullptr;
    CUresult (CUDAAPI* attribute)(int*, CUdevice_attribute, CUdevice) = nullptr;
    CUresult (CUDAAPI* granularity)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) = nullptr;
    CUresult (CUDAAPI* reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (CUDAAPI* address_free)(CUdeviceptr, size_t) = nullptr;
    CUresult (CUDAAPI* create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*,
                               unsigned long long) = nullptr;
    CUresult (CUDAAPI* release)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (CUDAAPI* map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    CUresult (CUDAAPI* unmap)(CUdeviceptr, size_t) = nullptr;
    CUresult (CUDAAPI* set_access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
    bool ok = false;
};

template <class F> bool entry(const char* name, F& f) {
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

const Fns& fns() {
    static const Fns v = [] {
        Fns x;
        x.ok = entry("cuDeviceGet", x.device_get) && entry("cuDeviceGetAttribute", x.attribute) &&
               entry("cuMemGetAllocationGranularity", x.granularity) && entry("cuMemAddressReserve", x.reserve) &&
               entry("cuMemAddressFree", x.address_free) && entry("cuMemCreate", x.create) &&
               entry("cuMemRelease", x.release) && entry("cuMemMap", x.map) && entry("cuMemUnmap", x.unmap) &&
               entry("cuMemSetAccess", x.set_access);
        return x;
    }();
    return v;
}

int current_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

CUmemAllocationProp device_prop(int dev) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    return prop;
}

// The fail-shim: one-shot arms per op (an op armed twice fails twice), plus positional skips
// (fail_after lets a row pass the earlier same-op calls of a procedure and fail a later one).
std::array<unsigned int, 9> armed{};
std::array<unsigned int, 9> skip_arm{};
unsigned int& arm_slot(FailOp op) { return armed[(size_t) op]; }
bool take_arm(FailOp op) {
    if (skip_arm[(size_t) op] > 0) { --skip_arm[(size_t) op]; return false; }
    if (arm_slot(op) == 0) return false;
    --arm_slot(op);
    return true;
}
}  // namespace

bool supported() {
    const Fns& v = fns();
    int supported = 0;
    CUdevice cu = 0;
    return v.ok && v.device_get(&cu, current_device()) == CUDA_SUCCESS &&
           v.attribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu) == CUDA_SUCCESS &&
           supported != 0;
}

uint64_t granularity() {
    const Fns& v = fns();
    if (!v.ok) return 0;
    const CUmemAllocationProp prop = device_prop(current_device());
    size_t g = 0;
    if (v.granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) return 0;
    return (uint64_t) g;
}

namespace {
// Separate backing allocations avoid the driver's single-allocation limit while preserving one address range.
constexpr uint64_t kHostPieceBytes = 2ull << 30;

bool host_prop(CUmemAllocationProp& prop, size_t& g) {
    const Fns& v = fns();
    int dev = 0, supported = 0, node = -1;
    CUdevice cu;
    if (!v.ok || cudaGetDevice(&dev) != cudaSuccess || v.device_get(&cu, dev) != CUDA_SUCCESS ||
        v.attribute(&supported, CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu) != CUDA_SUCCESS ||
        !supported || v.attribute(&node, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, cu) != CUDA_SUCCESS) return false;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location = {CU_MEM_LOCATION_TYPE_HOST_NUMA, node < 0 ? 0 : node};
    return v.granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS && g;
}
}

uint64_t host_granularity() {
    CUmemAllocationProp prop{};
    size_t g = 0;
    return host_prop(prop, g) ? g : 0;
}

uint8_t* alloc_host(uint64_t bytes, uint64_t& mapped) {
    mapped = 0;
    const Fns& v = fns();
    CUmemAllocationProp prop{};
    size_t g = 0;
    if (!bytes || !host_prop(prop, g) || kHostPieceBytes % g != 0) return nullptr;
    const int dev = current_device();
    if (bytes > SIZE_MAX - (g - 1)) return nullptr;
    size_t n = (bytes + g - 1) / g * g;
    CUdeviceptr p = 0;
    const auto refused = [&](const char* op, CUresult error, uint64_t off) {
        basalt::log::warn("basalt memory", "host VMM %s at %.2f of %.2f GiB on CUDA%d failed (driver code %d)",
                          op, off / 1073741824.0, n / 1073741824.0, dev, (int) error);
    };
    CUresult error = v.reserve(&p, n, 0, 0, 0);
    if (error != CUDA_SUCCESS) { refused("reserve", error, 0); return nullptr; }
    uint64_t off = 0;
    for (; off < n;) {
        const uint64_t piece = std::min<uint64_t>(kHostPieceBytes, n - off);
        CUmemGenericAllocationHandle h;
        const char* op = "create";
        error = v.create(&h, piece, &prop, 0);
        if (error == CUDA_SUCCESS) {
            op = "map";
            error = v.map(p + off, piece, 0, h, 0);
            v.release(h);   // a successful mapping owns the physical allocation
            if (error == CUDA_SUCCESS) {
                CUmemAccessDesc a[] = {{prop.location, CU_MEM_ACCESS_FLAGS_PROT_READWRITE},
                    {{CU_MEM_LOCATION_TYPE_DEVICE, dev}, CU_MEM_ACCESS_FLAGS_PROT_READWRITE}};
                op = "access";
                error = v.set_access(p + off, piece, a, 2);
                if (error == CUDA_SUCCESS) { off += piece; continue; }
                v.unmap(p + off, piece);
            }
        }
        refused(op, error, off);
        for (uint64_t at = 0; at < off; at += kHostPieceBytes)
            v.unmap(p + at, std::min<uint64_t>(kHostPieceBytes, off - at));
        v.address_free(p, n);
        return nullptr;
    }
    mapped = n;
    return reinterpret_cast<uint8_t*>(p);
}

void free_host(uint8_t* ptr, uint64_t mapped) {
    if (!ptr) return;
    auto p = reinterpret_cast<CUdeviceptr>(ptr);
    for (uint64_t off = 0; off < mapped; off += kHostPieceBytes)
        fns().unmap(p + off, std::min<uint64_t>(kHostPieceBytes, mapped - off));
    fns().address_free(p, mapped);
}

uint8_t* reserve(uint64_t bytes) {
    const Fns& v = fns();
    CUdeviceptr va = 0;
    if (!v.ok || v.reserve(&va, (size_t) bytes, 0, 0, 0) != CUDA_SUCCESS) return nullptr;
    return reinterpret_cast<uint8_t*>(va);
}

void release_range(uint8_t* va, uint64_t bytes) {
    const Fns& v = fns();
    if (v.ok && va != nullptr && bytes > 0) v.address_free(reinterpret_cast<CUdeviceptr>(va), (size_t) bytes);
}

unsigned long long map(uint8_t* va, uint64_t bytes) {
    const Fns& v = fns();
    if (!v.ok) return 0;
    if (take_arm(FailOp::Map)) return 0;
    const int dev = current_device();
    const CUmemAllocationProp prop = device_prop(dev);
    const CUdeviceptr p = reinterpret_cast<CUdeviceptr>(va);
    CUmemGenericAllocationHandle h = 0;
    if (v.create(&h, (size_t) bytes, &prop, 0) != CUDA_SUCCESS) return 0;
    if (v.map(p, (size_t) bytes, 0, h, 0) != CUDA_SUCCESS) {
        v.release(h);
        return 0;
    }
    if (take_arm(FailOp::SetAccess)) {
        v.unmap(p, (size_t) bytes);
        v.release(h);
        return 0;
    }
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (v.set_access(p, (size_t) bytes, &access, 1) != CUDA_SUCCESS) {
        v.unmap(p, (size_t) bytes);
        v.release(h);
        return 0;
    }
    return (unsigned long long) h;
}

unsigned long long map_host(uint8_t* va, uint64_t bytes) {
    const Fns& v = fns();
    if (!v.ok || !bytes) return 0;
    if (take_arm(FailOp::Map)) return 0;
    CUmemAllocationProp prop{};
    size_t g = 0;
    if (!host_prop(prop, g) || bytes % g || reinterpret_cast<uintptr_t>(va) % g) return 0;
    const CUdeviceptr p = reinterpret_cast<CUdeviceptr>(va);
    CUmemGenericAllocationHandle h = 0;
    if (v.create(&h, (size_t) bytes, &prop, 0) != CUDA_SUCCESS) return 0;
    if (v.map(p, (size_t) bytes, 0, h, 0) != CUDA_SUCCESS) {
        v.release(h);
        return 0;
    }
    if (take_arm(FailOp::SetAccess)) {
        v.unmap(p, (size_t) bytes);
        v.release(h);
        return 0;
    }
    const int dev = current_device();
    CUmemAccessDesc access[] = {{prop.location, CU_MEM_ACCESS_FLAGS_PROT_READWRITE},
        {{CU_MEM_LOCATION_TYPE_DEVICE, dev}, CU_MEM_ACCESS_FLAGS_PROT_READWRITE}};
    if (v.set_access(p, (size_t) bytes, access, 2) != CUDA_SUCCESS) {
        v.unmap(p, (size_t) bytes);
        v.release(h);
        return 0;
    }
    return (unsigned long long) h;
}

bool unmap(uint8_t* va, uint64_t bytes, unsigned long long handle) {
    const Fns& v = fns();
    if (!v.ok || handle == 0) return false;
    if (take_arm(FailOp::Unmap)) return false;
    const bool a = v.unmap(reinterpret_cast<CUdeviceptr>(va), (size_t) bytes) == CUDA_SUCCESS;
    const bool b = v.release((CUmemGenericAllocationHandle) handle) == CUDA_SUCCESS;
    return a && b;
}

unsigned long long create(uint64_t bytes) {
    const Fns& v = fns();
    if (!v.ok) return 0;
    if (take_arm(FailOp::Create)) return 0;
    const CUmemAllocationProp prop = device_prop(current_device());
    CUmemGenericAllocationHandle h = 0;
    return v.create(&h, (size_t) bytes, &prop, 0) == CUDA_SUCCESS ? (unsigned long long) h : 0;
}

bool map_handle(uint8_t* va, uint64_t bytes, unsigned long long handle) {
    const Fns& v = fns();
    if (!v.ok || handle == 0) return false;
    if (take_arm(FailOp::Map)) return false;
    const CUdeviceptr p = reinterpret_cast<CUdeviceptr>(va);
    // Peek-before-consume for SetAccess: when the access is armed but this call's map never
    // happens (or fails), the access arm must stay pending — else a later cleanup unmap is fed
    // an already-unmapped handle and the driver releases the payload with it (row-killers).
    if (v.map(p, (size_t) bytes, 0, (CUmemGenericAllocationHandle) handle, 0) != CUDA_SUCCESS) { take_arm(FailOp::SetAccess); return false; }
    if (take_arm(FailOp::SetAccess)) return false;
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = current_device();
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // Design f2/L2b: on access failure the handle STAYS mapped-without-access; the calling
    // procedure owns the cleanup unmap (an internal rollback would make the procedure's unmap
    // hit an already-unmapped range and phantom-stick). Shim and real failures behave alike.
    if (v.set_access(p, (size_t) bytes, &access, 1) == CUDA_SUCCESS) return true;
    return false;
}

bool unmap_only(uint8_t* va, uint64_t bytes) {
    const Fns& v = fns();
    if (take_arm(FailOp::UnmapOnly)) return false;  // armed: fails without touching driver state
    if (!v.ok || va == nullptr || bytes == 0) return false;
    return v.unmap(reinterpret_cast<CUdeviceptr>(va), (size_t) bytes) == CUDA_SUCCESS;
}

void release(unsigned long long handle) {
    const Fns& v = fns();
    if (take_arm(FailOp::Release)) return;
    if (v.ok && handle != 0) v.release((CUmemGenericAllocationHandle) handle);
}

void fail_next(FailOp op) {
    ++arm_slot(op);
}

void fail_after(FailOp op, unsigned skip, unsigned count) {
    skip_arm[(size_t) op] += skip;
    arm_slot(op) += count;
}

bool fail_armed(FailOp op) {
    // "The next call of this op fails" = the call will CONSUME an arm, not a positional skip:
    // with a skip pending the call passes and the armed failure belongs to a later same-op call.
    return arm_slot(op) > 0 && skip_arm[(size_t) op] == 0;
}

void reset_arms() {
    armed.fill(0);
    skip_arm.fill(0);
}

unsigned long long create_host(uint64_t bytes) {
    const Fns& v = fns();
    if (!v.ok || !bytes) return 0;
    if (take_arm(FailOp::CreateHost)) return 0;
    CUmemAllocationProp prop{};
    size_t g = 0;
    if (!host_prop(prop, g)) return 0;
    CUmemGenericAllocationHandle h = 0;
    return v.create(&h, (size_t) bytes, &prop, 0) == CUDA_SUCCESS ? (unsigned long long) h : 0;
}

bool map_host_handle(uint8_t* va, uint64_t bytes, unsigned long long handle) {
    const Fns& v = fns();
    if (!v.ok || !bytes || handle == 0) return false;
    if (take_arm(FailOp::Map)) return false;
    CUmemAllocationProp prop{};
    size_t g = 0;
    if (!host_prop(prop, g) || bytes % g || reinterpret_cast<uintptr_t>(va) % g) return false;
    const CUdeviceptr p = reinterpret_cast<CUdeviceptr>(va);
    // Peek-before-consume for SetAccess: when the access is armed but this call's map never
    // happens (or fails), the access arm must stay pending — else a later cleanup unmap is fed
    // an already-unmapped handle and the driver releases the payload with it (row-killers).
    if (v.map(p, (size_t) bytes, 0, (CUmemGenericAllocationHandle) handle, 0) != CUDA_SUCCESS) { take_arm(FailOp::SetAccess); return false; }
    if (take_arm(FailOp::SetAccess)) return false;
    const int dev = current_device();
    CUmemAccessDesc access[] = {{prop.location, CU_MEM_ACCESS_FLAGS_PROT_READWRITE},
        {{CU_MEM_LOCATION_TYPE_DEVICE, dev}, CU_MEM_ACCESS_FLAGS_PROT_READWRITE}};
    if (v.set_access(p, (size_t) bytes, access, 2) == CUDA_SUCCESS) return true;
    return false;  // mapped-without-access (L2b); the procedure owns the unmap
}

bool set_access_device(uint8_t* va, uint64_t bytes) {
    const Fns& v = fns();
    if (!v.ok || !va || !bytes) return false;
    if (take_arm(FailOp::SetAccess)) return false;
    const int dev = current_device();
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return v.set_access(reinterpret_cast<CUdeviceptr>(va), (size_t) bytes, &access, 1) == CUDA_SUCCESS;
}

bool set_access_host(uint8_t* va, uint64_t bytes) {
    const Fns& v = fns();
    if (!v.ok || !va || !bytes) return false;
    if (take_arm(FailOp::SetAccess)) return false;
    CUmemAllocationProp prop{};
    size_t g = 0;
    if (!host_prop(prop, g)) return false;
    const int dev = current_device();
    CUmemAccessDesc access[] = {{prop.location, CU_MEM_ACCESS_FLAGS_PROT_READWRITE},
        {{CU_MEM_LOCATION_TYPE_DEVICE, dev}, CU_MEM_ACCESS_FLAGS_PROT_READWRITE}};
    return v.set_access(reinterpret_cast<CUdeviceptr>(va), (size_t) bytes, access, 2) == CUDA_SUCCESS;
}

// The legacy cuMemcpy* driver symbols live in libcuda and are NOT resolvable through
// cudaGetDriverEntryPointByVersion (cudaErrorStubLibrary) — the runtime cudaMemcpy is this engine's
// synchronous-copy idiom (elastic_array already cudaMemsets mapped VAs), so the copies go through it.
bool copy_h2d(uint8_t* dst_dev, const uint8_t* src_host, uint64_t bytes) {
    const Fns& v = fns();
    if (!v.ok || !dst_dev || !src_host || !bytes) return false;
    if (take_arm(FailOp::CopyH2D)) return false;
    return cudaMemcpy(dst_dev, src_host, (size_t) bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool copy_d2h(uint8_t* dst_host, const uint8_t* src_dev, uint64_t bytes) {
    const Fns& v = fns();
    if (!v.ok || !dst_host || !src_dev || !bytes) return false;
    if (take_arm(FailOp::CopyD2H)) return false;
    return cudaMemcpy(dst_host, src_dev, (size_t) bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace basalt::core::vmm
