// src/core/elastic_array.cpp - see elastic_array.hpp.
#include "basalt/core/elastic_array.hpp"

#include "basalt/core/device.hpp"
#include "basalt/core/vmm.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace basalt::core {

ElasticArray::~ElasticArray() {
    if (!base_) return;
    if (plain_) {
        cudaFree(base_);
        return;
    }
    cudaDeviceSynchronize();
    for (const Extent& e : extents_) vmm::unmap(base_ + e.off, e.bytes, e.handle);
    vmm::release_range(base_, reserved_);
}

bool ElasticArray::open(uint64_t bytes, std::string& err) {
    if (bytes == 0) bytes = 1;
    if (vmm::supported()) {
        gran_ = vmm::granularity();
        if (gran_) {
            reserved_ = (bytes + gran_ - 1) / gran_ * gran_;
            base_ = vmm::reserve(reserved_);
            if (base_) return true;
        }
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        err = "out of device memory for the array";
        return false;
    }
    cudaMemset(p, 0, bytes);
    base_ = (uint8_t*) p;
    reserved_ = backed_ = bytes;
    plain_ = true;
    return true;
}

// map only to the requested bound, rounded UP to the granule the driver requires: never more
// than one granule past the bound (the plan charges the raw slice, so keep the overshoot small)
uint64_t ElasticArray::growth(uint64_t bytes) const {
    if (bytes <= backed_ || bytes > reserved_) return 0;
    return std::min((bytes + gran_ - 1) / gran_ * gran_, reserved_) - backed_;
}

bool ElasticArray::ensure(uint64_t bytes, uint64_t keep_free, std::string& err) {
    if (bytes <= backed_) return true;
    if (bytes > reserved_) {
        err = "the array is full";
        return false;
    }
    // past the host boundary the bytes live in RAM: nothing left for the device to map there
    if (host_start_ != kNoHost) {
        if (bytes > host_start_) bytes = host_start_;
        if (bytes <= backed_) return true;
    }
    const uint64_t n = growth(bytes);
    const uint64_t want = backed_ + n;
    if (keep_free > 0) {
        uint64_t free_b = 0, total_b = 0;
        std::string why;
        if (!device_memory(-1, free_b, total_b, why)) {
            err = "cannot read the free device memory: " + why;
            return false;
        }
        if (free_b < n + keep_free) {
            err = "the VRAM reserve of " + std::to_string(keep_free >> 20) + " MiB stays free";
            return false;
        }
    }
    const unsigned long long h = vmm::map(base_ + backed_, n);
    if (!h) {
        err = "out of device memory for the array";
        return false;
    }
    cudaMemset(base_ + backed_, 0, n);
    extents_.push_back({backed_, n, h});
    backed_ = want;
    return true;
}

bool ElasticArray::back_host(uint64_t bytes, std::string& err) {
    if (plain_ || !base_) {
        err = "host backing needs virtual memory management";
        return false;
    }
    if (bytes <= backed_) return true;
    if (bytes > reserved_) {
        err = "the array is full";
        return false;
    }
    const uint64_t hg = vmm::host_granularity();
    if (hg == 0) {
        err = "host backing needs virtual memory management";
        return false;
    }
    // the host tail must be host-granularity aligned: pad the end up (into the reservation) so `map_host` fits
    if (bytes % hg) {
        const uint64_t padded = (bytes + hg - 1) / hg * hg;
        if (padded <= reserved_) {
            bytes = padded;
        } else if (reserved_ % hg == 0) {
            bytes = reserved_;
        } else {
            err = "the host tail cannot be padded to the host granularity inside the reservation";
            return false;
        }
    }
    if (host_start_ == kNoHost) host_start_ = backed_;
    const uint64_t kHostPiece = 2ull << 30;   // the driver's single host allocation limit
    uint64_t at = backed_;
    while (at < bytes) {
        const uint64_t n = std::min(kHostPiece, bytes - at);
        const unsigned long long h = vmm::map_host(base_ + at, n);
        if (!h) {
            err = "out of host memory for the streamed range";   // what was mapped before stays backed
            return false;
        }
        extents_.push_back({at, n, h, true});
        backed_ = at + n;
        at = backed_;
    }
    return true;
}

void ElasticArray::trim(uint64_t bytes) {
    if (plain_ || !base_ || extents_.empty() || extents_.back().off < bytes) return;
    cudaDeviceSynchronize();
    while (!extents_.empty() && !extents_.back().host && extents_.back().off >= bytes) {   // host bytes stay: RAM is not the scarce one
        const Extent& e = extents_.back();
        vmm::unmap(base_ + e.off, e.bytes, e.handle);
        extents_.pop_back();
    }
    backed_ = extents_.empty() ? 0 : extents_.back().off + extents_.back().bytes;
}

}  // namespace basalt::core
