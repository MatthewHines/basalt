// include/basalt/core/elastic_array.hpp - a device array that reserves its whole address range up front and backs it with
// physical memory as it is used. The address never moves, so kernel arguments and captured graphs stay valid.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

class ElasticArray {
public:
    ElasticArray() = default;
    ~ElasticArray();
    ElasticArray(const ElasticArray&) = delete;
    ElasticArray& operator=(const ElasticArray&) = delete;

    /// Reserves `bytes` of address space. Without virtual memory management the whole array is allocated at once.
    bool open(uint64_t bytes, std::string& err);
    /// Backs at least the first `bytes` bytes (rounded up to the allocation granularity; the new bytes are zeroed).
    /// False with `err` when the device has no memory for them with `keep_free` left; what was backed stays.
    /// Past the host boundary set by `back_host`, the bytes are already backed in RAM and nothing maps.
    bool ensure(uint64_t bytes, uint64_t keep_free, std::string& err);
    /// The device bytes ensure(bytes) would back beyond what is backed now (its rounding included); 0 when it backs nothing.
    uint64_t growth(uint64_t bytes) const;

    /// Backs `[backed_, bytes)` with host NUMA physical memory (the bytes live in RAM; the device reaches them at
    /// the same addresses over the bus).  From then on `ensure` maps no device memory past the first host byte,
    /// and `trim` gives back no host bytes (RAM is not the scarce resource).  False with `err` when the first
    /// extent cannot map; extents mapped before the failure stay.  Zeroes the new bytes.
    bool back_host(uint64_t bytes, std::string& err);
    /// Gives back the whole allocations that lie past `bytes` (the device is synchronised first); the address range stays.
    void trim(uint64_t bytes);
    uint8_t* data() const { return base_; }
    uint64_t backed() const { return backed_; }
    uint64_t capacity() const { return reserved_; }
    /// The first byte backed in host memory, or `kNoHost` while every backed byte is on the device.
    uint64_t host_start() const { return host_start_; }

private:
    static constexpr uint64_t kNoHost = ~0ull;
    struct Extent {
        uint64_t off, bytes;
        unsigned long long handle;
        bool host = false;
    };
    uint8_t* base_ = nullptr;
    uint64_t reserved_ = 0, backed_ = 0, gran_ = 0;
    uint64_t host_start_ = kNoHost;
    bool plain_ = false;   // one cudaMalloc, no virtual memory
    std::vector<Extent> extents_;
};

}  // namespace basalt::core
