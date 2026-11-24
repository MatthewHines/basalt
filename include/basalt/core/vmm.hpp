// include/basalt/core/vmm.hpp - the driver's virtual memory management, looked up through the runtime.
//
// One reserved address range, backed piece by piece by physical allocations that can be unmapped and mapped again
// while every address in the range stays the same.  The expert cache's segments and the elastic KV pools both use
// it: captured graphs and kernel arguments hold addresses into these ranges, so only the backing moves.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::core::vmm {

/// Whether the current device supports virtual memory management (and the driver entry points were found).
bool supported();
/// The allocation granularity of the current device (every size and offset below is a multiple of it); 0 on failure.
uint64_t granularity();
/// The minimum host allocation granularity on the current device, or 0 without host VMM.
uint64_t host_granularity();
/// CUDA-pinned system memory accessible to the CPU and current GPU; other GPUs may copy through host endpoints.
/// Returns the padded mapping size for free_host; no portable registration is involved.
uint8_t* alloc_host(uint64_t bytes, uint64_t& mapped);
void free_host(uint8_t* ptr, uint64_t mapped);
/// Reserves `bytes` of address space (a multiple of the granularity); null on failure.
uint8_t* reserve(uint64_t bytes);
void release_range(uint8_t* va, uint64_t bytes);
/// Backs `[va, va + bytes)` with a new physical allocation on the current device, readable and writable by it.
/// Returns the handle (0 on failure: nothing is left mapped).
unsigned long long map(uint8_t* va, uint64_t bytes);
/// Backs `[va, va + bytes)` with a new physical allocation in host NUMA memory: the bytes live in RAM and both the
/// CPU and the current device reach them at `va` (the GPU over the bus).  `va` and `bytes` must fit the host
/// allocation's granularity.  Returns the handle (0 on failure: nothing is left mapped).
unsigned long long map_host(uint8_t* va, uint64_t bytes);
/// Unmaps what `map` mapped there and frees its physical memory.  The caller has made sure no work reads it.
bool unmap(uint8_t* va, uint64_t bytes, unsigned long long handle);
/// A physical allocation of `bytes` on the current device, mapped nowhere yet (0 on failure): one backing that
/// `map_handle` can put under several ranges at once, so they share its bytes.  `release` drops the caller's reference;
/// the memory is freed once every mapping of it is gone too.
unsigned long long create(uint64_t bytes);
/// Maps the first `bytes` of `handle` at `va`, readable and writable by the current device.
bool map_handle(uint8_t* va, uint64_t bytes, unsigned long long handle);
/// Unmaps `[va, va + bytes)` without releasing any handle (the mapping's reference goes with it).
/// True when the range really is unmapped; false on driver failure or when fail-shim-armed (then
/// nothing is touched). Callers that retry — the elastic-array unmap brackets — need the result.
bool unmap_only(uint8_t* va, uint64_t bytes);
void release(unsigned long long handle);

/// The fail-shim's one-shot operations.  Arming a matching op makes the next call to the wrapper that
/// performs that operation fail exactly once (consumed on the call); a second arm repeats the failure on
/// the next call.  An armed failure touches no driver state except the simulated unwind the design pins
/// for the map+setaccess pair.
enum class FailOp { Create, CreateHost, Map, SetAccess, Unmap, UnmapOnly, Release, CopyH2D, CopyD2H };
/// Arm a one-shot failure on the next matching wrapper call (same op may be armed more than once).
void fail_next(FailOp op);
/// Positional arming: the next `skip` matching calls SUCCEED, then the following `count` fail.
/// (Rows like "f fails, b succeeds" need to skip the earlier same-op calls in the same procedure.)
void fail_after(FailOp op, unsigned skip, unsigned count = 1);
/// Peek at whether an arm is pending (tests assert one-shot consumption with this).
bool fail_armed(FailOp op);
/// Drop every pending arm (test hygiene: rows must not inherit another row's unconsumed arms).
void reset_arms();

/// A physical allocation of `bytes` in host NUMA memory, mapped nowhere (0 on failure).
unsigned long long create_host(uint64_t bytes);
/// Maps `[va, va + bytes)` with the host canonical access ({HOST_NUMA rw, DEVICE rw}); false on failure.
bool map_host_handle(uint8_t* va, uint64_t bytes, unsigned long long handle);
/// Re-run the device canonical access over `[va, va + bytes)` (S4b repair); false on failure.
bool set_access_device(uint8_t* va, uint64_t bytes);
/// Re-run the host canonical access over `[va, va + bytes)` (S4b repair); false on failure.
bool set_access_host(uint8_t* va, uint64_t bytes);
/// Synchronous host-to-device copy of `bytes`; false on failure.
bool copy_h2d(uint8_t* dst_dev, const uint8_t* src_host, uint64_t bytes);
/// Synchronous device-to-host copy of `bytes`; false on failure.
bool copy_d2h(uint8_t* dst_host, const uint8_t* src_dev, uint64_t bytes);

}  // namespace basalt::core::vmm
