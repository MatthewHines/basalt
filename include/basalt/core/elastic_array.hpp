// include/basalt/core/elastic_array.hpp - one reserved address range; the bytes live in RAM or on the device.
//
// Layout (layout law, pass 5.3): one Extent per granule piece, keyed by offset, covering [0, backed_)
// exactly once.  The device extents form the PREFIX (the packed head region) plus promoted HOLES (device-backed
// granules inside a host-backed region); demotes remove holes; no gaps.  I1 = between operations every covered
// byte is mapped exactly once onto MAIN VA; I2 = live-device-handle bytes == device extent bytes (extent-backing
// only, stuck/pending/bytes-holder excluded).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::core {

// The result of a rehome family call: how many pieces were moved (committed) and why it stopped.
struct MoveResult {
    uint64_t moved = 0;
    std::string reason;
};

// In-process extent states (S1-DESIGN.md pass 5.3, state machine).
enum class ExtentState : unsigned int { S0 = 0, S3 = 1, S4b = 2, S4 = 3 };

// The readout shape of a stuck-list entry (the array keeps a parallel, byte-carrying record).
struct StuckEntry {
    uint64_t handle = 0;
    uint8_t* va = nullptr;   // null in the release-pending case
    int kind = 0;            // 0 = unmap-failed, 1 = release-pending
};

class ElasticArray {
public:
    ElasticArray() = default;
    ~ElasticArray();
    ElasticArray(const ElasticArray&) = delete;
    ElasticArray& operator=(const ElasticArray&) = delete;

    static constexpr uint64_t kNoHost = ~0ull;

    // Sync contract hook (front door of every public entry). Called before any mutation; returning false refuses
    // the request (nothing mutated). `err` carries the refusal reason; if left empty, "sync failed" is used.
    using SyncHook = bool (*)(void* ctx, std::string& err);
    void set_sync_hook(SyncHook fn, void* ctx) { sync_hook_ = fn; sync_hook_ctx_ = ctx; }

    // Ledger sink: null -> engine log::info; tests install a capturing sink.
    static void (*ledger_sink)(const char* line);

    bool open(uint64_t bytes, std::string& err);

    bool ensure(uint64_t bytes, uint64_t keep_free, std::string& err);
    bool back_host(uint64_t bytes, std::string& err);

    /// The device bytes ensure(bytes) would back beyond what is backed now (its rounding included, the extents
    /// region only — never the scratch windows); 0 when it backs nothing.  This is what a plan should charge.
    uint64_t growth(uint64_t bytes) const;
    void trim(uint64_t bytes);

    // host -> device (physical extents); granule-aligned or loud refusal.
    MoveResult rehome(uint64_t off, uint64_t bytes, uint64_t keep_free);
    // device -> host (physical extents); granule-aligned or loud refusal.
    MoveResult rehomes_down(uint64_t off, uint64_t bytes);

    uint8_t* data() const { return base_; }
    uint64_t backed() const { return backed_; }
    uint64_t capacity() const { return extents_limit_; }
    uint64_t host_start() const { return host_start_; }
    bool plain() const { return plain_; }

    // Readouts (pinned contract).
    uint64_t live_handle_count() const;
    ExtentState extent_state(uint64_t off) const;
    uint64_t bytes_holder(uint64_t off) const;
    std::vector<StuckEntry> stuck_entries() const;
    uint64_t scratch_live_maps() const;

    // Layout-law checks (T3 asserts I1/I2 every op).
    bool i1_ok() const;
    bool i2_ok() const;

private:
    struct Extent {
        uint64_t off = 0;
        uint64_t bytes = 0;
        uint64_t handle = 0;
        bool host = false;
        ExtentState state = ExtentState::S0;
        uint64_t pending = 0;    // S3's pending handle
        uint8_t staging = 0;     // S3: 1 = W_d, 2 = W_h
    };

    struct StuckRecord {
        uint64_t handle = 0;
        uint8_t* va = nullptr;
        uint64_t bytes = 0;
        int kind = 0;            // 0 = unmap-failed, 1 = release-pending
    };

    bool front_door(std::string& err);
    void repair_sweep();
    void emit_ledger(const char* op, bool failed, const char* step, uint64_t moved, uint64_t free_mib);

    void recompute_host_start();

    // members
    uint8_t* base_ = nullptr;
    uint64_t reserved_ = 0;       // reservation size: extents_limit_ + two granule scratch windows
    uint64_t extents_limit_ = 0;  // the user-facing extents region [0, extents_limit_)
    uint64_t backed_ = 0;         // bytes currently backed (extents cover [0, backed_))
    uint64_t gran_ = 0;           // device granule (== host granularity on the supported cards)
    uint64_t host_start_ = kNoHost;
    bool plain_ = false;
    SyncHook sync_hook_ = nullptr;
    void* sync_hook_ctx_ = nullptr;
    std::map<uint64_t, Extent> extents_;
    std::vector<StuckRecord> stuck_;
    bool w_d_live_ = false;
    bool w_h_live_ = false;
    uint64_t scratch_d_va_ = 0;
    uint64_t scratch_h_va_ = 0;
};

}  // namespace basalt::core
