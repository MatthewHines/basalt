#pragma once
// KV retire policy (S2-DESIGN R1-R5): pure, clocked, per-lane. The owner (a session)
// keeps a KvSlotPolicy per lane and asks kv_policy_decide() at its known gate; the
// policy never touches CUDA — retiring/warming is the caller's, via ElasticArray's
// rehome/rehomes_down under front_door. idle_seconds == 0 keeps today's behavior off.
#include <cstdint>
#include <vector>

namespace basalt::core {

struct KvPolicyConf {
    uint64_t idle_seconds = 0;         // 0 = feature OFF (exactly today's behavior)
    uint64_t keep_warm_cells = 8192;   // newest cells that stay device-backed through a retire (R3)
    bool keep_warm_adaptive = false;   // S3-M1: band the keep target on measured pressure (off: R3's constant)
    uint64_t adaptive_low_mib = 1024;  // S3-M1: LOW = reserve + this; below LOW a retire keeps nothing
};

// Per-lane bookkeeping. The owner maintains it; only the functions below read/write it.
struct KvSlotPolicy {
    bool retired = false;          // the lane's slice currently sits host-backed past keep-warm
    uint64_t last_wake_s = 0;      // last wake-from-retire (0: never)
    unsigned wakes_in_window = 0;  // wake-from-retire count inside the 5-minute flap window (R5)
    uint64_t next_eligible_s = 0;  // hysteresis floor; decides before this instant return None
};

enum class KvAction { None, Retire };

// One gate evaluation for one lane. `busy` = the lane runs or has a queued run (R1).
// `idle_since_s` = when the lane last went quiet (run end; boot for a lane that never
// ran); 0 means "unknown" and never retires (fail-safe). Decide is idempotent: a
// retired lane answers None forever until it wakes.
KvAction kv_policy_decide(uint64_t now_s, bool busy, uint64_t idle_since_s,
                          const KvPolicyConf& conf, KvSlotPolicy& st);

// A run began on a RETIRED lane (call only on wake-from-retire): clears the state,
// restarts the quiet period from the wake (R5), and a second wake within 5 minutes
// costs one extra idle period of eligibility (count, not vibes).
void kv_policy_note_wake(uint64_t now_s, const KvPolicyConf& conf, KvSlotPolicy& st);

// A retire completed (brackets landed; stuck entries belong to the array and its repair
// sweeps, not to the policy).
void kv_policy_note_retired(uint64_t now_s, KvSlotPolicy& st);

// ---- S3-M1: the adaptive keep target (pure; the caller measures free bytes) ----
// The keep cost of an idle lane IS its live bytes, so the roomy band is priced PER LANE, never
// as a global constant. `reserve_bytes` is the engine's own floor (a floor on measured FREE):
//   free < reserve + low           -> Nothing:  keep nothing (pressure; fresh sessions reserve ~nothing)
//   free >= reserve + live + low   -> LiveOnly: keep the whole live span; retire only the dead slab
//                                     above it (the plan device-allocates the whole slice; slice
//                                     minus live is pure waste in any band)
//   else                           -> Window:   today's configured keep (the self-scaling middle)
// Off (conf.keep_warm_adaptive false) always answers Window with the configured constant.
enum class KvKeep { Nothing, Window, LiveOnly };
KvKeep kv_keep_target(uint64_t free_bytes, uint64_t reserve_bytes, uint64_t live_bytes,
                      const KvPolicyConf& conf);

// ---- S3-M2: admission negotiation (pure) ----
// A waking lane declared its size and the card cannot fund its growth above the LOW floor: IDLE
// lanes hand over warm bytes NOW, before the growing cache's relief valve demotes expert copies
// (paid in answers, not only speed). A payer never drops below its M1 band target — when the band
// says keep nothing this pays with everything; when it says window, the 8192-cell opening stays.
struct KvNegLane {                       // one idle payer, measured by the caller (bytes)
    uint64_t id;                         // the caller's lane identity, passed through
    uint64_t live_bytes;                 // what the lane has reached (kv_live_bytes)
    uint64_t backed_bytes;               // what it still holds device-backed (kv_backed_bytes)
    uint64_t cell_bytes;                 // bytes per cell (shared by a batch engine's lanes)
};
struct KvNegKeep {                       // one plan entry: retire lane `id` to `keep_cells`
    uint64_t id;
    int64_t keep_cells;                  // 0 / conf.keep_warm_cells / -1 (whole live span)
    KvKeep band;                         // the band that priced it (for the log)
};
// The plan: payers in descending freeable order (fewest retire ops per wake; the span-wake cost
// rides the bytes, not the lane count), each priced by the band law at the RUNNING free estimate
// (free rises as payers pay). Stops as soon as `free + freed >= demand + low`; if even every
// payer cannot cover the gap, it lists them all — the model's own valve pays the rest.
// `freed_estimate` sums the windowed freeable bytes (the moves themselves measure the truth).
std::vector<KvNegKeep> kv_negotiate_keeps(uint64_t free_bytes, uint64_t reserve_bytes,
                                          uint64_t demand_bytes,
                                          const std::vector<KvNegLane>& lanes,
                                          const KvPolicyConf& conf, uint64_t& freed_estimate);

}  // namespace basalt::core
