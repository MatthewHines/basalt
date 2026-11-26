#pragma once
// KV retire policy (S2-DESIGN R1-R5): pure, clocked, per-lane. The owner (a session)
// keeps a KvSlotPolicy per lane and asks kv_policy_decide() at its known gate; the
// policy never touches CUDA — retiring/warming is the caller's, via ElasticArray's
// rehome/rehomes_down under front_door. idle_seconds == 0 keeps today's behavior off.
#include <cstdint>

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

}  // namespace basalt::core
