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

}  // namespace basalt::core
