#include "basalt/core/kv_policy.hpp"

namespace basalt::core {
namespace {
constexpr uint64_t kFlapWindowS = 300;  // R5's "any 5-minute window"
}

KvAction kv_policy_decide(uint64_t now_s, bool busy, uint64_t idle_since_s,
                          const KvPolicyConf& conf, KvSlotPolicy& st) {
    if (conf.idle_seconds == 0) return KvAction::None;   // R7: off is exactly today
    if (busy) return KvAction::None;                     // R1: a running or queued lane is never eligible
    if (st.retired) return KvAction::None;               // idempotent
    if (idle_since_s == 0) return KvAction::None;        // unknown quiet clock: never retire (fail-safe)
    if (now_s < idle_since_s + conf.idle_seconds) return KvAction::None;   // R1 quiet period
    if (now_s < st.next_eligible_s) return KvAction::None;                 // R5 hysteresis floor
    return KvAction::Retire;
}

void kv_policy_note_wake(uint64_t now_s, const KvPolicyConf& conf, KvSlotPolicy& st) {
    st.retired = false;
    if (st.last_wake_s != 0 && now_s >= st.last_wake_s && now_s - st.last_wake_s <= kFlapWindowS)
        ++st.wakes_in_window;
    else
        st.wakes_in_window = 1;
    st.last_wake_s = now_s;
    // R5: the quiet period restarts from the wake; two or more wakes inside the flap
    // window pay exactly one extra idle period (a fixed penalty, it never compounds).
    st.next_eligible_s = now_s + conf.idle_seconds + (st.wakes_in_window >= 2 ? conf.idle_seconds : 0);
}

void kv_policy_note_retired(uint64_t, KvSlotPolicy& st) {
    st.retired = true;
}

}  // namespace basalt::core
