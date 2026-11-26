#include "basalt/core/kv_policy.hpp"
#include <algorithm>
#include <vector>

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

KvKeep kv_keep_target(uint64_t free_bytes, uint64_t reserve_bytes, uint64_t live_bytes,
                      const KvPolicyConf& conf) {
    if (!conf.keep_warm_adaptive) return KvKeep::Window;   // off is exactly R3's constant
    const uint64_t low = reserve_bytes + (conf.adaptive_low_mib << 20);
    if (free_bytes < low) return KvKeep::Nothing;
    if (free_bytes >= low + live_bytes) return KvKeep::LiveOnly;   // live 0 + room: keep nothing, dead slab goes
    return KvKeep::Window;
}

// The device bytes a keep target leaves on the card: Nothing keeps nothing (the empty window at
// live: kv_retire(0) retires the head down to live and the slab above); Window keeps min(keep,
// live) (the window ends at max(live, keep) and starts at live - keep); LiveOnly keeps live.
namespace {
uint64_t keep_window_bytes(KvKeep band, int64_t keep_cells, const KvNegLane& lane) {
    if (band == KvKeep::Nothing) return 0;
    const uint64_t want = band == KvKeep::LiveOnly
                              ? lane.live_bytes
                              : std::min(lane.live_bytes, (uint64_t) keep_cells * lane.cell_bytes);
    return std::min(want, lane.backed_bytes);   // cannot keep what was never backed
}
}  // namespace

std::vector<KvNegKeep> kv_negotiate_keeps(uint64_t free_bytes, uint64_t reserve_bytes,
                                          uint64_t demand_bytes,
                                          const std::vector<KvNegLane>& lanes,
                                          const KvPolicyConf& conf, uint64_t& freed_estimate) {
    freed_estimate = 0;
    std::vector<KvNegKeep> plan;
    const uint64_t low = reserve_bytes + (conf.adaptive_low_mib << 20);
    if (demand_bytes + low <= free_bytes) return plan;   // the card funds the wake unaided
    // Order by what each lane could free at the ARRIVAL pressure (descending): the plan takes the
    // biggest payer first, so one wake costs the fewest retire operations for the same bytes.
    std::vector<KvNegLane> ordered = lanes;
    std::sort(ordered.begin(), ordered.end(), [&](const KvNegLane& a, const KvNegLane& b) {
        const auto price = [&](const KvNegLane& x) {
            const int64_t k = conf.keep_warm_cells > INT64_MAX ? INT64_MAX : (int64_t) conf.keep_warm_cells;
            return x.backed_bytes - keep_window_bytes(kv_keep_target(free_bytes, reserve_bytes, x.live_bytes, conf), k, x);
        };
        return price(a) > price(b);
    });
    uint64_t free_now = free_bytes;
    const int64_t keep_cfg = conf.keep_warm_cells > INT64_MAX ? INT64_MAX : (int64_t) conf.keep_warm_cells;
    for (const KvNegLane& lane : ordered) {
        if (free_now >= demand_bytes + low) break;   // funded
        const KvKeep band = kv_keep_target(free_now, reserve_bytes, lane.live_bytes, conf);
        const int64_t keep_cells = band == KvKeep::Nothing ? 0 : band == KvKeep::LiveOnly ? -1 : keep_cfg;
        const uint64_t win = keep_window_bytes(band, keep_cells, lane);
        if (lane.backed_bytes <= win) continue;      // pays nothing at this band: skip (never list)
        plan.push_back({lane.id, keep_cells, band});
        freed_estimate += lane.backed_bytes - win;
        free_now += lane.backed_bytes - win;         // the estimate; the moves measure the truth
    }
    return plan;
}

}  // namespace basalt::core
