// kv_policy_test — T-S2-1: the pure retire policy (S2-DESIGN R1/R5/R7), CPU-only.
#include "basalt/core/kv_policy.hpp"

#include <cstdio>

using basalt::core::KvAction;
using basalt::core::KvPolicyConf;
using basalt::core::KvSlotPolicy;
using basalt::core::kv_policy_decide;
using basalt::core::kv_policy_note_retired;
using basalt::core::kv_policy_note_wake;
using basalt::core::kv_keep_target;
using basalt::core::KvNegLane;
using basalt::core::kv_negotiate_keeps;
#include <vector>

static int fails = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++fails; }
}
static bool is_retire(KvAction a) { return a == KvAction::Retire; }

int main() {
    const KvPolicyConf off{};                       // defaults: OFF, keep-warm 8192
    check(off.idle_seconds == 0, "default feature is OFF (R7)");
    check(off.keep_warm_cells == 8192, "default keep-warm is 8192 cells (OQ-2 lean)");

    KvPolicyConf c30{}; c30.idle_seconds = 30;

    // R7: off never retires, at any quiet length.
    { KvSlotPolicy st; check(!is_retire(kv_policy_decide(100000, false, 10, off, st)), "off: never retires"); }

    // R1: quiet period boundaries (>= T retires; < T and busy do not).
    {
        KvSlotPolicy st;
        check(!is_retire(kv_policy_decide(129, false, 100, c30, st)), "busy-clock: 29 s quiet is not enough");
        check(is_retire(kv_policy_decide(130, false, 100, c30, st)), "exactly T quiet retires");
        check(!is_retire(kv_policy_decide(1000, true, 100, c30, st)), "R1: a busy lane never retires");
        check(!is_retire(kv_policy_decide(1000, false, 0, c30, st)), "unknown quiet clock never retires");
    }

    // Idempotence: a retired lane answers None until it wakes.
    {
        KvSlotPolicy st; kv_policy_note_retired(200, st);
        check(!is_retire(kv_policy_decide(100000, false, 0, c30, st)), "retired lane does not re-retire");
    }

    // R5: the quiet period restarts from the wake.
    {
        KvSlotPolicy st; kv_policy_note_retired(900, st);
        kv_policy_note_wake(1000, c30, st);
        check(!st.retired, "wake clears the retired state");
        check(!is_retire(kv_policy_decide(1034, false, 1005, c30, st)), "wake + 34 s: still inside the restarted period");
        check(is_retire(kv_policy_decide(1035, false, 1005, c30, st)), "idle_since + T retires (the R1 clock, when past the floor)");
    }

    // R5: a second wake inside 5 minutes costs exactly one extra idle period (no compounding).
    {
        KvSlotPolicy st; kv_policy_note_retired(900, st);
        kv_policy_note_wake(1000, c30, st);
        kv_policy_note_retired(1100, st);
        kv_policy_note_wake(1150, c30, st);          // second wake, 150 s later: inside the window
        check(!is_retire(kv_policy_decide(1209, false, 1160, c30, st)), "flap penalty: 1 cycle + T not enough at T");
        check(is_retire(kv_policy_decide(1210, false, 1160, c30, st)), "flap penalty: two periods pay exactly one extra cycle");
        kv_policy_note_retired(1300, st);
        kv_policy_note_wake(1350, c30, st);          // third wake, still within 5 min of the second
        check(is_retire(kv_policy_decide(1410, false, 1360, c30, st)), "penalty stays at one extra cycle (no compounding)");
    }

    // R5: a wake after the flap window resets the count.
    {
        KvSlotPolicy st; kv_policy_note_retired(900, st);
        kv_policy_note_wake(1000, c30, st);
        kv_policy_note_retired(2000, st);
        kv_policy_note_wake(2100, c30, st);          // > 300 s after the previous wake
        check(is_retire(kv_policy_decide(2135, false, 2105, c30, st)), "window elapsed: single quiet period again (R1 clock rules)");
    }

    // S3-M1: the keep-target bands (pure arithmetic; the tick passes measured bytes).
    {
        KvPolicyConf a{}; a.idle_seconds = 30; a.keep_warm_adaptive = true; a.adaptive_low_mib = 1024;
        using basalt::core::KvKeep;
        const uint64_t reserve = 2048ull << 20, live = 600ull << 20;   // floor 2 GiB; a 600 MiB live lane
        check(kv_keep_target(reserve, reserve, live, a) == KvKeep::Nothing, "at the floor: pressure keeps nothing");
        check(kv_keep_target(reserve + (1024ull << 20) - 1, reserve, live, a) == KvKeep::Nothing, "one byte under LOW is pressure");
        check(kv_keep_target(reserve + (1024ull << 20), reserve, live, a) == KvKeep::Window, "at LOW: today's window");
        check(kv_keep_target(reserve + (1024ull << 20) + live, reserve, live, a) == KvKeep::LiveOnly, "LOW+live is room (>=)");
        check(kv_keep_target(1ull << 40, reserve, live, a) == KvKeep::LiveOnly, "roomy keeps the live span, dead slab goes");
        check(kv_keep_target(reserve, reserve, 0, a) == KvKeep::Nothing, "fresh lane at the floor: keep nothing");
        check(kv_keep_target(reserve + (1024ull << 20), reserve, 0, a) == KvKeep::LiveOnly, "fresh lane with room: live span is nothing");
        check(kv_keep_target(0, 0, 0, off) == KvKeep::Window, "off answers the window at any free reading");
        KvPolicyConf off2{}; off2.keep_warm_adaptive = true;
        check(kv_keep_target(0, 0, 0, off2) == KvKeep::Nothing, "on with a zero floor is pressure at zero free");
    }

    // S3-M2: admission negotiation (payer order, band floors, the funded stop).
    {
        KvPolicyConf a{}; a.idle_seconds = 30; a.keep_warm_adaptive = true; a.adaptive_low_mib = 1024;
        KvPolicyConf z{}; z.idle_seconds = 30; z.keep_warm_adaptive = true; z.adaptive_low_mib = 0;
        using basalt::core::KvKeep;
        const uint64_t MiB = 1ull << 20, CB = 64 * 1024;
        std::vector<KvNegLane> one{{7, 400 * MiB, 600 * MiB, CB}};
        uint64_t est = 42;
        check(kv_negotiate_keeps(20ull << 30, 1ull << 30, 2ull << 30, one, a, est).empty() && est == 0,
              "M2: a funded wake has no payers and says nothing");
        auto p = kv_negotiate_keeps(2 * MiB, 1ull << 30, 300 * MiB, one, a, est);
        check(p.size() == 1 && p[0].id == 7 && p[0].keep_cells == 0 && p[0].band == KvKeep::Nothing
                  && est == 600 * MiB, "M2: pressure payer hands everything (keep 0, band answers)");
        auto pw = kv_negotiate_keeps(2 * MiB, 1ull << 30, 300 * MiB, one, off, est);
        check(pw.size() == 1 && pw[0].keep_cells == 8192 && est == 200 * MiB,
              "M2: rail off still negotiates, never below the window");
        std::vector<KvNegLane> tidy{{3, 100 * MiB, 100 * MiB, CB}};   // backed == its window: pays nothing
        check(kv_negotiate_keeps(2 * MiB, 1ull << 30, 300 * MiB, tidy, off, est).empty(),
              "M2: a payer with no spare bytes past its window is never listed");
        std::vector<KvNegLane> pair{{3, 1ull << 30, 2ull << 30, CB}, {7, 8ull << 30, 8ull << 30, CB}};
        auto pa = kv_negotiate_keeps(0, 1ull << 30, 5ull << 30, pair, z, est);
        check(pa.size() == 1 && pa[0].id == 7, "M2: biggest payer first, stop as soon as funded");
        auto pb = kv_negotiate_keeps(0, 1ull << 30, 9ull << 30, pair, z, est);
        check(pb.size() == 2 && pb[0].id == 7 && pb[1].id == 3 && pb[1].keep_cells == -1
                  && pb[1].band == KvKeep::LiveOnly,
              "M2: the second payer re-prices to ROOM mid-plan (its live span is the room price) and pays its dead slab");
        std::vector<KvNegLane> slab{{5, 10ull * MiB, 6ull << 30, CB}};   // live 10 MiB under 6 GiB backed
        auto pr = kv_negotiate_keeps(2ull << 30, 0, 10ull << 30, slab, z, est);
        check(pr.size() == 1 && pr[0].keep_cells == -1 && pr[0].band == KvKeep::LiveOnly,
              "M2: a room-band payer keeps the live span, the dead slab pays");
    }

    if (fails == 0) { std::printf("kv_policy_test: PASS\n"); return 0; }
    std::printf("kv_policy_test: FAIL (%d)\n", fails);
    return 1;
}
