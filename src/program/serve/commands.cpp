// src/program/serve/commands.cpp - scheduling and protocol command dispatch.
#include "state.hpp"
#include "../cli/tokens.hpp"

#include <cstdio>

namespace basalt::program::serve {

namespace {
// The machine-readable side of the keep-warm ledger (M4): printed at the same settle points as the human lines,
// as `INFO` facts the server republishes under /v1/status. A lane's band changes only where these print, so the
// status keys are settled truth, never a guess; kept -1 means the room band keeps the live span (live law).
// The laws being reported are the engine's own — the tick's window/pressure band (M1) and
// core::kv_negotiate_keeps (S3-M2) — each settled at its site; this helper only speaks what was decided.
void kv_facts(size_t lane, const char* band, long long kept, uint64_t moved) {
    std::printf("INFO kv%zu_band=%s kv%zu_kept=%lld kv%zu_moved_mib=%llu\n",
                lane, band, lane, kept, lane, (unsigned long long) (moved >> 20));
    std::fflush(stdout);
}
// the busy clock runs while a request or a batch window does
struct BusyClock {
    double& total;
    bool& on;
    Clock::time_point t0;
    BusyClock(double& total_, bool& on_, Clock::time_point& from) : total(total_), on(on_), t0(Clock::now()) {
        from = t0;
        on = true;
    }
    ~BusyClock() {
        total += std::chrono::duration<double>(Clock::now() - t0).count();
        on = false;
    }
};
}  // namespace

uint64_t fnv1a(const void* data, size_t n, uint64_t h) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

std::vector<ImgKey> Runtime::imgs_below(const std::vector<ImgKey>& all, int64_t L) {
    std::vector<ImgKey> v;
    for (const ImgKey& k : all) if (k.start < L) v.push_back(k);
    return v;
}

bool Runtime::batch_on() { for (const BSlot& b : bs) if (b.active) return true; return false; }

void Runtime::stats_tick(bool force) {
    if (!stats.due(Clock::now(), force)) return;
    std::string out = stats.slot_lines();
    int64_t total = 0;
    std::vector<uint8_t> homes;
    std::vector<uint32_t> routed;
    if (expert_map(total, homes, routed)) out += ConsoleStats::map_line(total, homes, routed);
    std::string cards;
    for (const CardFacts& c : card_facts()) {
        char bus[32] = {}, gbps[32] = "-";
        if (cudaDeviceGetPCIBusId(bus, sizeof bus, c.device) != cudaSuccess) continue;
        if (c.h2d_gbps > 0) std::snprintf(gbps, sizeof gbps, "%.1f", c.h2d_gbps);
        cards += (cards.empty() ? "" : ";") + std::string(bus) + "/" + std::to_string(c.expert_slots) + "/" + gbps + "/" +
                 std::to_string(c.expert_bytes >> 20) + "/" + (c.kv_bytes >= 0 ? std::to_string(c.kv_bytes >> 20) : "-");
    }
    if (!cards.empty() && cards != cards_sent_) {
        cards_sent_ = cards;
        out += "INFO cards=" + cards + "\n";
    }
    if (out.empty()) return;
    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
}

Flow Runtime::dispatch() {
    if (batch_on()) {
        if (!try_next_line(line)) {
            const BusyClock busy(busy_s, busy_on, busy_from);
            const size_t active = (size_t) std::count_if(bs.begin(), bs.end(), [](const BSlot& b) { return b.active; });
            if (!next_batch_window()) return Flow::error;
            // a slot that ended its request has its last figures out with its BDONE
            stats_tick(active != (size_t) std::count_if(bs.begin(), bs.end(), [](const BSlot& b) { return b.active; }));
            return Flow::next;
        }
    } else {
        kv_ceiling_note();   // between requests too: the free VRAM also moves without the KV
        // idle: once no line has come for a second (a client's next turn usually comes sooner), what a crash would
        // lose goes to disk, then the idle conversations whose time is up leave VRAM, one at a time while no line
        // waits
        bool got = false, parking = true;
        std::string pe;
        while (!got && (disk.enabled() || parking)) {
            const int r = next_line_within(line, 1000);
            if (r < 0) { basalt::log::info("basalt serve", "input pipe closed"); return Flow::quit; }
            got = r == 1;
            while (!got && !(got = try_next_line(line)) && disk.enabled() && spill_to_disk(false, 1) > 0) {}
            if (got) break;
            parking = parking && park_idle_engine(pe);
            retire_idle_lanes();   // S2: lanes quiet past the threshold hand their caches to host backing (never fatal)
            if (!pe.empty()) { std::printf("ERR %s\n", pe.c_str()); return Flow::error; }
            if (!parking) break;
        }
        // The wait for the next line wakes once per second: an engine with nothing to park, or one running
        // batch slots, reaches no other idle moment while the pipe is quiet, and the wall clock needs waking
        // to fire the S2 retire. (Model names must not appear in the shared serve layer — model_layering.)
        // EOF unwinds exactly as the plain next_line did.
        if (!got) {
            for (;;) {
                const int r = next_line_within(line, 1000);
                if (r == 1) break;
                if (r < 0) { basalt::log::info("basalt serve", "input pipe closed"); return Flow::quit; }
                retire_idle_lanes();   // S2: lanes quiet past the threshold hand their caches to host backing (never fatal)
            }
        }
    }
    // --batch: BSTOP <slot> ends that slot at its next batch window; BGEN <slot> <max_new> ... reads the
    // request's prompt and first token as a GEN 1, then continues it in that slot
    // S2: a generation line is its lane's activity edge. The gate and kv_note_activity share ONE prefix set
    // ("GEN", "SGEN", "BGEN" — the I image variants included): the reader rewrites GENT/SGENT/BGENT to those
    // forms, so the *T literals here were dead and batch wakes were lost.
    if (line.rfind("GEN", 0) == 0 || line.rfind("SGEN", 0) == 0 || line.rfind("BGEN", 0) == 0)
        kv_note_activity(line);
    if (line.rfind("BSTOP ", 0) == 0) {
        const int b = std::atoi(line.c_str() + 6);
        if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
        return Flow::next;
    }
    if (line.rfind("BYIELD", 0) == 0) return Flow::next;   // for a prompt read that has ended meanwhile
    if (line.rfind("ENC ", 0) == 0 || line.rfind("ENCV ", 0) == 0) {   // the in-engine encoder, between requests and windows
        enc_command(line);
        return Flow::next;
    }
    if (line.rfind("PROBE", 0) == 0) {   // between requests, not a request (basalt-probe's commands)
        std::string perr;
        if (batch_on()) perr = "PROBE: not while batch slots are decoding";
        else probe_command(line, perr);
        if (!perr.empty()) std::printf("ERR %s\n", perr.c_str());
        std::fflush(stdout);
        return Flow::next;
    }
    if (line.rfind("VRAM", 0) == 0) {   // between requests, not a request
        std::string verr;
        if (batch_on()) verr = "VRAM: not while batch slots are decoding";
        else vram_command(line, verr, false);
        if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
        std::fflush(stdout);
        stats_tick(true);   // the cards' expert slots moved
        return Flow::next;
    }
    // SLEEP / WAKE, between requests: the expert cache gives its VRAM back down to its smallest size (every
    // expert keeps a RAM copy and the slot it held is remembered), so the process can be suspended with little
    // VRAM left to save; WAKE maps the cache again and puts each expert back in its slot.  Reply: a VRAM line.
    if (line == "SLEEP" || line == "WAKE") {
        std::string verr;
        const bool down = line == "SLEEP";
        if (batch_on()) {
            verr = line + ": not while batch slots are decoding";
        } else {
            sleep_command(down, verr);
        }
        // asleep, the driver holds the engine's GPU state in RAM until the wake: the RAM guard judges no reading
        // then (no shed, no admission wait from it), and judges a fresh one at the wake
        if (down && verr.empty()) basalt::platform::ram_guard_pause(true);
        if (!down) basalt::platform::ram_guard_pause(false);
        if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
        std::fflush(stdout);
        stats_tick(true);
        return Flow::next;
    }
    if (line == "STATS") {   // the server's console: XSLOT and XMAP lines from now on (console_stats.hpp)
        stats.on = true;
        stats_tick(true);
        return Flow::next;
    }
    if (line.rfind("PMATCH ", 0) == 0 || line.rfind("PFREE ", 0) == 0) {   // a prompt held by handle, between requests and windows
        prompt_command(line);
        return Flow::next;
    }
    if (line.rfind("ERR_GENT ", 0) == 0) {
        std::printf("ERR %s\n", line.c_str() + 9);
        std::fflush(stdout);
        return Flow::next;
    }
    basalt::platform::ram_guard_pause(false);   // a request without a WAKE first: the guard judges again
    admit_slot = -1;
    solo_slot = -1;
    if (line.rfind("SGEN ", 0) == 0 || line.rfind("SGENI ", 0) == 0) {   // SGEN <slot> <max_new> ...: a GEN in that slot
        const bool simg = line.rfind("SGENI ", 0) == 0;
        char* e1 = nullptr;
        const long b = std::strtol(line.c_str() + (simg ? 6 : 5), &e1, 10);
        if (!slot0_solo() || bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || e1 == line.c_str() + (simg ? 6 : 5)) {
            std::printf("ERR SGEN: no such free slot or slot 0 is not the solo session\n");
            std::fflush(stdout);
            return Flow::next;
        }
        solo_slot = (int) b;
        line = std::string(simg ? "GENI" : "GEN") + e1;
    }
    if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
        const bool bimg = line.rfind("BGENI ", 0) == 0;
        char* e1 = nullptr;
        const long b = std::strtol(line.c_str() + (bimg ? 6 : 5), &e1, 10);
        char* e2 = nullptr;
        const long long mn = std::strtoll(e1, &e2, 10);
        if (bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || mn < 1 || e2 == e1) {
            std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", o.batch);
            std::fflush(stdout);
            return Flow::next;
        }
        admit_slot = (int) b;
        admit_max_new = mn;
        line = std::string(bimg ? "GENI 1" : "GEN 1") + e2;
        if (o.kv_admit_negotiate) kv_negotiate_arrival((int) b, (uint64_t) mn, line);
    }
    between_requests(line);
    if (line == "QUIT") return Flow::quit;
    // the watchdog watches a request from here until this iteration ends, whichever way it ends
    struct BusyScope {
        BusyScope() { basalt::core::progress().busy.store(true); basalt::core::progress_at("request"); }
        ~BusyScope() { basalt::core::progress().busy.store(false); basalt::core::progress_at("idle"); }
    } busy_scope;
    const BusyClock busy(busy_s, busy_on, busy_from);
    return run_request();
}

// ---- S2 idle-lane KV retire (S2-DESIGN R1/R2). The loop's idle moment calls retire_idle_lanes(); the wall clock
// makes every decision and the moment only wakes it. Retiring is a speed bonus for the rest of the card, never
// fatal: a refusal logs and the next tick tries again (the arrays repair their own stuck entries at the gate, S1).
namespace {
uint64_t kv_secs(Clock::time_point t) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
}
basalt::core::KvPolicyConf kv_conf(const basalt::program::CommonOptions& o) {
    return {(uint64_t) o.kv_idle_demote_seconds, (uint64_t) o.kv_keep_warm_cells,
            o.kv_keep_warm_adaptive, (uint64_t) std::max<int64_t>(o.kv_adaptive_low_mib, 0)};
}
}  // namespace

void Runtime::retire_idle_lanes() {
    if (o.kv_idle_demote_seconds <= 0) return;   // R7: off is exactly today's loop
    const size_t n = (size_t) std::max(o.batch, 1);
    if (kv_pol.size() != n) {   // boot (or the batch size's first use): every lane has been idle since now
        kv_pol.clear(); kv_pol.resize(n);
        kv_idle_at.assign(n, Clock::now());
    }
    const basalt::core::KvPolicyConf conf = kv_conf(o);
    const Clock::time_point now = Clock::now();
    const uint64_t now_s = kv_secs(now);
    if (!batch_on() && kv_line_busy) { kv_line_busy = false; kv_idle_at[0] = now; }   // a solo run ended between lines; this tick closes its clock
    for (size_t i = 0; i < n; ++i) {
        const bool busy = batch_on() && i < bs.size() && bs[i].active;
        if (basalt::core::kv_policy_decide(now_s, busy, kv_secs(kv_idle_at[i]), conf, kv_pol[i]) != basalt::core::KvAction::Retire)
            continue;
        uint64_t freed = 0;
        std::string err;
        // S3-M1: the adaptive rail bands the keep target on measured pressure; off, or an unmeasurable
        // card, is exactly today's constant (fail-safe to the window band).
        auto& sess = slot_session((int) i);
        int64_t keep = o.kv_keep_warm_cells;
        const char* band = "window";
        uint64_t free_at_decide = 0;
        if (conf.keep_warm_adaptive) {
            uint64_t free_bytes = 0;
            if (sess.kv_card_free(free_bytes)) {
                free_at_decide = free_bytes;
                const auto target = basalt::core::kv_keep_target(
                    free_bytes, (uint64_t) std::max<int64_t>(o.vram_reserve_mib, 0) << 20,
                    sess.kv_live_bytes(), conf);
                if (target == basalt::core::KvKeep::Nothing) { keep = 0; band = "pressure"; }
                else if (target == basalt::core::KvKeep::LiveOnly) { keep = -1; band = "room"; }
            }
        }
        if (!sess.kv_retire(keep, freed, err)) {
            basalt::log::info("basalt serve", "kv retire: lane %zu refused (%s); the next tick tries again", i, err.c_str());
            continue;
        }
        basalt::core::kv_policy_note_retired(now_s, kv_pol[i]);
        // The decision line logs even at freed 0: a room band that skipped a retire and a window band that
        // found nothing past its keep are facts worth seeing once per idle cycle (the retired mark gates it).
        basalt::log::info("basalt serve", "kv retire: lane %zu settled band %s (%lld warm cells kept, %.2f GiB moved, "
                          "%.2f GiB free at decide)",
                          i, band, (long long) keep, (double) freed / (double) (1ull << 30),
                          (double) free_at_decide / (double) (1ull << 30));
        kv_facts(i, band, (long long) keep, freed);
    }
}

void Runtime::kv_note_activity(const std::string& line) {
    if (o.kv_idle_demote_seconds <= 0) return;
    // The line carries the reader-rewritten forms: GEN names the solo lane; SGEN/BGEN (and their I variants)
    // name their slot. A retired lane's wake is the ARRIVAL of its request: the lane that gets work starts its
    // flap guard and clock — never lane 0 by default.
    size_t lane;
    if (line.rfind("GEN", 0) == 0) lane = 0;   // GEN and GENI are the solo lane, whatever follows the verb
    else if (line.rfind("SGEN", 0) == 0 || line.rfind("BGEN", 0) == 0) {
        const char* p = line.c_str() + 4;
        if (*p == 'I') ++p;   // SGENI/BGENI: the interleaved variants carry the same slot field
        lane = (size_t) std::atoi(p);
    } else return;
    if (lane >= kv_pol.size()) return;
    if (lane == 0 && !batch_on()) kv_line_busy = true;   // the solo clock closes its busy edge at the next idle tick
    if (kv_pol[lane].retired)
        basalt::core::kv_policy_note_wake(kv_secs(Clock::now()), kv_conf(o), kv_pol[lane]);
}

void Runtime::kv_stamp_idle(size_t lane) {
    if (o.kv_idle_demote_seconds <= 0 || lane >= kv_idle_at.size()) return;
    kv_idle_at[lane] = Clock::now();
}

// S3-M3 close-donation. serve's real close edge is the turn's finally line: the engine wrapper PFREEs the
// request's held prompt when the turn ends (process.py). It is NOT a farewell - a multi-turn conversation
// looks exactly like this between its turns - so a close must NOT shred the keep window; the idle_demote
// grace is the design's answer to "gone or just waiting?". What a close CAN do, instantly and lawfully, is
// release the host cache (the erase below - already today's behavior) and, IF the card is in the pressure
// band (someone is hungry NOW), yield the lane's device side at once instead of holding the window for the
// tick's clock to reach. Band from M1's own pricer: pressure donates; window/room keep. A close never
// evicts a live span: a lane still decoding (the common case - its turn closed mid-round, others run on)
// is left to the round's close and the tick. Solo/last-finisher closes are the ones that land here idle,
// and those are exactly the batch's final yield moments. Policy off = PFREE erases and nothing else.
void Runtime::prompt_closed(uint64_t handle) {
    size_t cells = 0, lane = 0;
    {
        const std::lock_guard<std::mutex> lk(gent_mu);
        const auto it = prompts.find(handle);
        if (it == prompts.end()) return;   // already gone (evicted or freed): no event, no line
        cells = it->second.ids.size();
        lane = it->second.owner;
        prompts.erase(it);   // the host cache - the memory transfer S2 measured - releases at once
    }
    if (o.kv_idle_demote_seconds <= 0 || lane >= kv_idle_at.size()) return;   // off = exactly the old erase
    if (batch_on() && lane < bs.size() && bs[lane].active) return;   // mid-round close: the round and the tick govern
    auto& s = slot_session((int) lane);
    const basalt::core::KvPolicyConf conf = kv_conf(o);
    uint64_t free_bytes = 0;
    if (!conf.keep_warm_adaptive || !s.kv_card_free(free_bytes)) return;   // unmeasured rail: the tick governs
    const auto target = basalt::core::kv_keep_target(
        free_bytes, (uint64_t) std::max<int64_t>(o.vram_reserve_mib, 0) << 20, s.kv_live_bytes(), conf);
    if (target != basalt::core::KvKeep::Nothing) return;   // window or room: the keep law holds, by design
    uint64_t freed = 0;
    std::string err;
    if (!s.kv_retire(0, freed, err)) {
        basalt::log::info("basalt serve", "kv donate: lane %zu closed under pressure; the next tick retries (%s)",
                          lane, err.c_str());
        return;
    }
    if (lane < kv_pol.size()) basalt::core::kv_policy_note_retired(kv_secs(Clock::now()), kv_pol[lane]);
    // the band fact prints even when nothing moved: a pressured close is a settled state either way (M4)
    kv_facts(lane, "pressure", 0, freed);
    if (freed == 0) return;   // the ledger's own doctrine: a pressured close that moves nothing says nothing
                              // (the resident shape's truth is exactly this line's silence; proof windows saw it speak once)
    uint64_t free_now = 0;
    s.kv_card_free(free_now);
    basalt::log::info("basalt serve",
                      "kv donate: lane %zu closed under pressure (%zu cells, %.2f GiB yielded, %.2f GiB free now)",
                      lane, cells, (double) freed / (double) (1ull << 30),
                      (double) free_now / (double) (1ull << 30));
}

// ---- S3-M2: admission negotiation. A waking slot declares its size (BGEN/SGEN lines carry max_new and the
// prompt). If the card cannot fund that growth above the LOW floor, the IDLE lanes hand over warm bytes NOW,
// each to its M1 band target and never below what the tick itself would keep. The order is the point: idle
// keeps are cheap (host backing, read-keyed span wake); the relief valve - expert-cache demotion, paid inside
// ensure() - is expensive and changes answers, not only speed. Silent when funded: nothing to do, nothing to say.
void Runtime::kv_negotiate_arrival(int slot, uint64_t max_new, const std::string& rest) {
    if (o.kv_idle_demote_seconds <= 0) return;   // the quiet clock is this law's substrate
    const size_t n = (size_t) std::max(o.batch, 1);
    // The tick's own lazy init needs an idle moment, and a busy batched line never offers one: initialize
    // here too, or an arrival before the first tick is mute (observed 2026-11-26: pol 0 while slots were warm).
    // The slot table is the truth at this instant: a non-active lane HAS been idle (the tick's own idle-moment
    // init makes the same statement); an active lane's clock starts now, so a chunk mid-decode never pays.
    if (kv_pol.size() != n) {
        kv_pol.assign(n, basalt::core::KvSlotPolicy{});
        const auto now = Clock::now();
        kv_idle_at.assign(n, now - std::chrono::seconds(2));
        for (size_t i = 0; i < n && i < bs.size(); ++i)
            if (bs[i].active) kv_idle_at[i] = now;
    }
    if (slot < 0 || (size_t) slot >= n) return;
    const uint64_t cb = slot_session(slot).kv_cell_bytes();
    if (cb == 0) return;   // no elastic KV to negotiate about (or nothing reportable about its size)
    size_t prompt_len = 0;
    std::vector<int32_t> ids;
    bool media = false;
    {
        const std::lock_guard<std::mutex> lk(gent_mu);
        std::vector<std::string> toks;
        for (size_t i = 0; i < rest.size() && toks.size() < 6;) {
            while (i < rest.size() && rest[i] == ' ') ++i;
            const size_t w0 = i;
            while (i < rest.size() && rest[i] != ' ') ++i;
            if (i > w0) toks.emplace_back(rest.substr(w0, i - w0));
        }
        // The batch form names its prompt "#<handle>[+<id>,...]" (process.py's wire law), among the request's
        // keys. The ledger answers the prompt's true length.
        const std::string* named = nullptr;
        for (size_t t = 2; t < toks.size() && named == nullptr; ++t)
            if (toks[t][0] == '#') named = &toks[t];
        if (named != nullptr) {
            const auto it = prompts.find(std::strtoull(named->c_str() + 1, nullptr, 10));
            if (it != prompts.end()) {
                ids = it->second.ids; media = it->second.media; prompt_len = ids.size();
                it->second.owner = (uint16_t) slot;   // S3-M3: close-donation follows the conversation home
            }
        }
        if (ids.empty()) {   // text form "GEN <mn> <nbytes> <text...>": the text's bytes bound the need
            size_t spaces = 0, seen = 0;
            for (char c : rest) { if (c == ' ') ++spaces; ++seen; }
            prompt_len = seen > 1 ? spaces - 1 : 0;
        }
    }
    // What costs nothing is what the arrival's lane ALREADY holds. The lane's own flag says "active" (this
    // GEN just marked it), so idle_ids declines it — but its SPAN still backs the previous conversation, the
    // same bytes PMATCH priced a moment ago. The ledger reads that span directly; the media law is PMATCH's
    // own: a media conversation shares only an exact head, and a head shorter than its media span re-reads.
    size_t reuse = 0;
    if (!ids.empty() && (size_t) slot < bs.size() && bs[(size_t) slot].cached) {
        const std::vector<int32_t>& held = bs[(size_t) slot].ids;
        size_t head = 0;
        const size_t lim = std::min(held.size(), prompt_len);
        while (head < lim && held[head] == ids[head]) ++head;
        if (!media) reuse = head;
        else if (!ids.empty()) {
            // the media spans are byte ranges of the PROMPT TEXT; without re-tokenizing, a media continuation
            // is credited only what the spans cannot deny: a head the shared bytes certainly cover is rare —
            // claim none, like PMATCH does for any conversation not sharing an exact head.
            reuse = 0;
        }
    }
    const long need_cells = std::max((long) prompt_len - (long) reuse, 0L) + (long) max_new + 64;
    const uint64_t demand_bytes = (uint64_t) std::max(need_cells, 0L) * cb;
    uint64_t free_bytes = 0;
    if (!slot_session(slot).kv_card_free(free_bytes)) return;   // unmeasurable: the valve decides, as today
    const basalt::core::KvPolicyConf conf = kv_conf(o);
    const uint64_t reserve_bytes = (uint64_t) std::max<int64_t>(o.vram_reserve_mib, 0) << 20;
    std::vector<basalt::core::KvNegLane> lanes;
    const uint64_t now_s = kv_secs(Clock::now());
    for (size_t i = 0; i < n; ++i) {
        if ((int) i == slot) continue;
        if (batch_on() && i < bs.size() && bs[i].active) continue;   // only idle lanes pay
        if (now_s < kv_secs(kv_idle_at[i]) + 0.05) continue;   // mid-copy pays nothing; a FRESH idle is exactly the payer
        auto& s = slot_session((int) i);
        // The payer is the lane that still HOLDS device backing. The tick's band law is the other clock: a lane
        // idle through the window's edge has ALREADY settled toward host (its wake is S2's read-keyed re-warm;
        // the band's keep is all it holds). Both clocks act, neither twice: at the quiet gate the tick retires,
        // mid-window (batch ON) an arrival negotiates. What each payer keeps is the band's to price — here, and
        // in the pure law below — never guessed twice.
        const uint64_t backed = s.kv_backed_bytes();
        if (backed < (16ull << 20)) continue;   // settled (or near it): there is nothing here to hand over
        lanes.push_back({(uint64_t) i, s.kv_live_bytes(), backed, cb});
    }
    uint64_t freed_est = 0;
    const auto plan = basalt::core::kv_negotiate_keeps(free_bytes, reserve_bytes, demand_bytes, lanes,
                                                       conf, freed_est);
    if (plan.empty()) return;
    const uint64_t low = reserve_bytes + (conf.adaptive_low_mib << 20);
    basalt::log::info("basalt serve",
                      "kv negotiate: slot %d wakes needing %.2f GiB, %.2f GiB free (%zu payers, est %.2f GiB)",
                      slot, (double) demand_bytes / (double) (1ull << 30),
                      (double) free_bytes / (double) (1ull << 30), plan.size(),
                      (double) freed_est / (double) (1ull << 30));
    uint64_t funded = 0;
    for (const basalt::core::KvNegKeep& act : plan) {
        if (free_bytes + funded >= demand_bytes + low) break;   // funded on MEASURED frees
        auto& s = slot_session((int) act.id);
        uint64_t freed = 0;
        std::string err;
        if (!s.kv_retire(act.keep_cells, freed, err)) {
            basalt::log::info("basalt serve", "kv negotiate: payer %llu refused (%s); the valve covers it",
                              (unsigned long long) act.id, err.c_str());
            continue;
        }
        funded += freed;
        if (act.id < kv_pol.size())
            basalt::core::kv_policy_note_retired(now_s, kv_pol[(size_t) act.id]);   // the tick honors this retirement
        basalt::log::info("basalt serve",
                          "kv negotiate: lane %llu settled band %s (%lld warm cells kept, %.2f GiB handed over, "
                          "funded %.2f GiB)",
                          (unsigned long long) act.id,
                          act.band == basalt::core::KvKeep::Nothing ? "pressure"
                            : act.band == basalt::core::KvKeep::LiveOnly ? "room" : "window",
                          (long long) act.keep_cells, (double) freed / (double) (1ull << 30),
                          (double) funded / (double) (1ull << 30));
        kv_facts((size_t) act.id,
                 act.band == basalt::core::KvKeep::Nothing ? "pressure"
                                                          : act.band == basalt::core::KvKeep::LiveOnly ? "room" : "window",
                 (long long) act.keep_cells, freed);
    }
}

}  // namespace basalt::program::serve
