---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — thru Aug 24 2026

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC; never "tape"; NO fabrication. Web deploy=push main→"Deploy RigaCap" GHA + smoke api.rigacap.com/api/market-data-status→200. Admin app=mobile-admin/ EAS OTA. NO Google Ads API (Erik does ads UI). RigaCap=PUBLISHER; 0 real external subs (the "6"=Erik+beta comps). perf_numbers SSOT. Verify before concluding; research maps to prod. Admin/test emails→erik@rigacap.com.

## 🧭 LIVE THREAD — FREE-FIRST conversion pivot (DESIGN done for free/paid line; not built) → full spec: [[project_free_first_spec]]
- WHY: 0 subs + unknown brand can't borrow trusted-incumbent (Fool/SA) card-friction → earn trust via FREE experience, card LATER. Ads bring OLD+MALE(55+≈65%) preservers who buy TRUST not FOMO; 83% mobile. Fixes the funnel leak (offer→CTA ~1/43 on /should-i-sell; hero/fold FINE at 48%).
- ✅ ALL 4 DECISIONS LOCKED: (1) FREE/PAID LINE = **PROOF ONLY, zero actionable** — free shows CLOSED-trade ledger (named results, reveal ON CLOSE: Maximizer ~29d, Preserver on-exit) + current book COUNTS/perf (NO names) + free market read (NO live tickers) + full track record; PAID = live names/entries/weights/exits/realtime. (2) Free-tier LEAD personalized by ad door + protect/grow toggle; tier CHOICE at upgrade only. (3) 30-day MONEY-BACK guarantee (replaces 7-day CC trial). (4) KEEP intro rate ($59→$129, 12mo lock) — reframe Founding/First-100 → "Introductory".
- KEY CATCH: daily market read NAMES live tickers → needs a FREE-tier ticker-free version; ANTI-LEAK generalizes to a PROSE-SURFACE audit (read/email/briefings), enforced at DATA layer (entitlement choke point → free never fetches live names) + fail-closed pre-send guard scanning free emails for ANY ticker.
- DATA MODEL: add `free` state, RETIRE `trial`, `is_valid()`=paid-only (active+grace); free users get explicit status='free' record at register; migration=non-event (comp internal/beta). QA: preview_state + force_state (parallels preview_tier/force_tier) to inspect every state's UI+email. Email capture (ANY point incl ABANDONED signup)→newsletter enroll (dedup by state, CAN-SPAM unsub, deliverability). Register→free (stop auto-Stripe); 2-step modal (card only at upgrade). Drip rewrite (free-nurture→paid, state-aware).
- BUILD PROGRESS: **#1 BACKBONE ✅ BUILT (Aug24, code only, NOT deployed, verified no-op)** — `Subscription.is_valid()`=paid-only + `.entitlement()` + `security.resolve_entitlement(user,sub)` choke point, wired into `/dashboard` (signals.py:2053; free seam tagged `entitlement:'free'`). No-op verified: only 2 of 9 subs valid today; nobody's validity changed; no migration. **Re-comped the 4 lapsed beta** (erik@amberland/jglynn/arnist/jpatrickfrei) to period_end 2027-08-24 via `{"run_migration",sql:UPDATE...RETURNING}` (Erik chose re-comp). **#2 IN PROGRESS** = enrich the free seam: (A) closed-trade "We Called It" ledger [named, reveal-on-close], (B) current-book COUNTS+perf per tier [NO names], (C) ticker-free market read — + frontend render + fail-closed anti-leak guard. Explore agent mapping the 3 data sources. THEN #3 register→free, #4 QA harness (preview_state/force_state), #5 upgrade/offer copy, #6 drip, #7 landing copy.

## ✅ SHIPPED recently (live): Maximizer vol-brake cold-start fix (bb3c9be, self-heals Aug24 scan); start-date sweep (c622e3e); newsletter weekly-pack+regime-run+tier rule (f127d0b); SPY-trend reads +3streak; 2 ad doors + dual funnel tracking + signup sub-steps; personal social posts; "Pascal's Portfolio" newsletter draft seeded ([[project_newsletter_pascal_topic]]).

## ▶ OTHER OPEN: ad negatives to paste (marketbeat/warrior trading/trendspider/etc); DOCS refresh (design/documents/* uncommitted); verify brake gauge after Aug24 4pm scan.