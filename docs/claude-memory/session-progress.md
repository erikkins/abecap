---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 2dce3134-d861-45c4-a371-80378750f8c0
---

# Session snapshot — Jul 31 2026

## Frozen spec (load-bearing)
- Never expose publicly: t30v/Core/Ensemble/Option-B/N=15/DWAP/capitulation overlay. Never "tape"/"NaN". Lean on **PITFWU** (survivorship-free) for all recomputes. Deploy = push main → wait for "Deploy RigaCap" workflow COMPLETE before resending emails.

## ▶ IN FLIGHT — public return-number audit → rolling-window re-baseline → page walkthrough → GOOGLE ADS (Erik: "I NEED subs")
- **SSOT EXISTS** (in docs/, not frontend): `docs/numbers-citations-registry.md §1` (canonical) + `scripts/tier_curves_21y.json` (21-yr daily PITFWU curves/tier: Preserver 8.65/0.88/−13.2, Maximizer 14.53/0.95/−20.4 ✓ matches site) + propagator `scripts/refresh_perf_citations.py` via `scripts/perf_citations_surface_map.json` (patterns STALE). `docs/canonical_numbers.json`=SUPERSEDED (8.3/0.73/19 = old single-strat, internal-only). Public numbers hardcoded in ~12 surfaces.
- **Problem found:** published 24-mo (31.3/48.9) = ONE favorable single-window @end-May-2026 — accurate for that window but not typical + now stale (to Jul ≈ 24/42). Defects: legacy TrackRecordPage.jsx + social cards still publish retired 8.3.
- **DECISIONS (Erik, Jul 31):** recent numbers = **ROLLING-WINDOW AVERAGES** (no monthly recalc), span **2021–26 (~5y modern bull market)**, headline stat = **average**. **LEAD modern returns / KEEP 21-yr as resilience+drawdown foundation.** Display: **returns = HERO; Sharpe/Calmar/MDD = supporting ("top tier", always shown), plain-language on consumer pages** (MDD→"worst drop"), technical only on adviser/methodology. Recent block = typical(rolling 12/24) + spread(%pos+best/worst range) + most-recent-24(current). **KNOB graphic** cites "last 2 years" → carries recent-24 (Max ~+42%, Pres ~+24%) + a "typical year" line + refreshed worst-DD; applies to landing dial + tier-announce email.
- **INTERIM modern 2021–26 (apples-to-apples, SPY loaded live):** S&P 12mo avg +11.7%/24mo +27.7%/path CAGR 14.2%/MDD −25.4%/Sharpe 0.87/Calmar 0.56. Preserver +11.4%/+22.1%/13.8%/−9.2%/1.33/1.50. Maximizer +27.2%/+58.6%/32.3%/−14.9%/1.54/2.17. → STILL EXCELLENT (Max beats S&P on return AND risk; Pres = market return at 1/3 drawdown). Honest caveat: Pres slightly trails S&P raw return in bull runs (by design).
- **RECOMPUTE RUNNING (bg bai0lga2a, /tmp/recompute.log):** scripts/recompute_canonical.py — tiers+SPY+raw-mom(12-mo factor lookback=252) 2021→today, full biweekly WF (~115 periods, slow). **ON COMPLETION: anchor-validate vs tier_curves_21y.json 2021-slice BEFORE trusting** (log shows WF-SERVICE strategy6/6pos/15% defaults — verify pwf.run overrides to t30v 20/4.5; fall back to certified curve if mismatch). Then post-process → final canonical table → Erik sign-off.
- **OPEN SUB-ITEM:** dashboard CERTIFIED_WF (tier_serving.py) Pres 2021-26 MDD −20.2% vs public curve −9.2% — separate reconciliation.

## NEXT (sequence Erik set)
1. Recompute → final canonical → sign-off. 2. **Page walkthrough** (landing/track-record/methodology/for-advisers/blogs/FAQ) = number fixes + copy/UX/conversion punch-list. 3. **GOOGLE ADS two-tier launch** (priority — needs subs): reuse prior stability-search-test-a groundwork, 100% mobile, fixed pricing/founding; personas Preserver=capital-preservation $250k+ / Maximizer=aggressive-growth; fix GA4→Ads conversion-import gap.
- Gate B (after sign-off): refresh registry §1 → WIRED SSOT (backend perf_numbers.py + frontend perf_numbers.js) → re-point surface_map → update ~12 surfaces + fix retired-8.3 leaks → rebuild + regen OG/social cards + PDFs. Plan file: /Users/erikkins/.claude/plans/unified-sauteeing-whale.md.
