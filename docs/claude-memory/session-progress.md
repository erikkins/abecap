---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 2dce3134-d861-45c4-a371-80378750f8c0
---

# Session snapshot — Jul 31 2026

## Frozen spec
- Never expose publicly: t30v/Core/Ensemble/Option-B/N=15/DWAP/capitulation overlay. Never "tape"/"NaN". PITFWU = survivorship-free source. Deploy = push main → wait "Deploy RigaCap" complete before resending emails.

## ▶ IN FLIGHT — public return-number audit → nail the REAL recent numbers → pages → GOOGLE ADS (Erik: "I NEED subs")
- **SSOT EXISTS** (docs/, not frontend): `docs/numbers-citations-registry.md §1` + `scripts/tier_curves_21y.json` (21-yr daily PITFWU continuous curve/tier) + standalone-clean recent from `tier_vintages_daily.py` (→ untracked `recent_tier_curves.json`) + propagator `refresh_perf_citations.py` (surface_map STALE). `canonical_numbers.json`=SUPERSEDED (old 8.3=internal). Public nums hardcoded ~12 surfaces.
- **DRIFT FOUND (Erik: "find drift, +62 doesn't seem likely" — correct):** full-history REGEN to-today (`tier_curves_21y_today.json`) gave Maximizer recent-24 +62% = BOGUS. Cause: certified vs regen curves have IDENTICAL returns thru 2021-2023 (constant ratio), diverge only 2024+ → the 2024-26 DATA was revised (universe/corp-actions) between certified vintage and today; fresh full-history regen amplifies it. Engine fine. **DISCARD regen modern numbers.** (Certified artifact tier_curves_21y.json INTACT: 14.53, ends 2026-05-29; backup .bak-may29.)
- **DECISIONS (Erik, Jul 31):** recent = ROLLING-WINDOW AVERAGES (stable), span 2021-26 (~5y modern bull), headline=AVERAGE. LEAD modern / KEEP 21-yr as resilience+drawdown foundation. Returns=HERO; Sharpe/Calmar/MDD=supporting ("top tier", always shown), PLAIN-LANGUAGE on consumer pages (MDD→"worst drop"), technical only on adviser/methodology. Recent block = typical(rolling 12/24) + spread(%pos+range) + most-recent-24. KNOB ("last 2 years") carries recent-24 + "typical year" line + refreshed worst-DD. Also SHOW recent-24 as "what's possible" alongside typical+range.
- **REAL NUMBERS (trusted = certified continuous curve + standalone-clean recent):** 21-yr anchor Preserver 8.6/0.88/−13, Maximizer 14.5/0.95/−20 (KEEP as-is). Rolling 2021-26: Preserver 12mo +11.4%/24mo +22.1%; Maximizer +27.2%/+58.6%. Modern-path Sharpe/Calmar/MDD: Pres 1.33/1.50/−9.2; Max 1.54/2.17/−14.9; S&P 0.87/0.56/−25.4 (CAGR Pres 13.8/Max 32.3/SPY 14.2). Recent-24 (standalone-clean, current, ~): Preserver +24%, Maximizer +42% (published +31/+49 were the May peak; soft Jun-Jul pulled in). **VERDICT: STILL EXCELLENT** — Max beats S&P on return AND risk; Preserver = market return at 1/3 drawdown.
- **CONFIRMATORY RUN IN FLIGHT (bg b0613xcyk, /tmp/daily_today.log):** scripts/tier_vintages_daily_today.py = standalone-clean construction (the exact build behind published 31/49) windows→today, to LOCK recent-24 (expect ~+24/+42). Present final reconciled canonical → sign-off.
- **OPEN SUB-ITEM:** dashboard CERTIFIED_WF (tier_serving.py) Pres 2021-26 MDD −20.2% vs curve −9.2% — separate reconciliation.

## NEXT (Erik's sequence): 1. confirmatory run → final canonical → sign-off. 2. PAGE WALKTHROUGH (fixes + conversion punch-list). 3. GOOGLE ADS two-tier (priority): reuse stability-search-test-a groundwork, 100% mobile, fixed pricing/founding; personas Preserver=capital-preservation $250k+ / Maximizer=aggressive-growth; fix GA4→Ads conversion-import gap.
## Gate B (after sign-off): refresh registry §1 → WIRED SSOT (backend perf_numbers.py + frontend perf_numbers.js) → re-point surface_map → update ~12 surfaces + fix retired-8.3 leaks (legacy TrackRecordPage.jsx + social cards) → rebuild + regen OG/social cards + PDFs. Plan: /Users/erikkins/.claude/plans/unified-sauteeing-whale.md.
## CLEANUP later: temp artifacts scripts/{tier_vintages_today.py, tier_vintages_daily_today.py, recompute_canonical.py, canonical_recompute.json, tier_curves_21y_today.json}.
