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
- Never expose publicly: t30v/Core/Ensemble/Option-B/N=15/DWAP/overlay-internals. Never "tape"/"NaN". PITFWU=survivorship-free. Deploy=push main→wait "Deploy RigaCap" complete before resending emails. BE VERY CAUTIOUS changing the tier books (Erik built them this week) — CHECK before any change.

## ▶ RESOLVED the construction question → awaiting Erik's OK to lock OVERLAY as canonical. Sequence: lock numbers → PAGE WALKTHROUGH → GOOGLE ADS (needs subs).
- **WHAT WE ACTUALLY SHIP (verified in code, NO changes made):** Core=t30v. **Preserver = OVERLAY** (preserver_service.py: core_ret×exposure, cut to 0.25 in capitulation {panic_crash,recovery,weak_bear}; does NOT hold pullback/oversold sleeves). **Maximizer = overlay base + a REAL held breakout book** (29-day hold-to-exit) in rotating_bull. Docstring confirms Rule-B sleeve attempt "collapsed to ≈Core" → overlay is the shipped resolution.
- **RESEARCH/marketed 8.6/14.5 = SLEEVE-IDEALIZED** = np.where(calm→pullback, cap→oversold, else→t30v) — an idealized instant costless daily return-stream swap that does NOT reproduce in a single pool. Live has NEVER fired a sleeve (100% rotating_bull since infra ~Jun-Jul 2026; non-rotating regimes were Mar-Jun 2025 pre-infra) → live=t30v only.
- **HONEST OVERLAY NUMBERS computed (from certified curves + cached regime, no re-run):**
  Preserver: 21yr 7.7%/0.87/−13.7 · 5yr 13.0%/1.28/−12.9 · 24mo(to-May peak) 28.1%/2.38/−6.7.
  Maximizer: 21yr 13.5%/0.93/−20.8 · 5yr 31.4%/1.51/−14.9 · 24mo(to-May peak) 59.9%/2.20/−12.5.
  **vs sleeve-marketed (8.6/14.5 21yr, 14.7/32.5 5yr): overlay is only ~1pp LOWER return but BETTER drawdown** (Max 5yr DD −14.9 overlay vs −22.7 sleeve — cash-raise cut 2022). Honest + deliverable + stronger risk story. RECOMMEND: adopt overlay as canonical.
- **3rd-party assessment (Erik ran it):** product looks good across 21/5/2yr; key caution = "live vs backtest / reproducible" — maps exactly to sleeve-doesn't-reproduce → overlay is the answer. Facts: backtest not live (live=t30v since Jun2026); fees excluded; ~15bps cost modeled, slippage not; mechanical; ~20 pos.
- **DRIFT (earlier):** regen-to-today +62% Max was a data-vintage artifact (recent liquidity-panel settling; NOT missing corp-actions [calendar complete to 2026-12-31], NOT engine bug). Discarded.
- **My session Core=Preserver book fix:** correct for the LIVE period (all rotating_bull → no capitulation → Preserver=Core); Preserver legitimately diverges from Core historically via the capitulation cash-raise (backtest). So fix stands.

## DECISIONS locked (Erik): recent=ROLLING-AVG span 2021-26; headline=average; LEAD modern/KEEP 21yr foundation; returns=HERO, Sharpe/Calmar/MDD=supporting (plain-language consumer, technical adviser); knob="last 2 years"=recent-24 + "typical year" + worst-DD; also show recent-24 as "what's possible"+range. Publish 21yr+24mo.

## NEXT: Erik OK to lock OVERLAY canonical → finalize current recent-24 (overlay, to-today) → Gate B (refresh registry §1 → WIRED SSOT backend perf_numbers.py + frontend perf_numbers.js → re-point surface_map → update ~12 surfaces, fix retired-8.3 leaks on legacy TrackRecordPage.jsx + social → rebuild + regen cards/PDFs) → PAGE WALKTHROUGH → GOOGLE ADS two-tier (Preserver=capital-preservation $250k+ / Maximizer=aggressive-growth; reuse stability-search-test-a; fix GA4→Ads conversion import). Plan: /Users/erikkins/.claude/plans/unified-sauteeing-whale.md.
## CLEANUP temp artifacts: scripts/{tier_vintages_today.py, tier_vintages_daily_today.py, recompute_canonical.py, canonical_recompute.json, tier_curves_21y_today.json, tier_curves_21y.json.bak-may29}.
