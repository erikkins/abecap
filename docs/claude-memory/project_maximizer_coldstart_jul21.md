---
name: project-maximizer-coldstart-jul21
description: "Maximizer vol-brake must warm-start (never cold-start); cold Jun15 launch = -22%, warmed = -12%; cold-start is a general tier launch hazard"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Maximizer cold-start finding + warm-start requirement (Jul 21 2026)

## 🚨 VERIFIED LIVE (Aug 20 2026) — the warm-start requirement was NOT applied to the live book
- Discovered while trying to run a faithful start-date sweep (couldn't validate → found this instead). Verified via db_read on maximizer_book_snapshots + code (maximizer_service `_VOL_WIN=20`, `_vol_scale` returns 1.0 if `len(bk_eq_hist) < 21`).
- Live Maximizer book: launched **Jul 8** (first fills Jul14–15), **re-based/format-changed Jul 24** (equity re-anchored $98,775.91, bk_eq_hist RESET to len 1). bk_eq_hist only **19 on Aug 19** → **vol_scale = 1.0 (brake OFF, full exposure) EVERY day of its life so far.** Self-warms ~Aug 21 as count crosses 21, but first ~6 weeks ran unbraked.
- Equity path (real): +0.45% (Jul15 peak) → −3.21% (Jul31 trough) → **−1.69% / $98,311.30 (Aug19)**. (The −0.9%/$99,125 we'd cited was Aug 14 — it slipped.)
- Cause: the book cold-started (bk_eq_hist accumulates from scratch, no backtest seed) + Jul-24 rebase wiped history — contra THIS file's "MUST warm-start (seed eq_hist from backtested equity)" rule. 0 live Maximizer subs so no customer harm yet, but affects book/track-record integrity.
- FIX (proposed, AWAITING ERIK sign-off — load-bearing brake): seed bk_eq_hist from backtested breakout equity so vol_scale is warm day 1 + re-seed on any rebase. THEN tranche customer entry + run the faithful start-date sweep validated against the reproducible Jul-24+ new-format era (replay Jul-24 fwd → must reproduce $98,311.30 before trusting alt-start numbers). Sweep handler NOT built/deployed (fork correctly stopped at the validation gate).


## Question (Erik): does Maximizer beat SPY? Expected yes.
Answer: NO over the Jun15→Jul20 drawdown window. Faithful replay (scripts/maximizer_backfill.py, pitfwu panel, breakout sleeve, n_pos=15):
- **COLD-start Jun15 (brake off — eq_hist empty <21d, all-fresh pile-in at the top): −22.2%**
- **WARMED (warmup from 2026-05-01 so brake + held book realistic at Jun15): −12.1%** (brake=0.37 at Jun15, 12 positions already held)
- vs Core/Preserver −7.5%, SPY −1.7%.

## Key methodology fix (Erik's catch): DON'T cold-start the vol-brake
The Barroso vol-brake reads the BOOK's own eq_hist; with 0 history it returns 1.0 (no brake) for the first ~21 days — exactly when protection is needed. A continuously-running strategy hits any date with the brake ALREADY warm + positions ALREADY held (aged), not cold-piling into the top. Cold-start test overstated the loss by ~10pp.

## Honest read
Even warmed, Maximizer −12% > Core −7.5% > SPY −1.7% in losses over THIS window. Expected: Maximizer is higher-beta (breakout) → loses more in drawdowns, gains more in rallies (the +63% ttm backtest is the other side). "Beats SPY" is a FULL-CYCLE question, not a 5-week-drawdown one. The live Maximizer shadow looked flat (−1.1%) ONLY because it launched Jul8 post-crash into CASH (held=0 Jul8-13; breakout found nothing to buy).

## PRODUCTION REQUIREMENT (bake into WS3 serving)
When Maximizer goes live, WARM-START the vol-brake — seed eq_hist from the strategy's backtested equity history, NOT cold. Else a bad-timed live launch = the −22% scenario. Same anti-cold-start lesson as Core's Jun-15 concentration pile-in ([[project_sector_cap_regression_jul20]]).

## OPEN
- Erik deciding: write the WARMED −12% Jun15-anchored Maximizer backfill to prod (overwrites live Jul8 shadow rows), or keep live shadow.
- Fair comparison caveat: Core/Preserver −7.5% is ALSO partly cold-start artifact (live book cold-started Jun15; a warmed/continuous book wouldn't have piled into the top either). Offered to quantify warmed Core/Preserver.
