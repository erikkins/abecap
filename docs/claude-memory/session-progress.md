---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Mon Aug 10 2026

## Frozen spec (load-bearing)
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP in PUBLIC (signal-intelligence = internal exception; "sleeve"/"Cascade Guard" OK public). PRINT DOCS = INK-ON-WHITE (no paper #F5F1E8 page/section bg). perf_numbers SSOT: Preserver 5yr 13.0/1.28/-12.9, 21yr 7.7/0.87/-13.7; Maximizer 5yr 31.4/1.51/-14.9, 21yr 13.5/0.93/-20.8; SPY 5yr 14.2/-25.4, 21yr 9.8/-55. Deploy=push main→GHA + smoke-test. Worker=rigacap-prod-worker AWS_PROFILE=rigacap.
- CASCADE GUARD real+live in BOTH prod (worker CIRCUIT_BREAKER_ENABLED=true) AND walk-forward (circuit_breaker_stops=3 default in StrategyParams+backtester; only off via _disable_cb ablation). 3 same-day trailing-stops→10d entry pause. ON live book, OFF for STR. Fired 7/7 (SNDK/WULF/NBIS). Docs accurate to include it.
- **MAXIMIZER ARCHITECTURE (verified maximizer_service.py):** Maximizer = Preserver layer + a SEPARATE full-notional gated-BREAKOUT standing book (MaximizerBook): own breakout entries (fire ONLY in rotating_bull regime = the gate), ~29-trading-day HOLD-TO-EXIT time-stop, continuous Barroso VOL-TARGET (target/trailing-vol, lagged, cap 1.0), ~15 slots, NO t30v leg inside. NOT "same book + overlay knob."

## ✅ TODAY (shipped)
- Reply engine anti-fabrication (fake WF-sim wins killed; PANW/STX drafts posted). Bug2 Maximizer STR 8→15 backfill+self-heal. Tier-books admin UI Core-collapsed + Current price/P&L%.

## ▶ IN FLIGHT — DOCS REFRESH review
- 6 docs rewritten ink-on-white + verified + PDFs exported/opened. NOT committed. Erik marking up w/ pen.
- **ACTIVE FIX PENDING ERIK OK:** signal-intelligence.html WRONGLY says Preserver/Maximizer are "same book, only diff = capitulation overlay" (lines 763,796,1144) and OMITS the breakout hunter. Proposed: add "The Maximizer breakout book" section + correct the 3 statements + fix Maximizer tier-card. Asked Erik to confirm framing (breakout as 2nd delineated layer) before I write. WAITING on his go/framing.
- 3 other open Qs: market-pricing figures confirm; BacktesterService name keep/rename; investor logo (type-only now, want claret mark).
- After edits: re-export affected PDFs + git commit doc set.

## ▶ QUEUED
- "A holding week" Maximizer label; GROWTH testimonials/churn/ads; plan unified-sauteeing-whale.md.
