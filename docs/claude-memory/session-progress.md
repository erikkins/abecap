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
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP in PUBLIC (signal-intelligence = internal exception; "sleeve"/"Cascade Guard" OK public). PRINT DOCS = INK-ON-WHITE (no paper #F5F1E8 page/section bg). perf_numbers SSOT: Preserver 5yr 13.0/1.28/-12.9, 21yr 7.7/0.87/-13.7; Maximizer 5yr 31.4/1.51/-14.9, 21yr 13.5/0.93/-20.8; SPY 5yr 14.2/-25.4, 21yr 9.8/-55. Deploy=push main→GHA+smoke-test. Worker=rigacap-prod-worker AWS_PROFILE=rigacap.
- CASCADE GUARD real+live in prod (worker CIRCUIT_BREAKER_ENABLED=true) AND walk-forward (circuit_breaker_stops=3 default). 3 same-day trailing-stops→10d pause. ON live, OFF STR. Fired 7/7.
- MAXIMIZER ARCH: Preserver layer + SEPARATE gated-breakout book (maximizer_service.py): rotating_bull-gated entries, ~29d hold-to-exit, Barroso vol-target, ~15 full-notional slots, NO t30v leg. NOT "same book + overlay."

## ✅ TODAY (shipped)
- Reply engine anti-fabrication (fake WF-sim wins killed; PANW/STX posted). Bug2 Maximizer STR 8→15 backfill+self-heal. Tier-books admin UI Core-collapsed + Current price/P&L%.

## ▶ IN FLIGHT — DOCS REFRESH review (6 docs rewritten ink-on-white, verified, PDFs exported/open; NOT committed; Erik pen-marking)
- **JUST DONE:** signal-intelligence.html — added dedicated Section 08 "The Maximizer breakout book" (2nd delineated layer: regime-gated breakout entries, ~29d hold-to-exit, Barroso vol-target seatbelt, ~15 slots, no double-count) + corrected 3 wrong "only diff = overlay" claims (intro/tier-card-caption/Section-04) + fixed Maximizer tier card + Section 07 lead/callout. Renumbered old 08-14→09-15 (kickers/banners/TOC/2 body xrefs), verified 01-15 clean, div-balanced, PDF re-exported (1.5MB) + opened.
- 3 OPEN Qs for Erik (fold into markup): (1) market-pricing figures confirm (retail $129/$1099/$59-founding +future $149-179; adviser $449/$899/enterprise, founding-firm $299); (2) BacktesterService name in tech doc keep/rename; (3) investor logo = type-only wordmark now — want claret mark?
- After Erik's markups: revise HTML → re-export affected PDFs → git commit the doc set (NOTHING committed yet).

## ▶ QUEUED
- "A holding week" Maximizer label; GROWTH testimonials/churn/ads; plan unified-sauteeing-whale.md (public-number audit).
