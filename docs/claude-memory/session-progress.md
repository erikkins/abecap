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
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP in PUBLIC (signal-intelligence dossier = internal exception; "sleeve" & "Cascade Guard" are OK public terms). PRINT DOCS = INK-ON-WHITE (no paper #F5F1E8 page/section bg — chips only). perf_numbers SSOT: Preserver 5yr 13.0/1.28/-12.9, 21yr 7.7/0.87/-13.7; Maximizer 5yr 31.4/1.51/-14.9, 21yr 13.5/0.93/-20.8; SPY 5yr 14.2/-25.4, 21yr 9.8/-55. Deploy=push main→GHA; smoke-test after. Worker=rigacap-prod-worker AWS_PROFILE=rigacap.
- **CASCADE GUARD = REAL + LIVE** (verified today): production circuit breaker (circuit_breaker_state.py), CIRCUIT_BREAKER_ENABLED=true on WORKER (unset on api, correct). 3 same-day trailing-stops → pause new entries 10d. Mirrors WF CB (~+3.7pp/yr). ON for live book, OFF for STR (same-day cascades = noise there). Actually FIRED 2026-07-07 (SNDK/WULF/NBIS) → paused to 7/17. S3 cb-state/live.json.

## ✅ EARLIER TODAY (shipped+deployed)
- Reply engine de-mech + ANTI-FABRICATION (fake WF-sim wins killed; PANW/STX drafts sent, Erik posting). Bug2 Maximizer STR 8→15 backfill+self-heal. Tier-books admin UI: Core collapsed + Current price/P&L%.

## ▶ IN FLIGHT — DOCS REFRESH (awaiting Erik pen markup)
- All 6 design/documents/ rewritten ink-on-white via workflow, VERIFIED (no navy/gold, numbers=SSOT, 2-tier, well-formed). Post-verify fixes applied. **6 PDFs EXPORTED** (headless Chrome, white bg) + opened — Erik printing to mark up with a pen.
- **NOT committed yet** (files modified on disk). Await Erik's markup → revise HTML → re-export PDFs → THEN git commit the set.
- 3 OPEN Qs for Erik (fold into markup): (1) confirm market-pricing figures (retail $129/$1099/$59-founding +future $149-179; adviser $449/$899/enterprise, founding-firm $299) — verify-only, unchanged; (2) BacktesterService class name in tech doc — keep/rename in doc; (3) investor logo = type-only wordmark now (navy SVG dropped) — want claret logo? Cascade Guard confirmed real → docs are accurate to include it.

## ▶ QUEUED
- "A holding week" Maximizer label; GROWTH testimonials/churn/ads; plan unified-sauteeing-whale.md (public-number consistency audit).
