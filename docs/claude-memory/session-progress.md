---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 14–15 2026

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC. PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Web deploy=push main→"Deploy RigaCap" GHA + smoke api.rigacap.com/api/market-data-status→200. Migration-first DB. NEVER lambda update-function-configuration --environment / terraform apply w/o plan. Admin app=mobile-admin/ Expo EAS OTA (`eas update --channel preview`; Erik reopens app). TIER_SERVING=true; 0 real external subs. Admin/test emails→erik@rigacap.com. Email=SMTP/aiosmtplib (NOT SES); scheduler logger NOT in CloudWatch → diagnose sends via email_events table.
- **RigaCap is a PUBLISHER** (signals-only, no custody). No "your portfolio value" screen by design — user perf lives in their own brokerage. YOUR CAPITAL = scaling input. Book equity ($92,846 Pres/$99,125 Max) = model book MTM from $100k CAP0 (−7.2%/−0.9%), NOT per-user.

## ✅ SHIPPED this session (all live)
- **Served-Maximizer PORTAL** ("both books"): [Preserver | Maximizer + Rotation watch] side-by-side, books-on-top, one capital control, Market Read pixel-aligned, whole shares no ≈, candidates full-width below, date+Last-updated one row, Rotation watch fills to Preserver bottom (measured floor). Recently-closed wired+DORMANT (0 sells yet, ~mid-Aug).
- **Maximizer EMAIL**: breakout book from build_tier_book (scaled shares + vol-target); order = Maximizer read+book+RADAR first, then Preserver base; "Preserver Book"+"Preserve · 30% trailing" tag; ensemble "Approaching" watchlist dropped→breakout radar (portal-match); GAUGES (day-clock + cushion-to-stop table bars). Sample tested→erik@rigacap.com.
- **CONSISTENCY**: weight rounds WHOLE % everywhere; email cushion label "% to stop" (match app).
- **ADMIN APP (mobile)**: /tier-books returns weight-sorted `positions` + `return_pct` + vol_scale; Portfolio tab renders weight-sorted holdings (matches webapp), equity labeled "model book · from $100k · ±X%", fills→distinct "Recent transactions". OTA'd (group a00a446c) — Erik reopens app.
- **Bug fixes**: /api/auth/refresh mirrors /me sub-resolution (admin banner popped on idle); admin test emails bypass _is_trading_day (weekend sends no-op'd); Stripe-sourced admin stats.

## ▶ AWAITING ERIK
- Read of admin-app OTA + latest email sample. Optional Preserver header adjective ("Anchor"?; Core forbidden). Optional email gauge 3-segment bar (declined images). Optional modeled "your mirrored book value" what-if.

## ▶ STILL OPEN
- DOCS refresh: signal-intel + tech-arch UNCOMMITTED; PDF re-export + investor/marketing/sales sweep + 3 Qs. Commit ONLY design/documents (scratch scripts/ untracked).
- ADS: watch search terms + cost-per-signup.
