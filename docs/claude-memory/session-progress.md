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
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC. PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Web deploy=push main→"Deploy RigaCap" GHA + smoke curl api.rigacap.com/api/market-data-status→200. Migration-first DB. NEVER lambda update-function-configuration --environment / terraform apply w/o plan. mobile-admin/=Expo EAS-OTA channel=preview. TIER_SERVING=true; 0 real external subs. Admin/test emails→erik@rigacap.com.

## ✅ SHIPPED (all live, deploys GREEN) — served-Maximizer "both books" portal + email
- **Portal layout FINAL:** [Preserver book | Maximizer book + Rotation watch] side-by-side, books-on-top, ONE capital control, Market Read PIXEL-aligned (JS measure `[data-market-read]`), whole shares NO ≈ ($ = exact target), candidates FULL-WIDTH below (no canyon). Date tag + "Last updated" on ONE row (two-book view). Rotation watch fills DYNAMICALLY (measured `rotRows` useLayoutEffect → rows reach `[data-books-left]` Preserver bottom; `data-rot-list/-max/-row`). Erik: NO wasted space.
- **Rotation watch** = 5+ nearest 29-day time-stops, urgency-sorted (book is weight-sorted), live hold-clocks, no WF. **Recently-closed** ("previously sold") = wired + DORMANT (prod: 15 maximizer BUYS Jul15-28, 0 sells; first exits ~mid-Aug) → auto-splits the space when sells log, both portal + email. No deploy needed.
- **Maximizer EMAIL:** breakout book renders from build_tier_book('maximizer') (portal parity) → per-name weight·whole-shares·$ scaled to recipient capital + vol-target exposure line; shared `_holding_row`; `breakout_tier_book` threaded through scheduler (built $100k, rescaled). Sent SAMPLE to erik@rigacap.com — Erik verifying.
- **Banner bug FIX:** /api/auth/refresh now mirrors /me subscription resolution (admin synthesis+override) — 11-min background refresh had been wiping admin's synthesized sub → SubscriptionBanner popped at top on idle.
- Earlier: Stripe-sourced admin stats (0 paid/$0 MRR, internal excl); admin app rebuilt+OTA'd.

## ✅ Aug 15 late — sample email fixed + delivered; rotation floor
- **280c27e**: Rotation watch uses `Math.floor(avail/rowH)` (was round → 1 row too many, overshot past Preserver). Now near-but-not-past.
- **Weekend send bug FIXED**: send_daily_emails bailed at `_is_trading_day` on Saturday; target_emails (admin test) now bypasses trading-day check too (mirrors freshness bypass). Maximizer SAMPLE then DELIVERED — email_events shows sent 18:31 + opened 18:32 UTC to erik@rigacap.com. (Transport = SMTP/aiosmtplib, NOT SES. Scheduler `logger` output NOT captured in CloudWatch — only print-based lines show; diagnose sends via email_events table, cols: email_address/event_type/email_type.)

## ✅ Maximizer email REORDER + parity DONE (commit 2aace8f, live; sample sent erik@rigacap.com 18:48)
- Two-book email now LEADS with Maximizer: breakout read → breakout book (scaled shares + vol-target) → breakout RADAR ("◆ Approaching a breakout trigger", new _breakout_radar_section, built in scheduler as max_radar + threaded breakout_radar) → THEN Preserver base (read → Preserver book → signals). Section order assembled in `_middle` (email_service, before html f-string); Preserver-only emails unchanged.
- Renamed "Our Book" → "Preserver Book" + tag "Preserve · 30% trailing" (parallels Maximizer "Grow · held ~29d"). Erik OK'd "Preserver Book"; offered header adjective swap (e.g. "Preserver Anchor Book") — "Core" FORBIDDEN (internal). AWAIT if he wants an adjective.
- DROPPED the ambiguous ensemble "Approaching — Nearing trigger" watchlist from two-book email (portal doesn't show it to served users + label collided with breakout radar). Kept for Preserver-only.
- Portal left side-by-side (Preserver left/Maximizer right) — NOT reordered (Erik: only the email).

## ▶ STILL OPEN
- Recently-closed (email+portal) auto-activates when real Maximizer sells log (~mid-Aug; 15 buys, 0 sells).
- DOCS refresh: signal-intel + tech-arch edits UNCOMMITTED; PDF re-export + investor/marketing/sales sweep + 3 Qs. Commit ONLY design/documents (scratch scripts/ untracked).
- ADS: watch search terms + cost-per-signup.
