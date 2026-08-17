---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 14–17 2026

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC. PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Web deploy=push main→"Deploy RigaCap" GHA + smoke api.rigacap.com/api/market-data-status→200. Migration-first DB. NEVER lambda update-function-configuration --environment / terraform apply w/o plan. Admin app=mobile-admin/ Expo EAS OTA (`eas update --channel preview`; Erik reopens app). TIER_SERVING=true; 0 real external subs. Admin/test emails→erik@rigacap.com. Email=SMTP/aiosmtplib (NOT SES); scheduler logger NOT in CloudWatch → diagnose sends via email_events table. NO Google Ads API wired (screenshots only; Erik applies ad changes in UI).
- **RigaCap is a PUBLISHER** (signals-only, no custody). No "your portfolio value" screen by design. Book equity = model MTM from $100k CAP0, NOT per-user.

## ✅ SHIPPED Aug 14–15 (all live) — served-Maximizer portal + email + admin app + fixes
- Portal: [Preserver | Maximizer + Rotation watch] side-by-side, books-on-top, one capital control, Market Read pixel-aligned, whole shares no ≈, candidates full-width, Rotation watch measured-fill. Recently-closed DORMANT (0 sells, ~mid-Aug).
- Maximizer EMAIL: Maximizer-first order (read+book+radar), Preserver base below, "Preserver Book"+"Preserve · 30% trailing", breakout radar replaces ensemble "Approaching", GAUGES (day-clock + cushion table bars). Weight rounds WHOLE % everywhere; cushion label "% to stop".
- Admin app: weight-sorted `positions` + equity labeled "model book · from $100k · ±X%" + "Recent transactions" block (OTA'd).
- Bug fixes: /api/auth/refresh admin sub-synthesis (banner-on-idle); admin test emails bypass trading-day gate (weekend no-op); Stripe-sourced admin stats.

## ▶ Aug 15 — SIS funnel WATCH (see [[project_sis_funnel_watch]])
- Baseline 44 landers, fold-through 36%, 0 reached Stripe, engagement was 1-day blip. Erik: let it run ~1wk/~150 landers; redesign above-the-fold if fold-through stays mid-30s. Re-pull via worker db_read on page_views.

## ▶ Aug 17 — AWAITING ERIK: Google Ads conflicting negative
- Phrase negative "trading signals" blocks positive kw "momentum trading signals". I recommended KEEP the negative + DELETE the keyword (day-trader crowd is off-thesis/churny; filtration > conversion). Alt: Apply (remove neg) + add specific negatives (day trading, forex, crypto, free signals, telegram, discord, scalping, intraday, options signals). Erik decides in UI.

## ▶ STILL OPEN
- DOCS refresh: signal-intel + tech-arch UNCOMMITTED; PDF re-export + investor/marketing/sales sweep + 3 Qs. Commit ONLY design/documents (scratch scripts/ untracked).
- Optional Preserver email header adjective; modeled "your mirrored book value" what-if.
