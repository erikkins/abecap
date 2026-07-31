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
- Public copy: "walk-forward" not "backtest"; never t30v/Core/sleeve/DWAP; "warts" OK in blog voice NOT ads. Survivorship-free OK in customer copy now. STEER toward Maximizer (+$100/mo upsell). Deploy=push main→"Deploy RigaCap".

## OVERLAY CANONICAL (SSOT: frontend/src/perf_numbers.js + backend/app/services/perf_numbers.py + docs/numbers-citations-registry.md §1)
- Preserver 21yr 7.7/0.87/−13.7, 5yr 13.0/1.28/−12.9, typical-yr +10.7%. Maximizer 21yr 13.5/0.93/−20.8, 5yr 31.4/1.51/−14.9, typical-yr +26.5%, 2022 −6.4% (upsell hero). S&P 21yr 9.8/−55. Raw-mom 13.2/0.69/−57.

## ✅ PUBLIC PAGES CONVERTED + LIVE (f51bbac deployed SUCCESS): Landing, TrackRecord, Methodology, ForAdvisers, 3 blogs (Blog2022→Maximizer hero). 
## ⬜ FAST-FOLLOW (not blocking ads): backend emails (email_service/newsletter_generator:121/ai_content:56 → hardcode overlay), SocialTab post library (2-tier REWRITE, still 8.3/19), og-card.png regen ("backtest"+"19%") + launch-1..5.png. Knob PNG fine.

## ✅ GOOGLE ADS FULLY LAUNCHED (Jul 31) — `rigacap-signals-2tier` LIVE. Doc = design/documents/google-ads-2tier-campaign.md.
- Settings: Search only (Display+Search-partners OFF), US, English, **$25/day campaign-level** (Google pushed $60; held 25), bidding=**Max Clicks + $6.50 CPC cap**, AI Max OFF. Conversion tracking already live (**Sign-up=Active+Primary**; old 0-conv was a REAL land→signup leak, not tracking).
- Ad Group "Preserve" → `/track-record?...utm_campaign=preserve` (our phrase/exact kws, 15 HL + 4 desc, strength "Average"). Ad Group "Maximize" → `/?...utm_campaign=maximize` (lead "In 2022 We Lost 6%" + "Beat the S&P, Less Risk"). BOTH SHARE the $25 budget → if it favors Preserve, split into 2 campaigns to force Maximize spend.
- Campaign-level extensions DONE: sitelinks (/track-record, /methodology, /for-advisers, /), negatives list `rigacap-negatives` (bare "volatile" kept OUT), 6 callouts (trimmed "Execute at your broker" to ≤25 char), structured snippet Types=Preserve/Maximize.
- LAUNCH GATE: **$350 / 2-week kill-or-scale — watch COST-PER-SIGNUP, not clicks.** If clicks but ~0 signups by ~$150 → land→signup/price leak (the real risk) → STOP + fix funnel, don't feed budget. Raise budget / switch to Max Conversions or tCPA only after ~15–30 real signups.

## ⬜ TOMORROW (Aug 1) — Erik deferred: (1) backend EMAIL number conversion (email_service ≥6 spots, newsletter_generator:121, ai_content:56 → hardcode overlay); (2) regen OG assets (og-card.png says "backtest"+"19%", launch-1..5.png) + SocialTab post library 2-tier rewrite (still 8.3/19). Ad clickers may hit these → clean before traffic compounds.

## CLEANUP temp scratch: scripts/{tier_vintages_today.py, tier_vintages_daily_today.py, recompute_canonical.py, canonical_recompute.json, tier_curves_21y_today.json, tier_curves_21y.json.bak-may29, breakout_*.py}. KEEP scripts/overlay_canonical.json.
