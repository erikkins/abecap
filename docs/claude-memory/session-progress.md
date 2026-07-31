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

## ▶ NOW: GOOGLE ADS — building `rigacap-signals-2tier` in the UI (no API; Erik clicks, screenshots, I confirm). Doc = design/documents/google-ads-2tier-campaign.md.
- CONFIRMED settings: Search only (Display+Search-partners OFF), US, English, $25/day, bidding=**Clicks + $6.50 max-CPC cap** (0 conv history), AI Max OFF, conversion tracking already live (**Sign-up=Active+Primary**; old 0-conv was a REAL land→signup leak, not tracking).
- ✅ CAMPAIGN PUBLISHED (one flaky save failed + lost ad-group progress once; re-entered fine). Budget $25/day campaign-level (Google pushed $60; held at 25). Preserve+Maximize SHARE budget → if it favors Preserve later, split into 2 campaigns to force Maximize spend.
- ✅ AD GROUP 1 "Preserve": our phrase/exact keywords (deleted Google's broad-match junk). RSA: Final URL `https://rigacap.com/track-record?...utm_campaign=preserve`; 15 headlines + 4 descriptions; display path signals/preserve (cosmetic, not real routes); no pins; strength "Average" (fine). Sitelinks (campaign-level, https://): /track-record, /methodology, /for-advisers, / (dropped "Pricing" — route unconfirmed).
- ▶ NOW: **Ad Group 2 "Maximize"** being added — Final URL `https://rigacap.com/?...utm_campaign=maximize`; keywords + 15 headlines (lead "In 2022 We Lost 6%" + "Beat the S&P, Less Risk") + 4 descriptions all delivered to Erik.
- ⬜ THEN: shared negatives list `rigacap-negatives` (from doc; keep bare "volatile" OUT); callouts + structured snippet (Preserve/Maximize); review + set live.
- LAUNCH GATE: $350/2-week kill-or-scale; if clicks but ~0 signups at ~$150 → funnel/price leak (the real risk) → stop+fix. Switch to Max Conversions/tCPA after ~15-30 conversions.

## CLEANUP temp scratch: scripts/{tier_vintages_today.py, tier_vintages_daily_today.py, recompute_canonical.py, canonical_recompute.json, tier_curves_21y_today.json, tier_curves_21y.json.bak-may29, breakout_*.py}. KEEP scripts/overlay_canonical.json.
