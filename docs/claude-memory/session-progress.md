---
name: session-progress
description: "Rolling snapshot of the current working session — accomplishments, in-flight work, key context for a fresh session"
metadata: 
  node_type: memory
  type: project
  originSessionId: b87c584c-343d-4a11-aca7-a450196570be
---

# Session progress — updated 2026-09-01 (~8PM ET)

## 🟢 DIGEST V3 — all fixes DEPLOYED; corrected tests sent to erik; awaiting his OK to broadcast
- **origin/main = 92ee9c8 "Daily digest v3: real sectors (S3 cache) + tier-wordmark masthead" — DEPLOYED (CI success).** All prior issues fixed:
  - S&P ▼−0.7% red (was ▲+0.0% bug) ✓
  - hook/regime/read on dark HTML band (not paper) ✓
  - Maximizer sectors now from S3 universe/sectors_cache.json (was blank) ✓
  - wordmark MASTHEAD at top (logo + tier wordmark baked into 14 per-tier hero PNGs hero_{regime}_{preserver|maximizer}.png) ✓
  - hook trailing period, dawn.png strip ✓
  - Maximizer uses own breakout read ✓; Preserver entry_status buckets ✓; ticker→chart links; NO truncation.
- **Just fired corrected tests to erik@rigacap.com both tiers (success). AWAITING Erik eyeball/approval.**
- **6PM auto-send STILL HELD:** EventBridge `rigacap-prod-daily-emails` DISABLED. Nothing to subscribers.

## ⏭️ ON ERIK'S "GO":
1. MANUAL full broadcast (late digest tonight): `{"daily_emails": {}}` (all subs) on live v3.
2. RE-ENABLE rule: `aws events enable-rule --name rigacap-prod-daily-emails --region us-east-1 AWS_PROFILE=rigacap` (tomorrow's 6PM).
3. Revert if needed: one-line call-site swap in email_service.send_daily_summary (digest_v3.build_digest_v3 → old generate_daily_summary_html, drop inline_images) + push; OR git revert.

## GIT STATE — what's in main
- ✅ Digest v3 FULLY in origin/main (deployed).
- ❌ Mirror NOT in origin/main — on branch **mirror-golive** (App.jsx +72: paid→mirror/free→signals default, two labeled Preserver/Maximizer sections, first+default tab, admin gate removed; build-verified). Kept separate on purpose. **ASKED ERIK if he wants Mirror pushed too** — awaiting. To push: `git checkout main && git merge --no-ff mirror-golive && git push` (or cherry-pick App.jsx). Branch digest-redesign-v2 = WRONG, never merge.

## KEY FILES
- backend/app/services/digest_v3.py (build_digest_v3, both tiers, picks hero_{regime}_{tier}.png, sectors from S3 cache), email_service.py (send_email inline_images; calls digest_v3; OLD fn intact=revert), scheduler.py (added spy_change_pct/regime_name to market_regime), backend/app/assets/email_heroes/ (14 hero PNGs + dawn.png).

## KEY FACTS / RULES
- Send tests to erik@rigacap.com (NOT ekins@cookma.com=this window's Claude login). AWS_PROFILE=rigacap. SMTP from worker Lambda env. Worker test path: {"daily_emails":{"target_emails":["erik@rigacap.com"],"force_tier":"preserver"|"maximizer"}} (target-only, bypasses gates).
- Single source of truth = today's dashboard; GENERATE NOTHING; NEVER truncate lists; each tier own market read.
- Brand claret/paper; NEVER navy/gold/olive; no DWAP/tape/PITFWU to customers.

## DEFERRED
- Mirror push (pending Erik). eclipse=Mirror alignment reward (live), orb=digests (regime). Remaining email surfaces: rest of drip (build_drip2.py), password reset, win-back.
