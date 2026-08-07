---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 7 2026

## Frozen spec (load-bearing)
- Public copy: "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/sleeve/DWAP; steer toward Maximizer. Deploy = push main → "Deploy RigaCap" GHA. If push doesn't trigger: `gh workflow run "Deploy RigaCap" --ref main`. **NEVER fire push + dispatch for the same commit → concurrent runs collide (Lambda ResourceConflictException). Serialize.** NEVER `aws lambda update-function-configuration --environment`. Worker = rigacap-prod-worker, AWS_PROFILE=rigacap, us-east-1. DB read `{"db_read":"SQL"}` (cast ts ::text). SQL write `{"run_migration":true,"sql":...}`.
- OVERLAY SSOT = perf_numbers.js/.py. Admin/test email → erik@rigacap.com. One-click links MUST use https://api.rigacap.com (NOT FRONTEND_URL — SPA host, /api/* 404s).

## ✅ SHIPPED THIS SESSION (all live)
- REPLY SCANNER: plain-spoken Erik-founder voice (open-with-discipline, KO few-shot); batch approval email (one digest); scan 4x/day via CLI EventBridge (approve-first wk1); monthly recap KILLED. Anti-repeat over last ~5 reply-days (days=8 window); NO ellipsis/truncation — full post + full reply in email, over-limit regenerates or skips. (615fbe6)
- 404 FIX (81bda23): approval link was on rigacap.com → fixed to api.rigacap.com. `{"resend_reply_approvals":true}` worker handler recovers pending drafts w/ correct links.
- **2 FORKS (fab89f4):** `classify_tier()` picks ONE voice by thread register — FOMO/breakout→Maximizer angle (ride-it-without-giving-back, edge=exit); fear/loss/drawdown→Preserver angle (protect gains, exit on rule not story); neutral→Preserver. Angle overlay in prompt (no product names). One reply/thread (dedup). Email shows colour-coded tier chip. No DB migration (tier recomputed). Smoke-tested OK.
- **🚨 X 403 FIX (78f09b5) — KEY:** X Free API tier CANNOT auto-post replies to 3rd-party tweets (403 not-authorized "can only reply where mentioned/author") — this is WHY Erik copy-pasted, NOT our code. Solution = DEEP LINK: approval button → `GET /compose-email?token=` → marks handled → 302 to X intent `twitter.com/intent/tweet?in_reply_to={id}&text={enc}` → composer opens as reply, pre-filled → Erik hits Post. $0, no API, no 403, no paste. Publish cron + auto_schedule_drafts both exclude contextual_reply. See [[project_x_api_reply_403]]. Our OWN posts still API-post fine.

## ▶ NEXT / IN FLIGHT
- Erik to TEST $KO deep-link end-to-end (tap → X opens reply pre-filled → Post). Verify intent URL threads as a reply on his device. All 5 drafts (incl $KO, post 712 — was reset from stranded 'approved') resent w/ deep-link + tier chips.
- OPEN decision: paid X tier (~$200/mo) for true auto-post — VERIFY Basic actually lifts the 403 before paying (X may still block automated replies). Not urgent.
- After wk1: (deep-link is already human-in-loop, so "flip approve-first→auto-post" is moot unless paid tier).
- TERRAFORM RECONCILE: import 2 CLI reply-scan EventBridge rules; remove monthly_recap block (~1809).
- Fast-follows: og-card.png regen; SocialTab 2-tier library; ai_content posts anti-repeat feed; newsletter anti-repeat window widen (Erik "curious" re A/B oscillation).

## Notes / non-blockers
- Billing portal 400 = NOT a bug (erikkins@gmail.com orphan; don't touch billing w/o ask). Ads: rigacap-signals-2tier live, recheck search terms ~Aug 8.
