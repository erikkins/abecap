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
- Public copy: "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/sleeve/DWAP; steer toward Maximizer. Deploy = push main → "Deploy RigaCap" GHA. If push doesn't trigger: `gh workflow run "Deploy RigaCap" --ref main` (workflow_dispatch). **NEVER fire push + dispatch for the same commit → concurrent runs collide (Lambda ResourceConflictException). Serialize.** NEVER `aws lambda update-function-configuration --environment`. Worker = rigacap-prod-worker, AWS_PROFILE=rigacap, region us-east-1. DB read `{"db_read":"SQL"}` (cast ts ::text). SQL write `{"run_migration":true,"sql":...}`.
- OVERLAY SSOT = frontend/src/perf_numbers.js + backend/app/services/perf_numbers.py. Admin/test email → erik@rigacap.com.
- **One-click links MUST use https://api.rigacap.com (NOT FRONTEND_URL=rigacap.com — SPA host, /api/* 404s).**

## ✅ SHIPPED THIS SESSION (all live)
- REPLY SCANNER voice dialed in (plain-spoken Erik-founder, open-with-discipline, KO few-shot). Frictionless plumbing: batch approval email (send_reply_approval_batch, one digest w/ 1-click approve per draft), scan_replies scheduled 4x/day via CLI EventBridge (approve-first wk1), monthly recap KILLED.
- Anti-repeat: `_load_recent_replies` now date-window (days=8 ≈ last 5 reply-days, limit=30); fed into generator avoid_block. NO ellipsis/truncation — over-limit replies regenerate shorter or skip; approval email shows FULL originating post + FULL reply. (commit 615fbe6)
- **404 FIX (commit 81bda23):** reply-approval link was built on FRONTEND_URL (rigacap.com → 404). Fixed to api.rigacap.com. Added `{"resend_reply_approvals":true}` worker handler → RE-SENT the 5 pending drafts (incl $KO) with corrected links. Erik confirmed getting the 404; resend fired OK (resent:5).

## ▶ NEXT / IN FLIGHT
- Erik to test-approve the $KO reply end-to-end from the fresh (corrected) batch email.
- BUILD: 2 FORKS — Preserver vs Maximizer voice by THREAD REGISTER (anxiety/loss→Preserver; momentum/FOMO/growth→Maximizer); ONE reply per thread (dedup already prevents same-tweet-twice).
- After wk1: flip approve-first → default-post + 1-click kill.
- TERRAFORM RECONCILE: import the 2 CLI-made reply-scan EventBridge rules; remove monthly_recap block from main.tf (~1809).
- Fast-follows: og-card.png regen; SocialTab 2-tier library; ai_content social posts anti-repeat feed; newsletter anti-repeat window widen (Erik "curious" about every-other-week A/B oscillation).

## Notes / non-blockers
- Billing portal 400 = NOT a bug (erikkins@gmail.com orphan; don't touch billing code w/o ask). Ads: rigacap-signals-2tier live, recheck search terms ~Aug 8.
