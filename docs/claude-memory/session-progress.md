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
- Public copy: "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/sleeve/DWAP; steer Maximizer. Deploy = push main → "Deploy RigaCap" GHA (or `gh workflow run`). NEVER push+dispatch same commit (serialize). NEVER `aws lambda update-function-configuration --environment` casually (restore script is the guarded exception). Worker=rigacap-prod-worker, AWS_PROFILE=rigacap. `{"db_read":"SQL"}` (ts ::text); `{"run_migration":true,"sql":...}`.
- OVERLAY SSOT = perf_numbers.js/.py. One-click links → api.rigacap.com. BRAND=claret/paper (Fraunces + IBM Plex; mono only for data). X banner=1500x500 native.
- TERRAFORM SAFE NOW ([[project_terraform_apply_unsafe]] resolved): both Lambdas have `lifecycle{ignore_changes=[image_uri,environment]}`. Full plan = 0-add/0-destroy/1-cosmetic (monthly_recap target). Still review plans; keep ignore_changes. tfvars gitignored (secrets local only).

## ✅ SHIPPED THIS SESSION (all live)
- REPLY scanner: plain voice, 5-day anti-repeat, no-ellipsis, 2 tier forks; X 403 → deep-link to X pre-filled composer ([[project_x_api_reply_403]]). Deep-link double-post FIX: repeat click shows "already opened" interstitial (+&force=1 reopen).
- CARDS: launch 1-5 growth-forward + 2x-crisp; og-card claret/paper centered/crop-safe (link-preview, auto-served); X banner knob rebuilt 1500x500. Anti-repeat extended to all own-post types (generate_post avoid_texts).
- OWN-SOCIAL CADENCE: autopost + 1-click kill, M/W/F alternating Preserver/Maximizer ({"autopost_own_social":{}}, EventBridge rigacap-prod-autopost-social). Live test post 715 scheduled ~7:27PM ET Aug 7.
- STALE DRAFTS cleaned (23 deleted; kept active reply history + 715).
- DR HARDENING: scripts/backup-lambda-env.sh + restore-lambda-env.sh (SSE S3 dr/lambda-env/, baseline uploaded worker55/api52). Weekly worker handler {"backup_lambda_env":true} + EventBridge rigacap-prod-backup-lambda-env cron(0 7 ? * SUN). main.tf blueprint: 15 live-only keys declared (var refs + vars, secrets sensitive), tfvars populated from live (gitignored). All CLI EventBridge rules imported into terraform.

## ▶ NEXT / IN FLIGHT
- Confirm 715 posted ~7:27PM ET + first real M/W/F autopost fires clean.
- OPTIONAL: already-replied-on-X interception (skip tweets @rigacap replied to manually) — reply tidiness, not built.
- GROWTH swings (north star 1→…→500 subs): testimonials/social-proof section; churn prevention (cancel survey + win-back); ads recheck search terms ~Aug 8.
- Reply auto-post paid X tier = OPEN (verify before paying).

## Notes
- Billing portal 400 = NOT a bug (erikkins@gmail.com orphan; don't touch billing w/o ask).
