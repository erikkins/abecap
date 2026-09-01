---
name: session-progress
description: "Rolling snapshot of the current working session — accomplishments, in-flight work, key context for a fresh session"
metadata: 
  node_type: memory
  type: project
  originSessionId: b87c584c-343d-4a11-aca7-a450196570be
---

# Session progress — updated 2026-09-01 (~6:15PM ET)

## 🔴 DIGEST V3 DEPLOYED but DESIGN DRIFTED — fixing before any broadcast
- v3 dashboard-sourced digest PUSHED + DEPLOYED (origin/main 26efa92, CI Deploy RigaCap SUCCESS 4m19s @22:07 UTC). Files: backend/app/services/digest_v3.py (new, +254), email_service.py (+45), 7 hero PNGs backend/app/assets/email_heroes/. Clean — NO App.jsx swept in (Mirror protected). Old digest fn kept = revert.
- **Erik: "the design changed a bit in the deployed version"** — the fork's digest_v3.py did NOT exactly match the APPROVED local renders (scratchpad digest_dash.py Preserver + build_max.py Maximizer). Erik INTERRUPTED my worker-test invoke. NEED to reconcile digest_v3.py to match digest_dash/build_max EXACTLY, then redeploy. Asked Erik to point out what drifted (or I diff v3 vs local myself).
- **6PM auto-send is HELD (safe):** EventBridge rule `rigacap-prod-daily-emails` DISABLED. NOTHING went to subscribers. Zero blast radius. (Balk-gate = _validate_data_freshness is for data integrity, not this deploy hold — Erik confirmed that's the nightly "flag" but it's data-driven, auto.)

## SEQUENCE TO FINISH (in order)
1. Reconcile digest_v3.py → exactly match approved digest_dash.py (Preserver) + build_max.py (Maximizer): market read TOP, hidden-empty Today's Moves, entry_status buckets (fresh/actionable=buy zone, extended=holding NOT buys), ticker→chart links, NO truncation, real sectors, Maximizer uses its OWN briefing (not Preserver market_context), days_held→~days-to-time-stop, per-regime ORB hero inline cid (dynamic hook/read = live HTML, not baked).
2. Redeploy (push main → CI ~4min). Worker-test both tiers to erik@rigacap.com: {"daily_emails":{"target_emails":["erik@rigacap.com"],"force_tier":"preserver"|"maximizer"}}.
3. Erik OK → MANUAL full broadcast {"daily_emails":{}} (all subs) on deployed v3.
4. RE-ENABLE rule: `aws events enable-rule --name rigacap-prod-daily-emails --region us-east-1 AWS_PROFILE=rigacap` (tomorrow's 6PM).
5. If bad: revert email_service.py call-site to old fn + push.

## GIT / OTHER
- frontend/src/App.jsx STAGED uncommitted = CLEAN Mirror go-live (paid→mirror/free→signals default, two labeled sections, first tab, admin gate removed, build-verified). Push SEPARATELY when ready (Erik approved). Branches: mirror-golive, digest-redesign-v2 (WRONG, never merge), digest-live-v3.
- Port fork adf7514ba80050117 may still be finishing/reporting; its report will detail what v3 built (compare to approved).

## KEY FACTS
- Data: /tmp/dash_today.json, /tmp/mbs_out.json (max book+briefing), /tmp/maxprev_out.json (radar), /tmp/sectors.json. Rules: single source=today's dashboard, GENERATE NOTHING, NEVER truncate, each tier own read.
- Send tests to erik@rigacap.com (NOT ekins@cookma.com). AWS_PROFILE=rigacap. SMTP from worker Lambda env.
- Brand claret/paper; NEVER navy/gold/olive; no DWAP/tape/PITFWU to customers.
