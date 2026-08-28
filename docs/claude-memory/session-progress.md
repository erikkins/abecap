---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 28 2026 (Mirror + SnapTrade LIVE + holdings-endpoint fixed; awaiting Erik refresh confirm)

## ▶▶ GO SLOW / BE PRECISE. NO "DWAP"/"t30v"/"tape" customer-facing. Worker payloads need TRUTHY value. SPA HARD-RELOAD after deploy. NEVER bare `lambda update-function-configuration --environment` (wipes 52 keys) — fetch-all+append+verify (Erik pre-authorized WITH verify-no-drops). Never log SnapTrade full URL (userSecret in query).

## ✅ SHIPPED main (all CI/CD ok): 71fa38e MIRROR; f4afca3 universe-split+combined book; daae733 per-book P/M badges; 3efd3a1 CSV upload; f17db20 SnapTrade connect; **7a3e3bc SnapTrade /positions/all fix + secret-log redaction**.

## 🔌 SNAPTRADE — WIRED, DEPLOYED, holdings endpoint FIXED. Awaiting Erik hard-reload confirm.
- Creds TEST key (in transcript+briefly in logs → ROTATE for prod): ClientID RIGACAP-LLC-TEST-EKAKS. On rigacap-prod-api env (54 keys, verified).
- Manual HMAC signing VERIFIED. Flow: register → login(customRedirect body) → accounts(GET /api/v1/accounts) → **positions: GET /api/v1/accounts/{id}/positions/all** (legacy /positions + /holdings are 410 for accounts made after 2026-05-11!). Positions under `results`; ticker = position.instrument.raw_symbol. Erik's Schwab IRA = 18 ETF positions (SCHI etc.), synced fine → will land in "Outside our universe", 0% mirrored ("not a mirroring account" = correct/expected).
- Files: backend/app/services/snaptrade_service.py (register_user/login_redirect_uri/all_holdings union across accounts; clean error raise, no secret leak). snaptrade_users table (user_id,user_secret; created via run_migration custom sql). Endpoints POST/GET /api/signals/mirror/snaptrade/connect|holdings (gated). Frontend Connect button→portal→/app?snaptrade=connected→merge; "Connected · <broker>" header (header-only source design). CSV + manual add/del coexist.
- run_migration event = 2 hardcoded ALTERs + custom `sql` param; big _run_schema_migrations runs via create_all on init.
- Stray test SnapTrade user rigacap-mirror-test-1 (harmless).

## 🪞 MIRROR (admin-only History tab): mirror-alignment vs combined entitled book (tier_book+preserver_book); 5 buckets via GET /api/signals/mirror-context (universe+entered sets); per-book P/M badges (Maximizer). localStorage rigacap_mirror_holdings.

## ⏭️ Also open: sector observatory DONE (distribution un-picked, chart https://claude.ai/code/artifact/64243537-cd5b-4250-afe3-0c2e3dc690ae); regime half of _mas SOT; DST Scheduler; scrub DWAP perf_numbers.js; old WhereStocksSit unmounted dead code. Worker invoke: rigacap-prod-worker --profile rigacap sync ok --cli-read-timeout 600+Bash 600000. WebSearch/WebFetch available (deferred, ToolSearch to load).
