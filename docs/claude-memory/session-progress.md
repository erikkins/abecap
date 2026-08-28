---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 28 2026 (Mirror tab + SnapTrade LIVE; awaiting Erik's live Schwab connect test)

## ▶▶ GO SLOW / BE PRECISE. NO "DWAP"/"t30v"/"tape"/"Consider adding" customer-facing. NO model change w/o full universe re-run. Worker payloads need TRUTHY value. SPA HARD-RELOAD after deploy. NEVER bare `lambda update-function-configuration --environment` (wipes all 52 keys) — always fetch-all+append+verify.

## ✅ SHIPPED main (all CI/CD ok): 71fa38e MIRROR tab; f4afca3 universe-split+combined book; daae733 per-book P/M badges; 3efd3a1 CSV upload; **f17db20 SnapTrade connect (multi-brokerage, read-only)**.

## 🪞 MIRROR TAB (admin-only, History tab) — mirror-alignment vs live model book (tier_book+preserver_book combined for entitlement). 3 input methods coexist: manual add/del, CSV upload, SnapTrade. 5 buckets (aligned/in-book-not-held/drifted/in-universe-no-signal/outside-universe) via gated GET /api/signals/mirror-context (universe+entered sets). Per-book P/M badges (Maximizer). localStorage rigacap_mirror_holdings.

## 🔌 SNAPTRADE — WIRED + DEPLOYED, awaiting Erik's live connect test:
- Creds (TEST key, in transcript — ROTATE for prod): ClientID RIGACAP-LLC-TEST-EKAKS. Added to rigacap-prod-api env (54 keys, verified no drops).
- Signing VERIFIED LIVE (manual HMAC-SHA256, no SDK): register/login(needs customRedirect body)/accounts(GET /api/v1/accounts) all 200. OLD /holdings is 410 → use accounts + per-account /positions union.
- backend/app/services/snaptrade_service.py (register_user, login_redirect_uri, all_holdings=union across accounts). snaptrade_users table (created via {"run_migration":true,"sql":"CREATE TABLE..."} — NOTE: run_migration event only runs 2 hardcoded ALTERs + custom `sql`; the big _run_schema_migrations list runs via create_all on init). Endpoints POST/GET /api/signals/mirror/snaptrade/connect|holdings (gated). Frontend Connect button → portal → /app?snaptrade=connected → merge + "Connected" header (header-only source design Erik chose).
- ⚠️ UNVERIFIED: real positions JSON shape (_extract_symbol is defensive, tries symbol.symbol/symbol.raw_symbol/etc). If Erik's Schwab ETFs don't populate after connect → pull real positions payload + fix parser. Expect his ETF IRA → "outside universe", 0% mirrored ("not a mirroring account" = correct).
- Stray test SnapTrade user "rigacap-mirror-test-1" registered during testing (harmless).

## ⏭️ Also open: sector observatory DONE (distribution un-picked); regime half of _mas SOT; DST Scheduler; scrub DWAP perf_numbers.js; old WhereStocksSit unmounted dead code. Worker invoke: rigacap-prod-worker --profile rigacap, sync ok w/ --cli-read-timeout 600+Bash 600000. Local venv Py3.9 str|None = RED HERRING (prod 3.12).
