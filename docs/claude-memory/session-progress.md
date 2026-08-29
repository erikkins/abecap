---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 28-29 2026 (Mirror+SnapTrade full connect/disconnect loop working via modal)

## ▶▶ GO SLOW / BE PRECISE. NO "DWAP"/"t30v"/"tape" customer-facing. Worker payloads TRUTHY. SPA HARD-RELOAD after deploy. NEVER bare `lambda ...--environment` (fetch-all+append+verify; Erik pre-authorized WITH verify-no-drops). Classifier blocks: direct prod-env mutate + SELECTing user_secret to transcript.

## ✅ SHIPPED main (all CI/CD ok): 71fa38e MIRROR; f4afca3 universe-split+combined book; daae733 P/M badges; 3efd3a1 CSV; f17db20 SnapTrade connect; 7a3e3bc positions/all+secret-redact; 152a9d2 group-broker+disconnect+clean-sourcing; 957e199 in-app MODAL (snaptrade-react); **d651651 disconnect DELETE /connection/{id}**.

## 🔌 SNAPTRADE — FULL LOOP WORKING. TEST key (ROTATE for prod): ClientID RIGACAP-LLC-TEST-EKAKS on rigacap-prod-api env (54 keys verified). Manual HMAC signing (no py SDK). **SnapTrade sunset MANY v1 endpoints for accounts made after 2026-05-11 — mapped current ones the hard way:** register POST /snapTrade/registerUser; login POST /snapTrade/login (customRedirect body); list GET /accounts; positions **GET /accounts/{id}/positions/all** (results[], ticker=instrument.raw_symbol); disconnect **DELETE /connection/{id}** (async/queued 200). Legacy /holdings,/positions,/authorizations DELETE all =410.
- Connect = IN-APP MODAL via snaptrade-react@3.2.5 (lazy-loaded, code-split, prerender-safe): backend /connect returns redirect_uri = modal loginLink; onSuccess→close+fetchSnapHoldings. Disconnect × per brokerage → optimistic header drop + re-fetch ~1.8s.
- backend/app/services/snaptrade_service.py (clean error raise = no secret leak — VERIFIED). snaptrade_users table. Endpoints /api/signals/mirror/snaptrade/connect|holdings|disconnect (gated). Sources GROUPED by authorization: [{institution,authorization_id,accounts:[names]}]. E*Trade obscures acct number → NAME is differentiator.
- Erik's live: Schwab IRA (18 ETFs→"outside universe") + E*Trade (2 accts=1 conn). Sandbox deleted during verify.

## 🪞 MIRROR (admin-only History tab): alignment vs combined entitled book (tier_book+preserver_book); 5 buckets via GET /api/signals/mirror-context; per-book P/M badges; 3 inputs coexist (manual localStorage / CSV / SnapTrade snapSymbols separate, effective=union; per-chip × only on manual).

## ⏭️ Also open: sector observatory DONE (dist un-picked, chart https://claude.ai/code/artifact/64243537-cd5b-4250-afe3-0c2e3dc690ae); regime half of _mas SOT; DST Scheduler; scrub DWAP perf_numbers.js; old WhereStocksSit unmounted dead code. WebSearch/WebFetch via ToolSearch. Worker invoke rigacap-prod-worker --profile rigacap.
