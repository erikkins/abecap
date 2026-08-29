---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 28-29 2026 (Mirror+SnapTrade DONE — connect/disconnect/modal + one-shot paint)

## ▶▶ GO SLOW / BE PRECISE. NO "DWAP"/"t30v"/"tape" customer-facing. Worker payloads TRUTHY. SPA HARD-RELOAD after deploy. NEVER bare `lambda ...--environment` (fetch-all+append+verify; Erik pre-authorized WITH verify-no-drops). Classifier blocks: direct prod-env mutate + SELECT user_secret to transcript + destructive DELETE on his acct.

## ✅ SHIPPED main (all CI/CD ok): ...f17db20 SnapTrade connect; 7a3e3bc positions/all+secret-redact; 152a9d2 group-broker+disconnect+clean-sourcing; 957e199 in-app MODAL (snaptrade-react); d651651 disconnect DELETE /connection/{id}; 06c30fc two-stage-load fix (cache snap + gate + parallelize); **2ec5854 finish one-shot paint (cache ctx+best%, gate Connected header)**.

## 🔌 SNAPTRADE — FULL LOOP WORKING + POLISHED. TEST key (ROTATE for prod): ClientID RIGACAP-LLC-TEST-EKAKS on rigacap-prod-api env (54 keys verified). Manual HMAC (no py SDK). **SnapTrade sunset v1 endpoints for accts made after 2026-05-11 — CURRENT paths (verified live):** register POST /snapTrade/registerUser; login POST /snapTrade/login (customRedirect body); GET /accounts; positions GET /accounts/{id}/positions/all (results[], ticker=instrument.raw_symbol); disconnect DELETE /connection/{id} (async 200). Legacy /holdings,/positions,/authorizations DELETE all =410.
- Connect = in-app MODAL (snaptrade-react@3.2.5, lazy/code-split): /connect returns redirect_uri=loginLink; onSuccess→close+refetch. Disconnect × per brokerage → optimistic drop + refetch. Backend all_holdings parallelizes per-account /positions/all (asyncio.gather); sources grouped by authorization [{institution,authorization_id,accounts:[names]}].
- **One-shot paint:** localStorage caches — rigacap_mirror_snap (holdings+sources), _mirror_ctx (universe/entered sets), _mirror_best (drifted %). Hydrate→instant complete render; background refresh reconciles. `ready=ctx&&snapReady` gates gauge+buckets+Connected header. Skeleton only cold first load.
- Erik live: Schwab IRA(18 ETFs→outside universe) + E*Trade(2 accts=1 conn). E*Trade obscures acct#→NAME differentiates. Sandbox deleted.

## 🪞 MIRROR (admin-only History tab): alignment vs combined entitled book (tier_book+preserver_book); 5 buckets via GET /api/signals/mirror-context; per-book P/M badges; 3 inputs coexist (manual localStorage / CSV / SnapTrade snapSymbols; effective=union; per-chip × only manual).

## ⏭️ Also open: sector observatory DONE (dist un-picked, https://claude.ai/code/artifact/64243537-cd5b-4250-afe3-0c2e3dc690ae); regime half of _mas SOT; DST Scheduler; scrub DWAP perf_numbers.js; old WhereStocksSit unmounted dead code. WebSearch/WebFetch via ToolSearch.
