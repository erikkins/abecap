---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 29 2026 (Mirror COCKPIT /app/next scaffolded w/ alignment eclipse; designing sleeve logic)

## ▶▶ GO SLOW / BE PRECISE. NO "DWAP"/"t30v"/"tape" customer-facing. Tool-safe = DESCRIPTIVE not prescriptive. Worker payloads TRUTHY. SPA HARD-RELOAD after deploy. NEVER bare `lambda ...--environment` (fetch-all+append+verify). Tier-preview param = `preview_tier` (NOT product-tier), admin-only.

## ✅ SHIPPED main (all CI/CD ok): full Mirror+SnapTrade (connect via snaptrade-react MODAL, multi-brokerage, disconnect DELETE /connection/{id}, one-shot-paint caches, KMS col-encryption INERT until prod-swap). **NEW: 6bc3255 /app/next COCKPIT scaffold; 0d2e1d6 forward preview_tier in cockpit.** Eclipse artifact: https://claude.ai/code/artifact/1a7cfbc6-fb0c-49db-8e74-b506e737d065.

## 🌑 /app/next COCKPIT (admin-gated route, ProtectedRoute→MirrorCockpit; non-admin→/app). AlignmentEclipse = React canvas (book=sun, portfolio=moon, alignment=eclipse, 100%=total+claret corona; driven by REAL pct via MirrorCheck onAlignment callback, tweened, reduced-motion). MirrorCockpit fetches /api/signals/dashboard (forwards preview_tier/state) → book/tier/regime → eclipse hero + live MirrorCheck below. KNOWN-ROUGH: paper Mirror card on dark sky (clash); eclipse+MirrorCheck gauge redundant. NEXT UI: dark-cohesion, drop small gauge, then DELTA FEED + DRIFT-over-time chart.

## 🧩 SLEEVE LOGIC — DESIGN LOCKED (thinking-through w/ Erik): "10% sleeve" = TWO axes. (1) FIDELITY=the eclipse = "of the book's positions, what share you hold" (moon covers sun; NOT diluted by extra holdings → total eclipse reachable by ALL, name-only, works for SnapTrade/CSV/manual, NO sleeve compensation needed for the visual). (2) SIZING="how much $ is in it" = separate readout, needs dollar amounts: SnapTrade=have it (units×price+balances); CSV=usually has Market Value col → ENHANCE parser to grab it; free-entry=names only→declare "~10%" or omit sizing. Honest combo: "you hold the book, and it's X% of your money." Erik pondering; likely next: teach CSV parser market-value + sizing readout + account-scoping picker (SnapTrade "Mirror in this account").

## 🪞 MIRROR (also on admin History tab): alignment vs combined entitled book (tier_book+preserver_book); 5 buckets via GET /api/signals/mirror-context; per-book P/M badges; 3 inputs (manual localStorage/CSV/SnapTrade). ECLIPSE motif = ORIGINAL (searched — no fintech uses eclipse-as-alignment; Venn/rings are the generic ancestors; ownable via consistent use).

## ⏭️ Also: prod-key swap runbook (prod SnapTrade key + KMS key + IAM + 3 env). Sector observatory DONE. regime _mas SOT; DST Scheduler; scrub DWAP perf_numbers.js; retire WhereStocksSit dead code. TEST SnapTrade key RIGACAP-LLC-TEST-EKAKS on rigacap-prod-api env (ROTATE at swap).
