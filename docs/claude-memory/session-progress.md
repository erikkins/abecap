---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 27 2026 (chart/previous-holds shipped; watching tonight's scan; SMCI-0 = stale-bundle)

## ▶▶ GO SLOW — verify don't assume. NO "DWAP"/"Wtd Avg" customer-facing. LONG JOBS async (--invocation-type Event). SPA: after a frontend deploy, a page RELOAD is needed (in-memory JS doesn't self-update); index.html is no-store + deploy invalidates CloudFront E1BA002TIC6UBN, so Cmd+R gets the fresh bundle.

## ✅ SHIPPED (Aug 27): data-integrity (frozen-universe/merger-heal/calendar, Aug 26). Chart: /api/stock/{sym}/previous-holds (ModelPosition+TierFill(max)+t30v WF + Maximizer breakout WF from maximizer_wf_trades.json=392trades/256syms). Overlay bands+dots+stagger labels+legend, foreground, 5Y range. Trade History clickable + ticker lookup + Days fix. ADMIN "Where Your Stocks Sit" widget (History tab, {isAdmin}).

## 🐞 SMCI=0 in widget — ROOT CAUSE FOUND = STALE IN-MEMORY BUNDLE (not a backend bug)
- Backend PROVEN fine: probe_maximizer_holds (worker, same image) reads SMCI→3; endpoint code+artifact+bucket+IAM all correct; API on latest image 1f96edd. CloudFront api.rigacap.com = Managed-CachingDisabled (not caching). Endpoint returns Cache-Control:no-store now.
- Erik's tab runs old c224e9c bundle (widget shipped there, cache-buster shipped LATER in 1f96edd). Re-clicking Check runs old JS → fixed URL → browser replays cached 200 → 0, and request never hits Lambda (no [prevholds-mx] log). FIX: RELOAD PAGE (Cmd+R) → loads 1f96edd bundle w/ `?t=Date.now()` cache-buster → fresh call → SMCI shows 3.
- ⏭️ After Erik reloads + it works: confirm [prevholds-mx] rows=3 in API logs, then STRIP debug scaffolding (probe_maximizer_holds handler in main.py + the `logger.warning("[prevholds-mx]...")` line).

## ⏳ IN FLIGHT — tonight's 4:30pm ET scan (cron `rigacap-prod-scanner` = cron(30 20) UTC; emails 6pm ET = cron(0 22)). Background monitor bjsi74csi watching dashboard.json generated_at→2026-08-27; re-invokes me on completion → run FULL verification (scan health, freshness, no phantom/ASST via calendar_audit re-run, book moves + WT/BHVN grade, email readiness). ~1.5h buffer before 6pm digest.

## 🕑 QUEUED (after tonight's email): DST-aware EventBridge SCHEDULER migration (aws_scheduler_schedule + America/New_York) — replaces fixed-UTC rules. Before Nov EST.
## 🎯 GRADE: [[project_maximizer_breakout_prediction_aug26]] WT/BHVN. OTHER: in-chart M badge; scrub DWAP perf_numbers.js; retire get_universe(); nasdaqtraded.txt ETF rule.
