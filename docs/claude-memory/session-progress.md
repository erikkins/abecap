---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 27 2026 (data-integrity DONE; CHART overlay iterating; tier-teaser data READY)

## ▶▶ GO SLOW — verify don't assume. PARQUET not pickle. NO "DWAP"/"Wtd Avg" customer-facing ([[feedback_no_dwap_customer_facing]]). Empty {} payloads FALSY→mangum→worker-errors alarm. LONG JOBS: invoke ASYNC (--invocation-type Event); 2-min bash wrapper SIGTERMs sync invokes.

## ✅ DATA-INTEGRITY (Aug 26, live): frozen-universe fix (`universe_refresh_v2`, cron SAT20:00); merger-aware heal; calendar completeness + cron SAT18:00 scope=full.

## ✅ CHART FEATURE (Aug 27, shipped/deploying)
- Backend `GET /api/stock/{symbol}/previous-holds` (main.py ~12430): Preserver=ModelPosition(live) + Maximizer=TierFill(maximizer,paired) + WF=WalkForwardSimulation.trades_json(is_daily_cache). gain/loss=(exit/entry-1)*100.
- Frontend StockChartModal: killed +20%; labels Average price/Entry trigger; verticals Crossed trigger/Buy signal. PREVIOUS-HOLDS overlay: shaded bands + gain/loss labels (VERTICAL-STAGGER custom renderer, fixes nested collisions) + entry/exit DOTS (plain circles: entry filled / exit ring, P&L-colored — custom-shape triangles were dropping) + legend "solid=live·dashed=backtested" + toggle. MOVED overlay to FOREGROUND (after price Area) so volume bars don't paint over dots.
- Trade History: rows clickable→chart; ticker LOOKUP search; Days column fixed (compute from dates).
- MRVL data check: 2 live holds (~$309 Jun→$217 Jul, −29.6%) + 1 WF ($114 Apr→$217 Jul, +90%); all exit same day→exit dots stack (expected).

## 🔧 TIER TEASER — DATA READY, wiring is NEXT (Erik: build it)
- ✅ Maximizer WF trades ARTIFACT built: `signals/maximizer_wf_trades.json` = 392 trades / 256 symbols (2021→2026: NVDA/SMCI/LRCX/AMAT/NOW/ON/FLEX...). replay_sleeve instrumented (collect side-channel, prod-safe) + piggybacked on build_signaled_symbols_5y. Breakout=Maximizer-only → NO dedup.
- ⏭️ WIRE: (a) endpoint reads maximizer_wf_trades.json → emit is_walkforward Maximizer holds (tier='maximizer'); (b) frontend "M" badge + "Maximizer add-on" framing (upsell teaser for Preserver viewers), compliance: label as other-product/backtested. DESIGN: show viewer's tier + badged divergent holds (not gate/flood).

## 🎯 GRADE after 4pm scan: [[project_maximizer_breakout_prediction_aug26]] WT/BHVN → maximizer_preview + build_todays_actions.
## ⏭️ OTHER: scrub DWAP perf_numbers.js; retire get_universe(); nasdaqtraded.txt ETF rule; rename_continuity(SUNB); remove diag handlers.
