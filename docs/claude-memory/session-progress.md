---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 26 2026 (frozen-universe FIXED + heal merger-aware + calendar audit done)

## ▶▶ GO SLOW — Erik sensitive after ASST bug hit 1st customer. Verify data. PARQUET not pickle. Never delete (additive; keep rollback). Erik catches design flaws — LISTEN. Calendar writes = confirm first (prior calendar-drop incident).

## ✅ DONE + LIVE
- **Frozen-universe root fix.** `universe_refresh_v2` (main.py) ranks CLEAN stock_universe_service list (NASDAQ/NYSE screener + EXCLUDED_PATTERNS, ETF-free) off FRESH fetch_raw_bars instead of frozen all_data.parquet. `heal_newcomers` full-backfills surgers (Alpaca union, raw). Ran write+heal → wrote `signals/universe-history/2026-08-26.json`, healed 206. scan_preview (new read-only handler) = 20 buys/0 suspicious. Was 64d stale, 206/600 membership wrong.
- **LIVE weekly cron REPOINTED** (Erik OK'd): `rigacap-prod-universe-refresh` (SAT 20:00 UTC) now fires `{"universe_refresh_v2":{"write":true,"heal_newcomers":true}}`. Terraform SSOT also updated. Verified FailedEntryCount:0.
- **Heal = MERGER-AWARE** (Erik's 2 catches both right: yfinance is split-adjusted→can't enter raw PITFWU; AZN 2026-02-02 = stock_MERGER not split→new identity). RIPPED OUT yfinance splice. Added `classify_short_symbols` (Alpaca corp-actions): corporate_action_boundary (merger/spinoff/removal→gated, correct) / rename_continuity (name change, same co, history under OLD ticker) / short_history (young). Verified 6 shorts: AZN=merger, SOLS=spinoff, CBRS/EQPT/FRVO=young, **SUNB=rename_continuity** (real rename caught). All safely gated; nothing synthetic ever enters raw store.

## 🚧 AWAITING ERIK GO — calendar fix (a WRITE, so paused)
- Ran read-only `calendar_audit` (new handler): 600 syms, 92 Alpaca splits, **3 MISSING from our calendar AND bars SPAN ex-date = maladjusted NOW**: **CRWD** fwd 4:1 (2026-07-02, px/dwap~0.49 → SLIPS the display gate = live exposure), **WETO** rev 100:1, **REAX** rev 10:1 (reverse→extreme ratio→display gate catches those 2).
- ROOT: **NO cron for calendar rebuild** — only ever manual → splits since last run missing.
- FIX (proposed, needs Erik OK — calendar writes): (1) `{"rebuild_corp_actions_calendar":{}}` — rebuild_calendar UNION-merges + backs up, never drops; adjust-at-read fixes all 3, NO bar rewrite. (2) wire WEEKLY calendar-rebuild cron (systemic fix). Then re-run calendar_audit → expect dangerous_count:0.

## ⏭️ NEXT / FOLLOW-UPS
- Erik "yes" → run calendar rebuild + wire weekly calendar cron + re-audit.
- rename_continuity CARRY-FORWARD unbuilt (SUNB-type: map old→new ticker from corp-action detail). Rare, low-pri.
- Cleanup later: read-only diag handlers (read_perf_test, fetch_scope_test, history_source_probe; keep scan_preview + calendar_audit). fold nasdaqtraded.txt ETF rule. X reply engine fix (paused).
- All ADDITIVE: all_data.parquet, old snapshots, existing PITFWU untouched → rollback available. AZN stored bars = clean post-merger Alpaca-raw (143); yfinance splice NEVER ran w/ execute.
