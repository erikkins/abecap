---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 26 2026 (FROZEN-UNIVERSE FIX shipped)

## ▶▶ GO SLOW — Erik sensitive after ASST data bug hit his 1st customer. Verify data before shipping. PARQUET not pickle. Never delete data (additive only, keep rollback).

## ✅ DONE THIS SESSION — frozen-universe root cause fixed
- **ROOT CAUSE:** `universe_refresh` (main.py:3318) ranked 60d volume off FROZEN `all_data.parquet` (Jun 15) → wrote `universe-history/{date}.json` → `_scoped_parquet_load` (data_export.py:177) loads only that top-600 → scan can only signal those. Membership frozen at Jun-15 volumes = surgers locked out. Diff proved snapshot 64d stale (2026-06-23), **206/600 (34%) wrong**.
- **FIX (Erik: frozen-universe first, ETF rule folded in later):** new `universe_refresh_v2` ranks the CLEAN `stock_universe_service` list (NASDAQ/NYSE screener + EXCLUDED_PATTERNS = already ETF-free, weekly-fresh) off a FRESH `fetch_raw_bars` (~1.3min, volume split-invariant). `heal_newcomers` full-backfills re-entering surgers' PITFWU (union merge, never deletes) so they clear the ≥250-bar gate + signal immediately. Commits 58c5d41, 69d8bec, + terraform.
- **RAN write+heal:** wrote `signals/universe-history/2026-08-26.json` (fresh top-600, ETF-free) + healed all 206 newcomers (1 new, 205 appended, 0 no_fetch). Verified CAT/AXP(2677)/CRWD(1812)/ABNB(1433)/CAVA(802) fresh thru today w/ indicators. **Tomorrow's scan will look WAY different** — CAT/CAVA/CRWD/ISRG/LLY/LOW/LULU/MA/AXP/ABNB/UNP now eligible.
- ETF POLICY CONFIRMED: ZERO ETFs, documented (signal-intelligence.html:840 "Universe hygiene", categorical). We ALREADY do this via stock_universe_service — don't reinvent. nasdaqtraded.txt = future HARDENING (blacklist→rule), not new.

## 🚫 BLOCKED — needs Erik (else regresses)
- Live weekly cron `rigacap-prod-universe-refresh` (SAT 20:00 UTC) still fires OLD `{universe_refresh:true}` (frozen). Terraform SSOT repointed to v2 (committed) but auto-mode BLOCKED the live `aws events put-targets`. **MUST repoint live before Saturday** or it overwrites our fresh snapshot. Cmd ready: `aws events put-targets --region us-east-1 --profile rigacap --rule rigacap-prod-universe-refresh --targets file:///tmp/uni_target.json`

## ⏭️ NEXT / OFFERED
- OFFERED to Erik: run tomorrow's scan candidates READ-ONLY now (ASST-safety — 206 new names = big rebalance; catch bad data before automated overnight send). Awaiting his go.
- Follow-ups: (a) AZN didn't heal (93 rows stale, display-gate-rejected; mega-cap, 420d fetch came back short → wider re-backfill / symbol-mapping check); 5 others <250 bars are genuinely young (safe). (b) daily `pitfwu_append` scopes data_cache.keys() → now auto-covers fresh top-600 (self-sustaining). (c) fold nasdaqtraded.txt ETF rule. (d) remove read-only diag handlers (read_perf_test, fetch_scope_test) later.
- All changes ADDITIVE: all_data.parquet, old dated snapshots, existing PITFWU bars untouched → rollback available.
