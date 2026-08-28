---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 28 2026 (Mirror tab loved by Erik "this is so cool"; SnapTrade is next when he's ready)

## ▶▶ GO SLOW / BE PRECISE. NO "DWAP"/"t30v"/"tape"/"Consider adding" customer-facing. NO model change w/o full universe re-run. Research MUST map to prod. Worker payloads need TRUTHY value ({} → mangum error). SPA HARD-RELOAD after deploy.

## ✅ SHIPPED main (all CI/CD ok): 875d370 voice-guard(tape); affe78d sector_rotation_study; 0b3c40c mean-rev; 06bb951 weekly regime; 71fa38e MIRROR tab; f4afca3 mirror universe-split+combined book; daae733 mirror per-book P/M badges.

## 🪞 MIRROR TAB (admin-only, History tab, MirrorCheck) — mirror-alignment vs live model book; manual add/del→localStorage; tool-safe copy+disclaimer. Built out via 3 Erik catches:
1. MSFT "outside universe" wrong → gated GET /api/signals/mirror-context (universe + ever-traded 'entered' sets; reuses _load_overlay_sets; membership only; reusable for SnapTrade). 5 buckets: aligned / in-book-not-held / drifted(entered) / in-universe-no-signal(MSFT/META/TSLA) / outside(ETFs).
2. AAPL "magically in model" → NOT new: entered 2026-07-20 (verified via worker {"model_portfolio":{"action":"summary","portfolio_type":"live"}}). Was mis-hidden b/c Mirror only compared vs Maximizer breakout book; FIXED = mirror vs FULL entitlement (tier_book + preserver_book combined).
3. "which book?" → per-book split preserverSyms(base)/maximizerSyms(breakout); gauge line "Preserver base X/N · Maximizer breakout Y/M"; P/M badge on aligned+in-book chips (Preserver users see none). Cross-tier for Preserver users NOT surfaced (paid-book leak). Old WhereStocksSit still defined+UNMOUNTED (dead code; offered delete). tier_book=served, preserver_book=base (present for Maximizer, signal_source='both').

## ⏳ NEXT (Erik enthused, NO RUSH — savor moment): SnapTrade wiring (Dev acct 5 free conns) — "connect broker" pulls holdings into same per-book alignment; Schwab ETF IRA → ETFs→outside universe, 0% mirrored ("not a mirroring account"), no look-through. Also offered: "entered [date]" note on aligned chips.

## 📊 SECTOR OBSERVATORY COMPLETE (chart same URL https://claude.ai/code/artifact/64243537-cd5b-4250-afe3-0c2e3dc690ae; scratchpad/sector_observatory.html). Verdict: sector timing NOT forecastable; value=regime→leadership social content. DISTRIBUTION (blog/PNG/living page) un-picked.

## ⏭️ Other open: regime half of _mas (ONE REGIME SOT — reporting unification vs scanner.py:675 needs re-run). Queued: DST EventBridge Scheduler (pre-Nov); scrub DWAP perf_numbers.js; retire get_universe(); X-reply ticker fix; perf-numbers SSOT audit. Worker invoke: rigacap-prod-worker --profile rigacap, sync ok w/ --cli-read-timeout 600+Bash 600000 OR async Event+poll S3, TRUTHY payload. Local venv Py3.9 str|None=RED HERRING (prod 3.12).
