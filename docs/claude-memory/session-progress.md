---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 10–14 2026

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC (signal-intel dossier=INTERNAL, may name internals). PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Deploy=push main→GHA+smoke-test. Migration-first DB cols. STORAGE (live): PARQUET primary (PITFWU per-symbol RAW bars+corp-actions, split-adjust AT READ; all_data.parquet fallback; pickle legacy). TIER_SERVING flag gates tier-aware serving.

## ▶ ACTIVE BUILD — ADDITIVE MAXIMIZER (Erik approved; layout = SIDE-BY-SIDE desktop, stacks mobile)
- GOAL: Maximizer subscriber sees BOTH books delineated (Preserver base signals + breakout book), not the old regime SWAP. Matches price ($129 base + $100 add-on).
- **BACKEND DONE + compiles:** tier_serving.apply_tier_serving maximizer branch now returns buy_signals=Preserver base (always) + new `breakout_book` array (always; held day-X/29 + fresh; aging/empty outside rotating_bull), signal_source="both". signals.py caller captures breakout_book (annotates in_user_position) + adds to dashboard response dict. Perf/simulated cards untouched (already Maximizer blend). Reversible via TIER_SERVING flag.
- **FRONTEND + EMAIL: delegated to background FORK (agentId ae16973c0abf55c8d)** — App.jsx signals panel: when signal_source==='both' render 2-col responsive grid (Preserver left / Maximizer breakout book right, day X/29 badges), don't touch non-'both' path; reuse SignalCard. email_service.py digest: add delineated breakout block for maximizer. Must pass npm build + py_compile. NOT commit/deploy. Awaiting completion notification.
- **NEXT when fork done:** review diff → npm build → deploy behind flag → PREVIEW link for Erik before real subscribers. Then reconcile signal-intel Section 09 "receives both books" line (now TRUE once shipped).

## ▶ ADS — running, watching (broad TA/tool negatives added; check last-1-2-day search terms for junk stop; funnel filling; judge cost-per-signup).

## ▶ DOCS REFRESH — NOT COMMITTED (Erik pen-marking signal-intel; 6 html+pdf modified; scratch scripts/ untracked — do NOT commit). Aug 14 fixes: storage flip, split-adjust-at-read, NEW Section 03 PITFWU (renumbered to 16), Production-params both-tiers (entry-ensemble + position-mgmt stat rows), Venn "5d+60d" nudged. PDF re-export PENDING (batch when Erik done reading). OPEN: sweep investor/marketing/sales; 3 Qs (pricing figures, BacktesterService name, investor logo). Then re-export + commit design/documents only.

## ▶ QUEUED: Maximize recovery landing page; "A holding week" label; testimonials/churn; unified-sauteeing-whale.md.
