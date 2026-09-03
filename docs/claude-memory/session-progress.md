---
name: session-progress
description: "Rolling snapshot of the current working session — accomplishments, in-flight work, key context for a fresh session"
metadata: 
  node_type: memory
  type: project
  originSessionId: b87c584c-343d-4a11-aca7-a450196570be
---

# Session progress — updated 2026-09-03

## 🔧 IN FLIGHT — backfill + persist ^VIX/^GSPC to all_data.parquet (fork ac7fd1aa2f736edd0)
- Root cause (verified): PRICE_SOURCE=parquet mode (live ~Jun 17) → daily scan SKIPS export_parquet (main.py:1652-1654, OOM avoidance). Tradeable symbols fresh via PITFWU per-symbol store (pitfwu_append); indices NOT in PITFWU (Alpaca can't serve ^VIX/^GSPC → yfinance only), so they live ONLY in all_data.parquet which froze. ^VIX last bar 2026-06-15 (close 16.2, vol 0). LIVE regime VIX is FINE (re-fetched into data_cache each scan); staleness only hurts research/WF reading ^VIX from parquet.
- Fork task: (1) BACKFILL ^VIX+^GSPC 2026-06-15→today (yfinance) into all_data.parquet; (2) ongoing PERSISTENCE = small memory-safe TARGETED index-row merge into all_data each scan (NOT full 600-symbol rewrite = the OOM that killed export_parquet). SAFETY RAILS required: backup all_data.parquet before write; merge-PRESERVE all non-index symbols/rows; verify via parquet_query (symbol count unchanged, sample non-index NVDA/PATH row counts unchanged, ^VIX/^GSPC max date now current); wrap scan-path code so it can't break the scan; commit backend ONLY (never App.jsx); STOP+report if all_data structure risky. AWAITING fork report; I review before trusting.
- NOTE for indices: dwap=NaN/volume=0 is EXPECTED for indices — do NOT fabricate; preserve schema.

## ✅ DONE TODAY (live in prod)
- Morning Health "DWAP valid 590/591" squashed — was ^VIX (index, zero-vol→NaN dwap). Fix pushed 51086e1: exclude ^-indices from validity counts (email builder ~10034 + daily-scan canary ~1490) + name any remaining offender. → 590/590 (100%).
- Mirror tour → real portal (auto-open first visit, "Take the tour" btn); "Your move" copy de-buy-pushed (bd45dc8). Mirror go-live (first+default tab). Daily digest v3 both tiers live+broadcast; cron on.

## ⏳ OPEN DECISIONS / QUEUE
- **Tour firing scope** A/B/C: current A=localStorage per-browser (fires for anyone unseen incl all existing subs next login). B=per-account server flag. C=B+new-signups-only. Awaiting Erik.
- **Landing "The Mirror" section** (LandingPageV2.jsx ACTIVE at /) — prime new signups for eclipse default tab; extract AlignmentEclipse (App.jsx:1069, not exported+circular) → components/AlignmentEclipse.jsx OR static PNG; honest "facts not instructions".
- **Drips** — productionize redesigned 6-step onboarding into email_service.send_onboarding_email (D1/D3/D7/D12/D15/D22) on v3 editorial; samples scratchpad build_drip.py/build_drip2.py/build_welcome.py; pattern=backend/app/services/digest_v3.py.
- Password reset + win-back surfaces.

## KEY FACTS / TOOLS
- Email tests → erik@rigacap.com (NOT ekins@cookma.com=this window's Claude login). AWS_PROFILE=rigacap. Deploys ~4min via push to main. Gmail MCP token EXPIRED.
- Worker diag: {"parquet_query":{"sql":"...FROM prices..."}} (DuckDB all_data, ≤200 rows), {"parquet_diagnose":true}, run_migration {"sql":[...]} (Postgres), maximizer_preview, {"daily_emails":{"target_emails":[...],"force_tier":...}}.
- Data flow: live read = PITFWU per-symbol store; indices/missing/short-history fall back to all_data.parquet (was frozen ~Jun). Regime reads data_cache (fresh via daily yfinance fetch). PRICE_SOURCE=parquet.
- Single source=today's dashboard; email GENERATES NOTHING; NEVER truncate lists; each tier own read. Brand claret/paper; NEVER navy/gold/olive; no DWAP/tape/PITFWU to CUSTOMERS (admin/internal OK).
