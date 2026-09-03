---
name: session-progress
description: "Rolling snapshot of the current working session — accomplishments, in-flight work, key context for a fresh session"
metadata: 
  node_type: memory
  type: project
  originSessionId: b87c584c-343d-4a11-aca7-a450196570be
---

# Session progress — updated 2026-09-03

## ✅ FIXED + DEPLOYED — Morning Health "DWAP valid 590/591" annoyance
- Root cause = **^VIX** (index, zero volume → volume-weighted DWAP always NaN; 1770 bars so counted as the 1 invalid of 591). Fix (pushed 51086e1, deployed): exclude `^`-prefixed indices from indicator-validity counts in BOTH main.py spots (Morning Health email builder ~10034 AND daily-scan canary ~1490); also NAME any remaining offender in the email row + scan log. Tomorrow → DWAP valid 590/590 (100%).

## 🔎 VIX STALENESS — investigated, NOT a live bug (Erik asked)
- `all_data.parquet` monolith `^VIX` frozen at 2026-06-15 (close 16.2) — froze at the mid-June PITFWU migration (indices read from all_data as base). Confirmed via `{"parquet_query":{"sql":...}}`.
- LIVE regime/dashboard VIX is FRESH: read from scanner_service.data_cache['^VIX'] (signals.py:854/1164/1172); ^VIX is a REQUIRED_SYMBOL re-fetched from yfinance every scan (market_data_provider forces indices→yfinance). Proof: Sep-1 dashboard vix_level=16.34 ≠ June parquet 16.2 → fetch is fresh; parquet just never gets index bars written back.
- Net: regime/signals/digests/Morning-Health VIX all fine. ONLY risk = research/walk-forward that reads ^VIX DIRECTLY from the parquet (no ^VIX after 2026-06-15). **ASKED Erik**: fix parquet persistence (write fresh ^VIX/index bars each scan) or leave it. Awaiting.

## ✅ EARLIER (live in prod)
- Mirror tour → real portal (auto-open first visit, "Take the tour" btn); "Your move" copy de-buy-pushed (bd45dc8). Mirror go-live (first+default tab). Daily digest v3 both tiers live + broadcast; cron on.

## ⏳ OPEN DECISIONS / QUEUE
- **Tour firing scope** (A/B/C): current A = localStorage per-browser (fires for anyone unseen, incl all existing subs next login). B=per-account server flag. C=B+new-signups-only. Awaiting Erik.
- **VIX parquet persistence** fix — awaiting Erik (above).
- **Landing "The Mirror" section** (LandingPageV2.jsx ACTIVE at /) — prime new signups for eclipse default tab; extract AlignmentEclipse (App.jsx:1069, not exported + circular) → components/AlignmentEclipse.jsx OR static PNG; honest "facts not instructions".
- **Drips** — productionize redesigned 6-step onboarding into email_service.send_onboarding_email (D1/D3/D7/D12/D15/D22) on v3 editorial system; samples scratchpad build_drip.py/build_drip2.py/build_welcome.py; pattern = backend/app/services/digest_v3.py.
- Password reset + win-back surfaces.

## KEY FACTS / RULES
- Email tests → erik@rigacap.com (NOT ekins@cookma.com = this window's Claude login). AWS_PROFILE=rigacap. Deploys ~4min via push to main. Gmail MCP token EXPIRED.
- Worker diag tools: {"parquet_query":{"sql":"...FROM prices..."}} (DuckDB, ≤200 rows), {"parquet_diagnose":true}, run_migration {"sql":[...]} (Postgres), maximizer_preview. All AWS_PROFILE=rigacap.
- Single source of truth = today's dashboard; email GENERATES NOTHING; NEVER truncate lists; each tier own read.
- Brand claret/paper (#F5F1E8/#141210/#7A2430); NEVER navy/gold/olive; no DWAP/tape/PITFWU to CUSTOMERS (admin/internal like Morning Health is fine).
- Data flow: live read = PITFWU per-symbol store; indices/missing/short-history fall back to all_data.parquet (frozen ~June). Regime reads data_cache (fresh via daily fetch).
