---
name: session-progress
description: "Rolling snapshot of the current working session — accomplishments, in-flight work, key context for a fresh session"
metadata: 
  node_type: memory
  type: project
  originSessionId: b87c584c-343d-4a11-aca7-a450196570be
---

# Session progress — updated 2026-09-03

## ✅ JUST FIXED — Morning Health email "DWAP valid 590/591" (Erik's daily annoyance)
- Root cause: **^VIX** — index carries ZERO volume; DWAP = volume-weighted avg → sum(price×0)/sum(0) = NaN → always the "1 invalid" of 591 (it has 1770 bars ≥200 so it was counted). MA50/MA200 are price-only so they showed 100%. Confirmed via `{"parquet_query":{"sql":...}}` worker event (found ^VIX, dwap=NaN, volume=0).
- Fix (pushed 51086e1, deploying): exclude `^`-prefixed index symbols from indicator-validity counts in BOTH main.py places — the Morning Health email builder (~10034) AND the daily-scan canary (~1490, so <90% alert isn't skewed). Also NAME any remaining offender in the email row + scan log ("DWAP valid X/Y — invalid: SYM") so it's never an unactionable bare count again. Tomorrow's email → DWAP valid 590/590 (100%).
- Aside (not fixed, cosmetic): ^VIX bars in the parquet store are stale (last 2026-06-15); live VIX for regime comes fresh from yfinance at scan time so signals unaffected. Offered Erik to refresh parquet ^VIX separately.
- Useful tool discovered: worker event `{"parquet_query":{"sql":"... FROM prices ..."}}` runs arbitrary DuckDB on the parquet, returns up to 200 rows. Great for data diagnosis. Also `{"parquet_diagnose":true}`.

## ✅ EARLIER TODAY (all live in prod)
- Mirror tour → real portal (auto-opens first Mirror visit, "Take the tour" btn). Copy fix: "Your move" step no longer says "Nothing to buy yet" (removed buy-push) → comparison framing (pushed bd45dc8).
- Mirror go-live (first+default tab paid→mirror/free→signals, two Preserver/Maximizer sections). Daily digest v3 (both tiers) live + broadcast; cron on.

## ⏳ OPEN — Mirror tour firing scope (Erik asked, undecided)
- Current = A: localStorage flag = per-BROWSER; fires for anyone who hasn't seen it (incl. all existing subs next login; re-fires new device; wrong on shared browser). B = per-account server flag. C = B + new-signups-only. Awaiting Erik.

## ⏭️ QUEUED (this morning's asks, not yet done)
1. **Landing "The Mirror" section** (LandingPageV2.jsx ACTIVE at /) — prime new signups for the eclipse (now default tab). Impl: extract AlignmentEclipse (App.jsx:1069) → components/AlignmentEclipse.jsx (it's not exported + circular) OR static PNG. Honest "facts not instructions" framing.
2. **Drips** — productionize redesigned 6-step onboarding into email_service.send_onboarding_email (D1/D3/D7/D12/D15/D22) on v3 editorial system; samples in scratchpad build_drip.py/build_drip2.py/build_welcome.py; pattern = backend/app/services/digest_v3.py (inline_images cid heroes).
3. Password reset + win-back surfaces.

## KEY FACTS / RULES
- Email tests → erik@rigacap.com (NOT ekins@cookma.com = this window's Claude login). AWS_PROFILE=rigacap. Deploys ~4min via push to main (CI Deploy RigaCap). Gmail MCP token EXPIRED (can't read inbox).
- Single source of truth = today's dashboard; email GENERATES NOTHING; NEVER truncate lists; each tier own read.
- Brand claret/paper (#F5F1E8/#141210/#7A2430); NEVER navy/gold/olive; no DWAP/tape/PITFWU to CUSTOMERS (internal/admin like Morning Health is fine).
- Worker admin/data ops: run_migration {"sql":[...]} (Postgres), parquet_query {"sql":...} (DuckDB parquet), maximizer_preview, all AWS_PROFILE=rigacap.
