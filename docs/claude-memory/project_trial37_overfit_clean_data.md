---
name: Trial 37 advertised numbers were on corrupted data; clean-data rerun in progress
description: The +240% / 0.89 Sharpe / 24% MDD "Trial 37" numbers were produced on a pickle with unadjusted NVDA/CMG/WMT/AVGO splits. Same params on clean data = +96% / 0.58 / 36.6% MDD.
type: project
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
## Discovery (Apr 15 2026)

Trial 37 params (the set cited throughout marketing + docs) were optimized against a pickle containing unadjusted splits for NVDA, CMG, WMT, AVGO (and 75 other split events surfaced by full-universe audit). The optimizer found entry timing that exploited the split-day price "jumps" — not real momentum.

**Same Trial 37 params on clean data (permanent Adjustment.SPLIT, 79-split refetch, data-quality filters):**
- ~+96% cumulative (vs advertised +240%)
- 0.58 Sharpe (vs 0.89)
- -36.6% MDD (vs 24%, WORSE)

**Why this matters:**
- Marketing docs still cite the old numbers. Need to update once clean-data optimization completes.
- Trial 37 params are effectively a local optimum for a world that no longer exists.

**How to apply:**
- Don't cite +240% / 0.89 / 24% MDD externally until re-validated
- TPE run3 is re-optimizing on clean pickle with anti-squeeze filters added (Lever 9: `max_recent_return_pct`, `price_velocity_cap_pct`). Running in background, ~12h to complete.
- After run3: pick new representative trial, validate across 8 start dates, update design docs + landing page numbers
- Pre-Trial-37 params snapshot preserved in `project_ensemble_pre_trial37_params.md` as rollback reference

**MDD concentration finding:** The -36.6% MDD on clean data traces to Feb 2021 meme-squeeze exits (EH -58%, TLRY -50%, LCID -39%). Anti-squeeze filters target this specifically.
