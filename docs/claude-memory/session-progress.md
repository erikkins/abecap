---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — thru Aug 21 2026

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC; never "tape" ([[feedback_no_tape_brand_voice]]). Web deploy=push main→"Deploy RigaCap" GHA + smoke api.rigacap.com/api/market-data-status→200. Admin app=mobile-admin/ EAS OTA. NO Google Ads API. RigaCap=PUBLISHER (no custody). Research MUST map to prod penny-to-penny; verify before concluding; no fabrication.

## ✅ Maximizer vol-brake cold-start — RESOLVED ([[project_maximizer_coldstart_jul21]])
- Found (verified) the live book's vol-brake sat at 1.0 its whole life (bk_eq_hist<21, never warm-started; launched Jul8, rebased Jul24; now cold −1.69%/$98,311). NOT a WF/marketing flaw — verified the WF engine (maximizer_portfolio.vol_scaled_returns / maximizer_sleeves.vol_scale) brakes correctly; its ~21d cold-start is negligible over 5-21yr; marketed numbers CORRECT + untouched.
- FIX SHIPPED (bb3c9be, live, verified diff): _vol_scale COLD-WINDOW ESTIMATOR — WARM (≥21) = certified return-stream path BYTE-IDENTICAL (marketed penny-to-penny intact); COLD (<21) = real held-names' value-weighted realized vol from data_cache (lagged, cap1.0); empty→1.0. Forward-looking, no live mutation. Brake active off real holdings vol from tonight's scan; certified own-history path takes over MONDAY Aug24 (bk_eq_hist hits 21).
- DISPLAY note: portal "VOL-TARGET EXPOSURE" reads tier_serving._vol_scale_from_hist (cold-gated, shows 100% while <21) — SELF-HEALS Mon when bk_eq_hist≥21 (matches live warm brake). Optional display fix (persist _last_vol_scale to show cold-window brake for FUTURE launches) = cosmetic, 0 subs, SKIPPED for now.

## ✅ Start-date SWEEP — DONE + VERIFIED (handler c622e3e, read-only, live; I re-ran → identical)
- Warm-braked certified-engine sweep, 14 launch dates Jun23–Jul30→Aug20: −7.22% to +0.19%, median −1.44, mean −2.54, std 2.41 → $92,780–$100,190 on $100k (~$7.4k launch-day swing). Jul-8 real launch = −0.71% (79th pctile, a good date). Anchor: trailing-365 Maximizer blend +53.8% (SPY +19.5%), breakout +49.2%. Headline: every date flat-to-down = "launched into a soft 6-8wk breakout patch," but breakout +49%/trailing-yr → timing artifact not strategy. Onboarding: TRANCHE entry → locks mean, cuts unlucky-launch tail (dampens, doesn't erase; can't manufacture gains in a down window).

## ✅ Other shipped: SPY-trend-aware market reads (+mandatory 3+ streak); 2 ad doors /should-i-sell + /momentum + ExploreMore + dual funnel tracking; funnel checkout_redirect/signup sub-steps; personal social launch posts (design/documents/personal-launch-social-posts.txt).

## ▶ NEWSLETTER improvements — DISCUSSED Aug 22, no code yet (Erik reviewing draft)
- 3 issues Erik flagged: (1) "Second week running in Rotating Bull" is WRONG — verified rotating_bull unbroken 81 trading days / ~16wk since Apr 8 (fabrication: generator not fed real regime-run length → AI guesses). (2) "Gold +2%/treasuries slipped/small caps +0.8%" is a DAILY EOD snapshot presented as the WEEK — should be a true weekly recap. (3) "What the system is not doing" is Preserver-only → RULE: ≥1 of 3 points must be Maximizer/breakout tier. "Note from Erik" = fine, leave.
- MY READ: #1 & #2 share ROOT CAUSE = newsletter_generator_service reuses the DAILY market_context instead of a weekly-computed pack. #3 = one-line tier rule.
- ✅ ERIK: "add those changes + create another draft." BUILDING (fork a8adb29461d00e883, bg): implement (1) real regime-run length fed + ban invented durations, (2) WEEKLY data pack (week SPY/gold/bonds/smallcap moves from real data_cache closes + regime path + signal churn + notable days; recap the WEEK not Friday), (3) "not doing" ≥1 Maximizer point. Leave "Note from Erik". Then DRAFT-ONLY regenerate + email to erik@rigacap.com (NEVER subscribers, publish/lock untouched). Guardrails: numbers REAL/spot-checked, no fabrication, brand rules, py_compile, deploy+smoke. Will report diff + draft + confirm 3 fixes. Erik keeps first draft to cherry-pick.

## 📊 ADS analysis (Aug 8–22, ~$356 / 144 clicks / 0 conv) — Aug 22
- CPC HALVED ($3-4→$1.53-1.84 recent) + impressions rising = campaign getting efficient (good trajectory).
- ⭐ KEY: audience strongly OLDER + MALE (Male 55+≈45%, all 55+≈65%, male≈74%) = PERFECT Preserver/capital-preservation fit (retirement/401k protectors). Lean messaging into protect-retirement/nearing-retirement. Preserver = the main horse.
- One keyword = the engine: "is the market going to crash" $196/55% of spend/86 clicks/4.47% CTR. Crash-anxiety cluster carries it.
- Maximize starved+low-quality ($45/15/2.46%; momentum search terms = ETF/screener/list seekers, not subscription intent) → don't over-invest in momentum door yet.
- 83% mobile spend + older users → /should-i-sell MOBILE UX is THE conversion lever.
- Wasted competitor clicks to negative: marketbeat, warrior trading scanner, trendspider app, wall street zen, bullishalgoai, chinesesignalsbotpro, market chameleon, arbitrage radar, protraderanalysistool.
- 0 conv = FUNNEL not ads (ads deliver right demo cheap). Offered: paste-able negatives list / dayparting dig.

## ▶ IN FLIGHT: newsletter fork a8adb29461d00e883 (weekly-pack + regime-run + tier rule → draft to erik@rigacap.com) — awaiting completion.

## ▶ OPEN / WATCH
- Funnel: wait a few days → /momentum vs /should-i-sell; offer→CTA leak; signup modal (single-field + one-tap OAuth) ([[project_sis_funnel_watch]]). Ads Erik-side.
- DOCS refresh: design/documents/* UNCOMMITTED (signal-intel/tech-arch/investor/etc); PDF re-export + sweep pending. Commit ONLY design/documents.