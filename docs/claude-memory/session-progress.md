---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 24 2026 (FREE-FIRST PIVOT, live-built all day)

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC; never "tape"; NO fabrication. Web deploy=push main→"Deploy RigaCap" GHA (~4min) + smoke api.rigacap.com/api/market-data-status→200. Worker events: `{"db_read":"SQL"}` (read-only), `{"run_migration":true,"sql":...}` (writes/commits), invoke `rigacap-prod-worker` AWS_PROFILE=rigacap. NO Google Ads API. **NO PICKLE — live=PARQUET/PITFWU** ([[feedback_no_pickle_parquet]]). perf_numbers SSOT. Verify before concluding.

## 🧭 THE MODEL (Erik decided today) — no-card 30-day FULL trial → 14-day phase-out → proof floor → paid
- Signup (email/Google/Apple, NO card) → `status='trial'` trial_end=+30d → is_valid()=full product for FIRST 14 days (`TRIAL_FULL_DAYS`, database.py) → then PROOF FLOOR (FreeProofView, NOT lockout) → upgrade to `active` (+30-day money-back). Both tiers in trial via `Subscription.has_maximizer_access()` (trial-aware, non-persistent — no compmax leak). Admin=paid; admin `?preview_state=free` = QA path.
- **Layout branch = STABLE `freeTier`** (App.jsx ~1929: `isAdmin ? subscription_required===true : isAuthenticated && !hasValidSubscription`) — from /me, pre-fetch, so NO stale-cache flash. All paid chrome gated on `freeTier`. FreeProofView = proof floor (dynamic "where we are" phase readout + 21yr record w/ drawdowns + counts + closed WINNERS "We Called It" + ticker-free read + persistent "Unlock the full product" banner).
- FREE = PROOF ONLY, zero live actionable. Reads never leak tickers (choke point `security.resolve_entitlement`). [[project_free_first_spec]] = full spec.

## ✅ SHIPPED TODAY (all live on main): entitlement backbone (is_valid paid-only+trial-14d, resolve_entitlement, /dashboard free seam) · FreeProofView + flash-fix · register→trial (all 3 paths) · two-step no-card LoginModal · **fixed OAuth/checkout LOOP** (redirectToCheckoutIfNeeded no-ops; auto-checkout effect retired) · Stripe cancel/success→/app (was /pricing 404) · **HOTFIX register 422** (decorator was on wrong fn) · compmax leak fix · Founder→Introductory landing reframe (dropped seats FOMO) · Maximizer admin-alert %bug (dollars-as-%) · **Breakout Radar** (precompute in scan→cache; enriched table Trigger/%-to-go/6mo-mom/Vol; manually patched live via `{"patch_breakout_radar":true}` — backs up to signals/backups/ first) · Maximizer in Morning Health email (main.py:9138) · **Missed-Opps reframe** (hidden for mirror/paid — they see real "Recently closed"; free keeps "We Called It") · retired record-entry column+button (mirror model) · many alignment fixes (books/date-row/headers px, cookie-banner pb-28).
- 📋 FINDINGS: no-DWAP symbol=**^VIX** (0 volume→NaN, benign). **Universe NOT auto-refreshed** — no cron for `universe_refresh` (top-600 ossifies). First Maximizer closes: ERAS −9.7% / S +4.5% (29d).

## ✅ SHIPPED (2nd half): weekly `universe_refresh` cron (terraform, targeted+plan-first, Sat 20:00 UTC — closes universe-ossify gap; ~60 existing resources untouched) · ad-door CTAs → "Start free — no card" · **CTA DYNAMIC PRICING** (billing.py founding_status now returns `pricing` SSOT {monthly 129/annual 1099/intro_monthly 59-if-open/intro_active/intro_lock_months}; FreeProofView fetches founding-status → CTA leads "$59/mo introductory" + annual w/ 12mo-lock + 30-day money-back) · removed redundant "Momentum names..." subtext · tier-note aligned px-3. NO pickle code written (Erik emphatic ×2 — parquet only; decom PARKED, audit agent killed).
## ⚠️ ALIGNMENT: Erik chose **FLUSH px-0** for the both-books column, then refined: tier-note CAN stay inset; the KEY = "Preserver signals"+"Momentum..." subtext align w/ the table's SYMBOL col (all px-3 now — verified in code; if Erik still sees offset it's a STALE BUNDLE → hard-refresh / CloudFront invalidate). Full flush-px-0 mechanical pass still pending (books/date-row px-0, sections+table-first-col flush) — lower priority after the subtext removal.
## 🔀 NEW GAP (Erik surfaced, awaiting go): **two-book "Today's Moves" in the email.** Maximizer subs mirror BOTH books but `build_todays_actions` (tier_serving:284) queries tier='maximizer' ONLY → no Preserver Buy/Sell box. FIX = add Preserver moves sourced from TODAY's live ModelPosition entries/exits (t30v, different plumbing than TierFills), label per-book. RECOMMENDED — asked Erik.
## ⏳ QUEUED / OPEN (asked Erik, awaiting):
- Add weekly `universe_refresh` cron (close universe-recalc gap)?
- Idle-token-expiry guard (idle tab shows stale hybrid instead of re-auth) — offered.
- CTA dynamic pricing ($1,099 btn + banner → live intro price from /api/billing/founding-status — which returns {taken,limit,remaining,open}, NO prices yet).
- DRIP rewrite (#6): re-key `scheduler.send_onboarding_drip_emails` (days_since created_at) to 14-day window: D1-3 how-to-act, ~D12-13 "access winds down, subscribe", D15+ proof-floor conversion; nightly digest+alerts ALREADY auto-phase via is_valid; + fail-closed pre-send guard.
- Residual "founding" copy sweep (ShouldISell/Momentum/blogs). QA harness force_state for emails.
- Erik iterating via live walkthrough + screenshots (hard-refresh often = stale bundle).