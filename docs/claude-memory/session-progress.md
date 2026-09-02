---
name: session-progress
description: "Rolling snapshot of the current working session — accomplishments, in-flight work, key context for a fresh session"
metadata: 
  node_type: memory
  type: project
  originSessionId: b87c584c-343d-4a11-aca7-a450196570be
---

# Session progress — updated 2026-09-02 (~noon ET)

## ✅ LIVE IN PROD (shipped)
- **Daily digest redesign v3** (both tiers, dashboard-sourced, per-regime+tier heroes) — deployed + broadcast to all subs last night; cron `rigacap-prod-daily-emails` re-enabled. Old digest fn intact = 1-line revert (email_service.send_daily_summary).
- **Mirror go-live** — first+default tab (paid→mirror/free→signals), two labeled Preserver/Maximizer sections, admin gate removed. In origin/main.
- **Mirror tour → real portal** — JUST pushed (origin/main 33eb01a, deploying). Wired MirrorTour into the in-app Mirror tab (was admin /app/next only): auto-opens on first Mirror-tab visit (localStorage 'rigacap_mirror_tour_seen', 400ms delay), MirrorView onState→mirrorLivePct feeds tour final step, + "Take the tour" relaunch btn. Build passed. Erik: "add it to the real mirror tab and we can continue iterating" — he'll iterate on the tour live.

## ⏭️ THIS MORNING'S ASKS (Erik) — sequence proposed, awaiting his steer
1. **Mirror tour** — shipped, iterate live (his active focus).
2. **Landing "The Mirror" section** (LandingPageV2.jsx = ACTIVE, routed at /). Currently NO mention of Mirror/eclipse → new paid signups land on the eclipse (default tab) COLD. Add an intro section (honest "facts, not instructions" framing). IMPLEMENTATION NOTE: AlignmentEclipse (App.jsx:1069) + EclipseGlyph (1050) are NOT exported and LandingPageV2 is imported BY App.jsx (circular) → to use the real eclipse, EXTRACT AlignmentEclipse → components/AlignmentEclipse.jsx (import in both). Alt = static eclipse PNG. My rec = extract (it's the signature visual). ASKED Erik: start landing now vs hold; sequence ok vs pull drips forward.
3. **Drips** — productionize redesigned 6-step onboarding drip into email_service.send_onboarding_email (steps D1/D3/D7/D12/D15/D22) on the v3/editorial system (eclipse heroes per step, inline cid). Samples built in scratchpad: build_drip.py (D1/D7/D12), build_drip2.py (settings/wound/door D3/D15/D22), build_welcome.py, build_regime.py. Digest v3 (backend/app/services/digest_v3.py) is the pattern to follow (inline_images cid heroes + editorial HTML).
4. **Other surfaces** — password reset, win-back/churn.

## LANDING V2 STRUCTURE (for the Mirror section)
- LandingPageV2.jsx: Hero → ValuePropSection(190 "What You're Paying For") → EdgeSection(223) → PerformanceSection → FounderSection → HowItWorksSection(~415) → Pricing → FAQ. Page body composes <XSection/> at ~725. SectionLabel component at :17. Brand: font-display (Fraunces), claret accents, paper-card sections, max-w-[800]/[1120].

## KEY FACTS / RULES
- Send email tests to erik@rigacap.com (NOT ekins@cookma.com=this window's Claude login). AWS_PROFILE=rigacap. SMTP from worker Lambda env.
- Single source of truth = today's dashboard; email GENERATES NOTHING; NEVER truncate lists; each tier own read.
- Brand claret/paper (#F5F1E8/#141210/#7A2430); NEVER navy/gold/olive; no DWAP/tape/PITFWU to customers.
- Mirror concept: book=sun, you=moon; more alignment = moon covers sun; total eclipse+corona = fully running strategy (the reward, live/data-driven for honesty). Orb (not eclipse) = digests (regime color).
- Auto-snapshot hook sometimes commits staged files under its own msg. Forks drifted/rate-limited on the digest yesterday.
