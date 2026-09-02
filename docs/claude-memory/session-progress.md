---
name: session-progress
description: "Rolling snapshot of the current working session — accomplishments, in-flight work, key context for a fresh session"
metadata: 
  node_type: memory
  type: project
  originSessionId: b87c584c-343d-4a11-aca7-a450196570be
---

# Session progress — updated 2026-09-02 (~afternoon ET)

## ✅ LIVE IN PROD
- Daily digest redesign v3 (both tiers) — deployed + broadcast to all subs; cron re-enabled. Old digest fn = 1-line revert.
- Mirror go-live — first+default tab (paid→mirror/free→signals), two labeled Preserver/Maximizer sections, admin gate removed.
- **Mirror tour in the real portal** (origin/main 33eb01a): auto-opens on first Mirror-tab visit (localStorage 'rigacap_mirror_tour_seen', 400ms delay), MirrorView onState→mirrorLivePct feeds final step, "Take the tour" relaunch btn.
- **Tour copy fix** (origin/main bd45dc8, deploying): "Your move" step reworded — removed "Nothing to buy yet" (buy-push) → "…that's all the mirror needs. It simply shows how your holdings line up with the book; what you do with that is always yours to decide."

## ⏳ OPEN DECISION — tour firing scope (Erik asked; awaiting A/B/C)
- Current = **A**: localStorage flag = per-BROWSER not per-account. Fires for anyone who hasn't seen it → incl. ALL existing subscribers on next login (Mirror is default). Re-fires on new device; wrong on shared browser (prior user's flag blocks new user).
- **B** = per-account server flag (add user.mirror_tour_seen DB field + API set-on-close; fires once/user across devices).
- **C** = B + only auto-fire for NEW signups (existing users use the "Take the tour" button; don't surprise base).

## TOUR STRINGS (Erik reviewing/iterating)
- "Your starting point" (eclipse step) has 2 variants by pct>0: DEFAULT title "Further along than you think" / EMPTY-variant title "A blank sky, waiting" (App.jsx ~1198). Erik approved these.
- MIRROR_TOUR array at App.jsx:1189; MirrorTour component ~1208; empty switch ~1214 `if (cur.emptyVariant && !(pct>0))`.

## ⏭️ THIS MORNING'S ASK QUEUE (Erik) — sequence, awaiting steer
1. Mirror tour — shipped; ITERATING live (his active focus; more copy tweaks likely).
2. **Landing "The Mirror" section** (LandingPageV2.jsx ACTIVE at /). No Mirror/eclipse mention → new signups land on eclipse cold. Add intro (honest "facts not instructions"). IMPL: AlignmentEclipse (App.jsx:1069) NOT exported + circular (LandingPageV2 imported by App) → EXTRACT to components/AlignmentEclipse.jsx (rec) OR static PNG. Structure: page body composes <XSection/> ~725; SectionLabel :17.
3. **Drips** — productionize redesigned 6-step onboarding (email_service.send_onboarding_email D1/D3/D7/D12/D15/D22) on v3 editorial system (eclipse heroes per step, inline cid). Samples: scratchpad build_drip.py/build_drip2.py/build_welcome.py. Pattern = backend/app/services/digest_v3.py.
4. Password reset + win-back surfaces.

## KEY FACTS / RULES
- Email tests → erik@rigacap.com (NOT ekins@cookma.com=this window's login). AWS_PROFILE=rigacap. SMTP from worker Lambda env.
- Single source = today's dashboard; email GENERATES NOTHING; NEVER truncate lists; each tier own read.
- Brand claret/paper (#F5F1E8/#141210/#7A2430); NEVER navy/gold/olive; no DWAP/tape/PITFWU to customers.
- Mirror = book:sun/you:moon; more align=moon covers sun; total eclipse+corona=fully running (live/data-driven, honesty). Orb=digests(regime).
- Auto-snapshot hook sometimes commits staged files under its own msg. Deploys ~4min via push to main (CI Deploy RigaCap).
