---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 2dce3134-d861-45c4-a371-80378750f8c0
---

# Session snapshot — Jul 31 2026

## Frozen spec
- Never expose publicly: t30v/Core/Ensemble/Option-B/N=15/DWAP/overlay-internals/sleeve-internals. Never "tape"/"NaN". PITFWU=survivorship-free. Deploy=push main→wait "Deploy RigaCap" complete before resending emails. BE CAUTIOUS w/ the tier books Erik built this week — CHECK before change.

## CANONICAL = OVERLAY, LOCKED (Erik). Saved scripts/overlay_canonical.json.
- **Shipped construction (verified in code):** Preserver = t30v + capitulation cash-raise (exposure 0.25); Maximizer = that base + REAL held breakout book in rotating_bull. The marketed 8.6/14.5 were SLEEVE-IDEALIZED (don't reproduce in single pool — "collapsed to ≈Core"). Overlay is what ships.
- **OVERLAY numbers (public-safe):** Preserver 21yr **7.7%/0.87/−13.7**, 5yr **13.0/1.28/−12.9**, rolling-12mo-avg **10.7%** (77% pos)/24mo 20.8%. Maximizer 21yr **13.5/0.93/−20.8**, 5yr **31.4/1.51/−14.9**, rolling-12mo **26.5%** (77% pos)/24mo 57.5%. S&P 2021-26 14.2/0.87/−25.4. Raw-mom 21yr 13.2/0.69/−57. Re-baseline from sleeve costs only ~1pp return, BETTER drawdown → honest + still excellent.
- Recent-24 single window: to-today read CONTAMINATED by settling drift (bogus +61) → LEAN toward publishing ROLLING avg only (drift-proof), drop single window.

## 3rd-PARTY DUE-DILIGENCE (Erik ran it, twice): research 9/10, Preserver 8/10, Maximizer 6.5-7/10. Transparency is the MOAT. Answered 5 methodology Qs: fixed rules rolled fwd (no per-period re-opt); ~10-12 params; survivorship-free; point-in-time since 2016 (pre-2016 disclosed caveat); breakout sleeve is NEWEST/hindsight-exposed (Maximizer's extra edge less certain → Preserver=robust core, Maximizer=higher-upside satellite). Allocator: haircut planning assumptions (Pres 11-13%, Max 13-17%) + track core/sleeve/combined live separately.

## ▶ IN FLIGHT: PAGE WALKTHROUGH (read-only punch-list done, no edits yet). Positioning = BOLD because research is honest+strong.
- Punch-list covers: LandingPageV2 (hero/knob + comp table + FAQ + "How We Test"), TrackRecordPageV2 (rolling lead + 2016+ high-confidence framing), MethodologyPageV2 (How-We-Test showcase), ForAdvisersPage, 3 blogs, LEGACY TrackRecordPage.jsx (retire — publishes retired 8.3), SocialTab cards (fix 8.3), backend text (email/newsletter/ai_content).
- **3 DECISIONS I recommended, awaiting Erik's OK to bake in:** (1) recent number = ROLLING avg (drop drift-prone single window); (2) HAIRCUT/planning-assumptions = IN but placed in DEPTH (methodology/for-advisers/FAQ), NOT hero — trust-builder for $250k+ target, filters impulse buyer; (3) /for-advisers should FEATURE Maximizer as a growth SATELLITE alongside Preserver core sleeve (currently 2 Maximizer vs 5 Preserver mentions, table-only).

## NEXT: bake the 3 decisions into punch-list → verify exact LandingPageV2 hero/knob copy → Gate B edits: refresh registry §1 → WIRED SSOT (backend perf_numbers.py + frontend perf_numbers.js) → re-point surface_map → update ~12 surfaces + fix retired-8.3 leaks + add How-We-Test/confidence-tier framing → rebuild + regen cards/PDFs. THEN GOOGLE ADS two-tier (Preserver=capital-preservation $250k+ / Maximizer=aggressive-growth satellite; reuse stability-search-test-a; fix GA4→Ads conversion import). ALSO roadmap: live attribution tracker (core/sleeve/combined). Plan: /Users/erikkins/.claude/plans/unified-sauteeing-whale.md.
## CLEANUP temp artifacts: scripts/{tier_vintages_today.py, tier_vintages_daily_today.py, recompute_canonical.py, canonical_recompute.json, tier_curves_21y_today.json, tier_curves_21y.json.bak-may29, overlay_canonical.json[keep]}.
