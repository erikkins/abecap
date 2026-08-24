---
name: project_prediction_ledger_idea
description: "FUTURE growth idea — dated probabilistic market/regime predictions, SCORED into a track record, publish once proven"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Prediction Ledger + Scored Regime Track Record (Erik idea, Aug 24 2026)

Erik: "would be REALLY cool to predict market movement... say 'next week we predict X', see if it comes true; once solid, PUBLISH — if we're right about a big move, socials + subs go off the rails." Filed as a real idea (not just musing).

## The insight
The seed already exists: **`regime_forecast_snapshots`** is written every scan (7-regime probabilities + transition odds — the "23% chance regime holds" line). What's MISSING is **scoring** — we've never graded whether the forecast regime actually happened. That scoring layer is the whole game.

## The honest/credible design
1. **Log a dated, falsifiable, PROBABILISTIC call each scan** (regime + directional lean + confidence, timestamped). Mostly already logged.
2. **Score it when the outcome is known** — weekly job marks each past call right/wrong → running hit-rate.
3. **Internal first → publish when it proves out.** The shareable asset is the SCORED record ("regime read correct X% over N weeks + what it told subs to do"), NOT a one-off lucky call. Verifiable track record > flex.

## HARD caution
- **Probabilistic only, NEVER point-predict** ("SPY +3% next week" = crystal-ball → one miss torches credibility + wanders into investment-advice/compliance). Frame: "60% odds rotating-bull holds; it did, 6 weeks running."
- On-brand: predictions reinforce posture-on-a-rule (regime→action), not prophecy. The prediction is the hook; the **graded accuracy is the product.**

## Build (mostly plumbing on existing data)
Prediction ledger table + scoring/accuracy job on top of `regime_forecast_snapshots` → internal accuracy dashboard → public "our calls, graded" page once hit-rate earns it. Spec properly when Erik greenlights.