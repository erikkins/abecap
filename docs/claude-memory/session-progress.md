---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 2dce3134-d861-45c4-a371-80378750f8c0
---

# Session snapshot — Jul 30 2026

## Frozen spec (load-bearing)
- Tiers: everyone served **Preserver** floor (never raw t30v/Core). **Maximizer** = paid `has_maxpp_addon` OR admin `compmax`; = Preserver EXCEPT rotating_bull → breakout book (N=15, vol-target). Flag-gated by `TIER_SERVING` (env set on BOTH api + worker Lambdas).
- Trades scoped to the strategy that opened them (`positions.source`: 'breakout'=hold-29 / 'preserver'=trailing-30%). Upgrade/downgrade never re-manages open trades.
- **NEVER expose publicly:** "t30v"/"Core"/"Option-B"/"N=15"/DWAP/the capitulation overlay. **Never "tape"** (enforce via voice_filters, not just prompts). **Never render "NaN".**
- Certified WF (tier_serving.CERTIFIED_WF): Maximizer +301.4/1.47/−15.5 ("Maximizer Ensemble"); Preserver +89.2/0.97/−20.2 ("Preserver Ensemble"); window "2021–2026 full cycle".
- Deploy = push main → wait for **"Deploy RigaCap"** workflow to COMPLETE before resending emails (`gh run list --workflow "Deploy RigaCap"`). Long ops → invoke `rigacap-prod-worker` directly. Admin email sends target erik@rigacap.com.

## Done this session
- **Admin audit** → hid deprecated tabs (Strategies/Lab/Auto-Pilot; multi-strategy leftovers; kept code + `?tab=` reachable). Other 9 tabs KEEP. Shipped 56d3ec0.
- **Framing**: Simulated Portfolio now shows full-cycle certified numbers beside rolling trailing-365 (tier_backtest.full_cycle + App.jsx line). Shipped 56d3ec0.
- **Per-tier capital = CLOSED, no code** — Erik: "one tier per user!" → one `user.portfolio_size` already persists it. Nothing to split.
- **Email fonts**: bumped line-item descriptors (BREAKOUT/HOLDING·DAY/Nd TO EXIT/score·label/pnl) 10-13px→14-16px + darkened muted gray. Shipped 79b726e (deployed). Fresh Maximizer sample re-sent to erik@rigacap.com — awaiting Erik's phone confirm.

## Next / queued (Erik's call which first)
- Next-wave Maximizer widgets: book equity curve vs SPY; day-29 exit alerts (email/push adherence); closed-trades STR; book stats.
- Real beta **tier-announcement** blast (send_tier_announcement, knob email-knob-v3.png) to beta list → let them pick tier. Then Google Ads → buyers.
- Subscriber-facing tier alerts (day-29 exit etc.).
