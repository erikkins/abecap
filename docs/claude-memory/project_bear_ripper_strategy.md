---
name: Bear Ripper — targeted bear-market MDD reduction strategy
description: After clean-data strategy is locked, develop a specialized bear-regime sub-strategy that finds 1-2 high-conviction trades during bear markets to pull up overall results and reduce MDD.
type: project
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
## Bear Ripper

**When to start:** After TPE run5 completes and the adaptive strategy is validated on clean data. The 30% MDD is the weak spot in an otherwise strong result (+297%, 1.10 Sharpe). Bear Ripper targets that gap specifically.

**Why:** The adaptive strategy's worst periods cluster in bear/panic regimes (Feb 2021 -16.6%, Mar 2021 -12.5%, Feb 2025 -9.5%). Currently the regime filter goes to cash during these periods — safe but leaves MDD on the table from the drawdown that triggers the cash exit. A targeted sub-strategy could find 1-2 high-conviction trades DURING those periods that offset the drawdown.

**Concept:**
- Activate only during weak_bear, panic_crash, or recovery regimes
- Very specific, tight rules — not trying to trade the full bear, just catch 1-2 opportunities
- Candidates: oversold bounces in mega-caps, sector rotation plays (defensive → cyclical), inverse momentum (stocks that held up during the selloff = relative strength leaders)
- Max 1-2 positions, smaller size (maybe 10% each vs normal 15%), tighter trailing stop (8% vs 12%)
- Goal: not to be profitable in isolation, but to reduce overall portfolio MDD by 5-10 percentage points

**Potential approaches:**
1. **Mega-cap defensive overlay** — during regime exit, hold 1-2 mega-caps with strong balance sheets that historically decline less (AAPL, MSFT, JNJ class). Limits MDD without going fully flat.
2. **RS Leaders during bear** — stocks showing relative strength (declining less than SPY) during bear regimes are the first to rip on recovery. Enter RS leaders when panic subsides.
3. **VIX-timed bounce trades** — when VIX spikes above 35-40 AND starts declining, enter 1-2 high-quality names for the relief bounce. Very short hold (1-2 weeks).
4. **Inverse-vol position** — small allocation to inverse-volatility or put-write strategy during extreme fear. Advanced, may not fit the signal-service model.

**How to apply:**
- Build as a separate strategy_type in the backtester (not modifying the ensemble)
- Test in WF simulation as a "bear sleeve" that activates alongside the main strategy during bear regimes
- Measure impact on overall MDD without sacrificing bull-market returns
- If successful, integrate into the ensemble as a regime-conditional overlay

**Success criteria:** reduce MDD from 30% to 20-22% without dropping total return below +250%.
