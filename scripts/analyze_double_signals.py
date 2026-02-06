#!/usr/bin/env python3
"""
Analyze Double Signal Performance

Compares returns of:
1. DWAP-only signals (DWAP trigger but not in momentum top 20)
2. Momentum-only signals (top 20 momentum but no DWAP trigger)
3. Double signals (BOTH DWAP trigger AND top 20 momentum)

Run from project root:
    python scripts/analyze_double_signals.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional

# Import from backend
from app.services.scanner import scanner_service
from app.core.config import settings


@dataclass
class SignalResult:
    """Track a signal and its outcome"""
    date: str
    symbol: str
    signal_type: str  # 'dwap_only', 'momentum_only', 'double'
    entry_price: float
    momentum_rank: Optional[int]
    momentum_score: Optional[float]
    pct_above_dwap: Optional[float]
    # Outcomes (filled in after holding period)
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    return_20d: Optional[float] = None
    max_gain_20d: Optional[float] = None
    max_loss_20d: Optional[float] = None


def load_data():
    """Load price data from S3 cache"""
    print("Loading price data from S3...")
    import asyncio

    async def load():
        await scanner_service.load_full_universe(max_cache_age_hours=168)

    asyncio.run(load())
    print(f"Loaded {len(scanner_service.data_cache)} symbols")


def get_momentum_rankings(date: pd.Timestamp, top_n: int = 20) -> Dict[str, dict]:
    """
    Calculate momentum rankings for a specific date.
    Returns dict of symbol -> {rank, score} for top N stocks.
    """
    candidates = []

    for symbol, df in scanner_service.data_cache.items():
        # Filter to data up to this date
        df_to_date = df[df.index <= date]
        if len(df_to_date) < 60:
            continue

        try:
            row = df_to_date.iloc[-1]
            price = row['close']

            # Quality filters
            ma_20 = df_to_date['close'].tail(20).mean()
            ma_50 = df_to_date['close'].tail(50).mean()
            high_50d = df_to_date['close'].tail(50).max()
            volume = row['volume']

            if price < 20 or volume < 500000:
                continue
            if price < ma_20 or price < ma_50:
                continue

            dist_from_high = (high_50d - price) / high_50d * 100
            if dist_from_high > 5:  # More than 5% below 50d high
                continue

            # Calculate momentum
            price_10d_ago = df_to_date.iloc[-10]['close'] if len(df_to_date) >= 10 else price
            price_60d_ago = df_to_date.iloc[-60]['close'] if len(df_to_date) >= 60 else price

            short_mom = (price / price_10d_ago - 1) * 100
            long_mom = (price / price_60d_ago - 1) * 100

            # Volatility (20-day)
            returns = df_to_date['close'].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(252) * 100

            # Composite score
            score = short_mom * 0.5 + long_mom * 0.3 - volatility * 0.2

            candidates.append({
                'symbol': symbol,
                'score': score,
                'short_mom': short_mom,
                'long_mom': long_mom
            })
        except Exception:
            continue

    # Sort and rank
    candidates.sort(key=lambda x: x['score'], reverse=True)

    return {
        c['symbol']: {'rank': i + 1, 'score': c['score']}
        for i, c in enumerate(candidates[:top_n])
    }


def get_dwap_signals(date: pd.Timestamp, threshold_pct: float = 5.0) -> Dict[str, dict]:
    """
    Find stocks with DWAP signals on a specific date.
    Returns dict of symbol -> {pct_above_dwap, price}
    """
    signals = {}

    for symbol, df in scanner_service.data_cache.items():
        df_to_date = df[df.index <= date]
        if len(df_to_date) < 200:
            continue

        try:
            row = df_to_date.iloc[-1]
            # Check if this is actually the target date
            if pd.Timestamp(row.name).date() != date.date():
                continue

            price = row['close']
            volume = row['volume']
            dwap = row.get('dwap', np.nan)

            if pd.isna(dwap) or dwap <= 0:
                continue
            if price < 20 or volume < 500000:
                continue

            pct_above = (price / dwap - 1) * 100

            if pct_above >= threshold_pct:
                signals[symbol] = {
                    'pct_above_dwap': pct_above,
                    'price': price
                }
        except Exception:
            continue

    return signals


def calculate_returns(symbol: str, entry_date: pd.Timestamp, entry_price: float) -> dict:
    """Calculate forward returns for a signal"""
    df = scanner_service.data_cache.get(symbol)
    if df is None:
        return {}

    df_after = df[df.index > entry_date]
    if len(df_after) < 20:
        return {}

    results = {}

    # 5-day return
    if len(df_after) >= 5:
        price_5d = df_after.iloc[4]['close']
        results['return_5d'] = (price_5d / entry_price - 1) * 100

    # 10-day return
    if len(df_after) >= 10:
        price_10d = df_after.iloc[9]['close']
        results['return_10d'] = (price_10d / entry_price - 1) * 100

    # 20-day return
    if len(df_after) >= 20:
        price_20d = df_after.iloc[19]['close']
        results['return_20d'] = (price_20d / entry_price - 1) * 100

        # Max gain/loss in 20 days
        prices_20d = df_after.iloc[:20]['close']
        results['max_gain_20d'] = (prices_20d.max() / entry_price - 1) * 100
        results['max_loss_20d'] = (prices_20d.min() / entry_price - 1) * 100

    return results


def analyze_signals(lookback_days: int = 252, sample_every_n_days: int = 5):
    """
    Main analysis function.

    Args:
        lookback_days: How far back to analyze
        sample_every_n_days: Sample every N trading days (for speed)
    """
    print(f"\nAnalyzing {lookback_days} days of history (sampling every {sample_every_n_days} days)...")

    # Get a reference symbol for trading days
    spy_df = scanner_service.data_cache.get('SPY')
    if spy_df is None:
        print("ERROR: SPY not in cache")
        return

    # Get trading days to analyze
    end_date = spy_df.index[-21]  # Stop 21 days before end to allow return calculation
    start_date = end_date - timedelta(days=lookback_days * 1.5)  # Extra buffer for weekends

    trading_days = [d for d in spy_df.index if start_date <= d <= end_date]
    trading_days = trading_days[::sample_every_n_days]  # Sample

    print(f"Analyzing {len(trading_days)} trading days from {trading_days[0].date()} to {trading_days[-1].date()}")

    all_signals: List[SignalResult] = []

    for i, date in enumerate(trading_days):
        if i % 20 == 0:
            print(f"  Processing {date.date()}... ({i+1}/{len(trading_days)})")

        # Get rankings and signals for this date
        momentum_top20 = get_momentum_rankings(date, top_n=20)
        dwap_signals = get_dwap_signals(date)

        momentum_symbols = set(momentum_top20.keys())
        dwap_symbols = set(dwap_signals.keys())

        # Categorize
        double_symbols = momentum_symbols & dwap_symbols
        dwap_only_symbols = dwap_symbols - momentum_symbols
        momentum_only_symbols = momentum_symbols - dwap_symbols

        # Record signals
        for symbol in double_symbols:
            entry_price = dwap_signals[symbol]['price']
            signal = SignalResult(
                date=str(date.date()),
                symbol=symbol,
                signal_type='double',
                entry_price=entry_price,
                momentum_rank=momentum_top20[symbol]['rank'],
                momentum_score=momentum_top20[symbol]['score'],
                pct_above_dwap=dwap_signals[symbol]['pct_above_dwap']
            )
            returns = calculate_returns(symbol, date, entry_price)
            signal.return_5d = returns.get('return_5d')
            signal.return_10d = returns.get('return_10d')
            signal.return_20d = returns.get('return_20d')
            signal.max_gain_20d = returns.get('max_gain_20d')
            signal.max_loss_20d = returns.get('max_loss_20d')
            all_signals.append(signal)

        for symbol in dwap_only_symbols:
            entry_price = dwap_signals[symbol]['price']
            signal = SignalResult(
                date=str(date.date()),
                symbol=symbol,
                signal_type='dwap_only',
                entry_price=entry_price,
                momentum_rank=None,
                momentum_score=None,
                pct_above_dwap=dwap_signals[symbol]['pct_above_dwap']
            )
            returns = calculate_returns(symbol, date, entry_price)
            signal.return_5d = returns.get('return_5d')
            signal.return_10d = returns.get('return_10d')
            signal.return_20d = returns.get('return_20d')
            signal.max_gain_20d = returns.get('max_gain_20d')
            signal.max_loss_20d = returns.get('max_loss_20d')
            all_signals.append(signal)

        # For momentum-only, use close price as entry
        for symbol in momentum_only_symbols:
            df = scanner_service.data_cache.get(symbol)
            if df is None:
                continue
            df_to_date = df[df.index <= date]
            if len(df_to_date) == 0:
                continue
            entry_price = df_to_date.iloc[-1]['close']

            signal = SignalResult(
                date=str(date.date()),
                symbol=symbol,
                signal_type='momentum_only',
                entry_price=entry_price,
                momentum_rank=momentum_top20[symbol]['rank'],
                momentum_score=momentum_top20[symbol]['score'],
                pct_above_dwap=None
            )
            returns = calculate_returns(symbol, date, entry_price)
            signal.return_5d = returns.get('return_5d')
            signal.return_10d = returns.get('return_10d')
            signal.return_20d = returns.get('return_20d')
            signal.max_gain_20d = returns.get('max_gain_20d')
            signal.max_loss_20d = returns.get('max_loss_20d')
            all_signals.append(signal)

    return all_signals


def generate_report(signals: List[SignalResult]):
    """Generate analysis report"""

    # Group by signal type
    by_type = defaultdict(list)
    for s in signals:
        if s.return_20d is not None:  # Only include signals with complete return data
            by_type[s.signal_type].append(s)

    print("\n" + "="*80)
    print("DOUBLE SIGNAL ANALYSIS REPORT")
    print("="*80)

    print(f"\nAnalysis Period: {signals[0].date} to {signals[-1].date}")
    print(f"Total Signals Analyzed: {len(signals)}")

    # Summary by type
    print("\n" + "-"*80)
    print("SIGNAL COUNTS")
    print("-"*80)
    for signal_type in ['double', 'dwap_only', 'momentum_only']:
        count = len(by_type[signal_type])
        pct = count / len(signals) * 100 if signals else 0
        print(f"  {signal_type:20s}: {count:5d} signals ({pct:5.1f}%)")

    # Performance comparison
    print("\n" + "-"*80)
    print("AVERAGE RETURNS BY SIGNAL TYPE")
    print("-"*80)
    print(f"{'Signal Type':<20} {'Count':>8} {'5-Day':>10} {'10-Day':>10} {'20-Day':>10} {'Max Gain':>10} {'Max Loss':>10}")
    print("-"*80)

    for signal_type in ['double', 'dwap_only', 'momentum_only']:
        sigs = by_type[signal_type]
        if not sigs:
            continue

        avg_5d = np.mean([s.return_5d for s in sigs if s.return_5d is not None])
        avg_10d = np.mean([s.return_10d for s in sigs if s.return_10d is not None])
        avg_20d = np.mean([s.return_20d for s in sigs if s.return_20d is not None])
        avg_max_gain = np.mean([s.max_gain_20d for s in sigs if s.max_gain_20d is not None])
        avg_max_loss = np.mean([s.max_loss_20d for s in sigs if s.max_loss_20d is not None])

        print(f"{signal_type:<20} {len(sigs):>8} {avg_5d:>9.2f}% {avg_10d:>9.2f}% {avg_20d:>9.2f}% {avg_max_gain:>9.2f}% {avg_max_loss:>9.2f}%")

    # Win rate comparison
    print("\n" + "-"*80)
    print("WIN RATE BY SIGNAL TYPE (% of signals with positive return)")
    print("-"*80)
    print(f"{'Signal Type':<20} {'5-Day Win%':>12} {'10-Day Win%':>12} {'20-Day Win%':>12}")
    print("-"*80)

    for signal_type in ['double', 'dwap_only', 'momentum_only']:
        sigs = by_type[signal_type]
        if not sigs:
            continue

        win_5d = len([s for s in sigs if s.return_5d and s.return_5d > 0]) / len(sigs) * 100
        win_10d = len([s for s in sigs if s.return_10d and s.return_10d > 0]) / len(sigs) * 100
        win_20d = len([s for s in sigs if s.return_20d and s.return_20d > 0]) / len(sigs) * 100

        print(f"{signal_type:<20} {win_5d:>11.1f}% {win_10d:>11.1f}% {win_20d:>11.1f}%")

    # Risk-adjusted (simplified Sharpe proxy)
    print("\n" + "-"*80)
    print("RISK-ADJUSTED RETURNS (Return / StdDev)")
    print("-"*80)
    print(f"{'Signal Type':<20} {'20-Day Sharpe Proxy':>20}")
    print("-"*80)

    for signal_type in ['double', 'dwap_only', 'momentum_only']:
        sigs = by_type[signal_type]
        if not sigs:
            continue

        returns_20d = [s.return_20d for s in sigs if s.return_20d is not None]
        if returns_20d:
            avg = np.mean(returns_20d)
            std = np.std(returns_20d)
            sharpe_proxy = avg / std if std > 0 else 0
            print(f"{signal_type:<20} {sharpe_proxy:>19.2f}")

    # Double signal examples
    print("\n" + "-"*80)
    print("SAMPLE DOUBLE SIGNALS (showing top 15 by 20-day return)")
    print("-"*80)

    double_sigs = sorted(by_type['double'], key=lambda x: x.return_20d or 0, reverse=True)[:15]
    print(f"{'Date':<12} {'Symbol':<8} {'Rank':>6} {'DWAP%':>8} {'20d Ret':>10}")
    print("-"*80)
    for s in double_sigs:
        print(f"{s.date:<12} {s.symbol:<8} {s.momentum_rank:>6} {s.pct_above_dwap:>7.1f}% {s.return_20d:>9.1f}%")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    double_avg = np.mean([s.return_20d for s in by_type['double'] if s.return_20d]) if by_type['double'] else 0
    dwap_avg = np.mean([s.return_20d for s in by_type['dwap_only'] if s.return_20d]) if by_type['dwap_only'] else 0
    mom_avg = np.mean([s.return_20d for s in by_type['momentum_only'] if s.return_20d]) if by_type['momentum_only'] else 0

    print(f"\n20-Day Average Returns:")
    print(f"  Double Signals:    {double_avg:>6.2f}%")
    print(f"  DWAP-Only:         {dwap_avg:>6.2f}%")
    print(f"  Momentum-Only:     {mom_avg:>6.2f}%")

    if double_avg > dwap_avg and double_avg > mom_avg:
        improvement_vs_dwap = double_avg - dwap_avg
        improvement_vs_mom = double_avg - mom_avg
        print(f"\n✅ DOUBLE SIGNALS OUTPERFORM!")
        print(f"   +{improvement_vs_dwap:.2f}% vs DWAP-only")
        print(f"   +{improvement_vs_mom:.2f}% vs Momentum-only")
        print(f"\n   Recommendation: Consolidate to single 'Ensemble Signals' view")
    else:
        print(f"\n⚠️  Double signals do NOT clearly outperform.")
        print(f"   Consider keeping separate views or investigating further.")


def main():
    load_data()
    signals = analyze_signals(lookback_days=252, sample_every_n_days=5)

    if signals:
        generate_report(signals)
    else:
        print("No signals found!")


if __name__ == "__main__":
    main()
