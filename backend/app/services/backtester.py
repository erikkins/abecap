"""
Backtester Service - Simulate trading over historical data

Runs the DWAP strategy over historical data to generate:
- Simulated open positions (what we would currently hold)
- Trade history (closed trades)
- Performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from app.core.config import settings
from app.services.scanner import scanner_service


@dataclass
class SimulatedPosition:
    """A position from backtesting"""
    id: int
    symbol: str
    shares: float
    entry_price: float
    entry_date: str
    current_price: float
    stop_loss: float
    profit_target: float
    pnl_pct: float
    pnl_dollars: float
    days_held: int
    dwap_at_entry: float
    pct_above_dwap_at_entry: float

    def to_dict(self):
        return asdict(self)


@dataclass
class SimulatedTrade:
    """A completed trade from backtesting"""
    id: int
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    days_held: int
    dwap_at_entry: float

    def to_dict(self):
        return asdict(self)


@dataclass
class BacktestResult:
    """Complete backtest results"""
    positions: List[SimulatedPosition]
    trades: List[SimulatedTrade]
    total_return_pct: float
    win_rate: float
    total_trades: int
    open_positions: int
    total_pnl: float
    max_drawdown_pct: float
    sharpe_ratio: float
    start_date: str
    end_date: str

    def to_dict(self):
        return {
            **asdict(self),
            'positions': [p.to_dict() for p in self.positions],
            'trades': [t.to_dict() for t in self.trades]
        }


class BacktesterService:
    """
    Simulates the DWAP trading strategy over historical data
    """

    def __init__(self):
        self.initial_capital = 100000
        self.position_size_pct = settings.POSITION_SIZE_PCT / 100  # 6%
        self.max_positions = settings.MAX_POSITIONS  # 15
        self.stop_loss_pct = settings.STOP_LOSS_PCT / 100  # 8%
        self.profit_target_pct = settings.PROFIT_TARGET_PCT / 100  # 20%
        self.dwap_threshold_pct = settings.DWAP_THRESHOLD_PCT / 100  # 5%
        self.min_volume = settings.MIN_VOLUME
        self.min_price = settings.MIN_PRICE
        self.volume_spike_mult = settings.VOLUME_SPIKE_MULT

    def run_backtest(
        self,
        lookback_days: int = 252,  # 1 year default
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest over historical data

        Args:
            lookback_days: Number of trading days to simulate
            end_date: End date for backtest (default: today)

        Returns:
            BacktestResult with positions, trades, and metrics
        """
        if not scanner_service.data_cache:
            raise RuntimeError("No data loaded. Run a scan first.")

        end_date = end_date or datetime.now()

        # Get all symbols with enough data
        symbols = []
        for symbol, df in scanner_service.data_cache.items():
            if len(df) >= lookback_days + 200:  # Need extra for DWAP calculation
                symbols.append(symbol)

        if not symbols:
            raise RuntimeError("Not enough historical data for backtest")

        # Initialize tracking
        capital = self.initial_capital
        positions: Dict[str, dict] = {}  # symbol -> position info
        trades: List[SimulatedTrade] = []
        equity_curve = []
        position_id = 0
        trade_id = 0

        # Get common date range
        sample_df = scanner_service.data_cache[symbols[0]]
        dates = sample_df.index[-lookback_days:]

        # Simulate each trading day
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')

            # Check existing positions for exits
            symbols_to_close = []
            for symbol, pos in positions.items():
                if symbol not in scanner_service.data_cache:
                    continue

                df = scanner_service.data_cache[symbol]
                if date not in df.index:
                    continue

                current_price = df.loc[date, 'close']
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']

                # Check stop loss
                if current_price <= pos['stop_loss']:
                    trade_id += 1
                    trades.append(SimulatedTrade(
                        id=trade_id,
                        symbol=symbol,
                        entry_date=pos['entry_date'],
                        exit_date=date_str,
                        entry_price=pos['entry_price'],
                        exit_price=current_price,
                        shares=pos['shares'],
                        pnl=round((current_price - pos['entry_price']) * pos['shares'], 2),
                        pnl_pct=round(pnl_pct * 100, 2),
                        exit_reason='stop_loss',
                        days_held=(date - pd.Timestamp(pos['entry_date'])).days,
                        dwap_at_entry=pos['dwap_at_entry']
                    ))
                    capital += pos['shares'] * current_price
                    symbols_to_close.append(symbol)

                # Check profit target
                elif current_price >= pos['profit_target']:
                    trade_id += 1
                    trades.append(SimulatedTrade(
                        id=trade_id,
                        symbol=symbol,
                        entry_date=pos['entry_date'],
                        exit_date=date_str,
                        entry_price=pos['entry_price'],
                        exit_price=current_price,
                        shares=pos['shares'],
                        pnl=round((current_price - pos['entry_price']) * pos['shares'], 2),
                        pnl_pct=round(pnl_pct * 100, 2),
                        exit_reason='profit_target',
                        days_held=(date - pd.Timestamp(pos['entry_date'])).days,
                        dwap_at_entry=pos['dwap_at_entry']
                    ))
                    capital += pos['shares'] * current_price
                    symbols_to_close.append(symbol)

            # Remove closed positions
            for symbol in symbols_to_close:
                del positions[symbol]

            # Look for new entries if we have room
            if len(positions) < self.max_positions:
                # Score all potential entries
                candidates = []

                for symbol in symbols:
                    if symbol in positions:
                        continue

                    df = scanner_service.data_cache[symbol]
                    if date not in df.index:
                        continue

                    row = df.loc[date]
                    price = row['close']
                    volume = row['volume']
                    dwap = row.get('dwap', np.nan)
                    vol_avg = row.get('vol_avg', np.nan)
                    ma_50 = row.get('ma_50', np.nan)
                    ma_200 = row.get('ma_200', np.nan)

                    if pd.isna(dwap) or dwap <= 0:
                        continue

                    pct_above_dwap = (price / dwap - 1)

                    # Check buy conditions
                    if (pct_above_dwap >= self.dwap_threshold_pct and
                        volume >= self.min_volume and
                        price >= self.min_price):

                        # Calculate signal strength
                        vol_ratio = volume / vol_avg if vol_avg > 0 else 0
                        is_strong = (
                            vol_ratio >= self.volume_spike_mult and
                            not pd.isna(ma_50) and not pd.isna(ma_200) and
                            price > ma_50 > ma_200
                        )

                        candidates.append({
                            'symbol': symbol,
                            'price': price,
                            'dwap': dwap,
                            'pct_above_dwap': pct_above_dwap,
                            'is_strong': is_strong,
                            'vol_ratio': vol_ratio
                        })

                # Sort by strength (strong first, then by pct above DWAP)
                candidates.sort(key=lambda x: (not x['is_strong'], -x['pct_above_dwap']))

                # Enter positions up to max
                for cand in candidates:
                    if len(positions) >= self.max_positions:
                        break

                    # Calculate position size
                    position_value = self.initial_capital * self.position_size_pct
                    if position_value > capital:
                        continue  # Not enough capital

                    shares = position_value / cand['price']
                    capital -= shares * cand['price']

                    position_id += 1
                    positions[cand['symbol']] = {
                        'id': position_id,
                        'entry_price': cand['price'],
                        'entry_date': date_str,
                        'shares': round(shares, 2),
                        'stop_loss': round(cand['price'] * (1 - self.stop_loss_pct), 2),
                        'profit_target': round(cand['price'] * (1 + self.profit_target_pct), 2),
                        'dwap_at_entry': round(cand['dwap'], 2),
                        'pct_above_dwap_at_entry': round(cand['pct_above_dwap'] * 100, 1)
                    }

            # Calculate daily equity
            position_value = sum(
                pos['shares'] * scanner_service.data_cache[sym].loc[date, 'close']
                for sym, pos in positions.items()
                if sym in scanner_service.data_cache and date in scanner_service.data_cache[sym].index
            )
            equity_curve.append({
                'date': date_str,
                'equity': capital + position_value
            })

        # Convert remaining positions to SimulatedPosition objects
        final_positions = []
        today = datetime.now()

        for symbol, pos in positions.items():
            df = scanner_service.data_cache[symbol]
            current_price = df.iloc[-1]['close']
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']

            final_positions.append(SimulatedPosition(
                id=pos['id'],
                symbol=symbol,
                shares=pos['shares'],
                entry_price=round(pos['entry_price'], 2),
                entry_date=pos['entry_date'],
                current_price=round(current_price, 2),
                stop_loss=pos['stop_loss'],
                profit_target=pos['profit_target'],
                pnl_pct=round(pnl_pct * 100, 2),
                pnl_dollars=round((current_price - pos['entry_price']) * pos['shares'], 2),
                days_held=(today - pd.Timestamp(pos['entry_date'])).days,
                dwap_at_entry=pos['dwap_at_entry'],
                pct_above_dwap_at_entry=pos['pct_above_dwap_at_entry']
            ))

        # Calculate metrics
        total_pnl = sum(t.pnl for t in trades)
        wins = [t for t in trades if t.pnl > 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        # Calculate max drawdown
        peak = equity_curve[0]['equity'] if equity_curve else self.initial_capital
        max_dd = 0
        for point in equity_curve:
            if point['equity'] > peak:
                peak = point['equity']
            dd = (peak - point['equity']) / peak
            if dd > max_dd:
                max_dd = dd

        # Final equity
        final_equity = equity_curve[-1]['equity'] if equity_curve else self.initial_capital
        total_return_pct = (final_equity - self.initial_capital) / self.initial_capital * 100

        # Sharpe ratio (simplified - daily returns)
        if len(equity_curve) > 1:
            returns = []
            for i in range(1, len(equity_curve)):
                daily_ret = (equity_curve[i]['equity'] - equity_curve[i-1]['equity']) / equity_curve[i-1]['equity']
                returns.append(daily_ret)
            if returns:
                avg_ret = np.mean(returns) * 252  # Annualized
                std_ret = np.std(returns) * np.sqrt(252)
                sharpe = avg_ret / std_ret if std_ret > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        return BacktestResult(
            positions=final_positions,
            trades=sorted(trades, key=lambda t: t.exit_date, reverse=True),
            total_return_pct=round(total_return_pct, 2),
            win_rate=round(win_rate, 1),
            total_trades=len(trades),
            open_positions=len(final_positions),
            total_pnl=round(total_pnl, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            start_date=dates[0].strftime('%Y-%m-%d'),
            end_date=dates[-1].strftime('%Y-%m-%d')
        )


# Singleton instance
backtester_service = BacktesterService()
