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
    Simulates the momentum trading strategy over historical data

    Strategy v2 features:
    - Momentum-based ranking (10/60 day)
    - Trailing stops (15%)
    - Weekly rebalancing (Fridays)
    - Market regime filter (SPY > 200MA)
    """

    def __init__(self):
        self.initial_capital = 100000
        self.position_size_pct = settings.POSITION_SIZE_PCT / 100  # 18%
        self.max_positions = settings.MAX_POSITIONS  # 5
        self.trailing_stop_pct = settings.TRAILING_STOP_PCT / 100  # 15%
        self.short_mom_days = settings.SHORT_MOMENTUM_DAYS  # 10
        self.long_mom_days = settings.LONG_MOMENTUM_DAYS  # 60
        self.min_volume = settings.MIN_VOLUME
        self.min_price = settings.MIN_PRICE
        # Legacy DWAP settings for backward compatibility
        self.stop_loss_pct = settings.STOP_LOSS_PCT / 100  # 8%
        self.profit_target_pct = settings.PROFIT_TARGET_PCT / 100  # 20%
        self.dwap_threshold_pct = settings.DWAP_THRESHOLD_PCT / 100  # 5%
        self.volume_spike_mult = settings.VOLUME_SPIKE_MULT

    def _should_rebalance(self, date: pd.Timestamp, last_rebalance: Optional[pd.Timestamp]) -> bool:
        """
        Check if we should rebalance on this date (weekly on Fridays)

        Args:
            date: Current date
            last_rebalance: Date of last rebalance

        Returns:
            True if we should rebalance
        """
        is_friday = date.weekday() == 4
        if last_rebalance is None:
            return True
        days_since = (date - last_rebalance).days
        return is_friday and days_since >= 5

    def _check_market_regime(self, date: pd.Timestamp) -> bool:
        """
        Check if SPY is above 200-day MA (favorable market)

        Returns:
            True if market is favorable (SPY > 200MA)
        """
        if 'SPY' not in scanner_service.data_cache:
            return True  # Default to favorable if no SPY data

        spy_df = scanner_service.data_cache['SPY']
        if date not in spy_df.index:
            return True

        row = spy_df.loc[date]
        spy_price = row['close']
        spy_ma200 = row.get('ma_200', np.nan)

        if pd.isna(spy_ma200):
            return True

        return spy_price > spy_ma200

    def _calculate_momentum_score(self, symbol: str, date: pd.Timestamp) -> Optional[dict]:
        """
        Calculate momentum score for a symbol on a given date

        Returns:
            Dict with score info or None if invalid
        """
        if symbol not in scanner_service.data_cache:
            return None

        df = scanner_service.data_cache[symbol]
        if date not in df.index:
            return None

        # Get location of date and ensure we have enough history
        loc = df.index.get_loc(date)
        if loc < max(self.short_mom_days, self.long_mom_days, 50):
            return None

        row = df.loc[date]
        price = row['close']
        volume = row['volume']

        # Basic filters
        if price < self.min_price or volume < self.min_volume:
            return None

        # Calculate momentum
        short_mom_price = df.iloc[loc - self.short_mom_days]['close']
        long_mom_price = df.iloc[loc - self.long_mom_days]['close']

        short_mom = (price / short_mom_price - 1) * 100
        long_mom = (price / long_mom_price - 1) * 100

        # Calculate volatility (20-day)
        returns = df['close'].iloc[loc-19:loc+1].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 30

        # MA filters
        ma_20 = df['close'].iloc[loc-19:loc+1].mean()
        ma_50 = df['close'].iloc[loc-49:loc+1].mean() if loc >= 49 else price

        # Distance from 50-day high
        high_50d = df['close'].iloc[max(0, loc-49):loc+1].max()
        dist_from_high = (price / high_50d - 1) * 100

        # Quality filter
        passes_trend = price > ma_20 and price > ma_50
        passes_breakout = dist_from_high >= -settings.NEAR_50D_HIGH_PCT
        passes_quality = passes_trend and passes_breakout

        # Composite score
        composite_score = (
            short_mom * settings.SHORT_MOM_WEIGHT +
            long_mom * settings.LONG_MOM_WEIGHT -
            volatility * settings.VOLATILITY_PENALTY
        )

        return {
            'symbol': symbol,
            'price': price,
            'short_mom': short_mom,
            'long_mom': long_mom,
            'volatility': volatility,
            'composite_score': composite_score,
            'passes_quality': passes_quality,
            'dist_from_high': dist_from_high
        }

    def run_backtest(
        self,
        lookback_days: int = 252,  # 1 year default
        end_date: Optional[datetime] = None,
        use_momentum_strategy: bool = True
    ) -> BacktestResult:
        """
        Run backtest over historical data

        Args:
            lookback_days: Number of trading days to simulate
            end_date: End date for backtest (default: today)
            use_momentum_strategy: Use v2 momentum strategy (True) or legacy DWAP (False)

        Returns:
            BacktestResult with positions, trades, and metrics
        """
        if not scanner_service.data_cache:
            raise RuntimeError("No data loaded. Run a scan first.")

        end_date = end_date or datetime.now()

        # Get all symbols with enough data
        symbols = []
        for symbol, df in scanner_service.data_cache.items():
            if len(df) >= lookback_days + 200:  # Need extra for indicator calculation
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
        last_rebalance: Optional[pd.Timestamp] = None
        in_cash_mode = False  # True when market is unfavorable

        # Get common date range
        sample_df = scanner_service.data_cache[symbols[0]]
        dates = sample_df.index[-lookback_days:]

        # Simulate each trading day
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')

            # Check market regime (v2 strategy)
            if use_momentum_strategy and settings.MARKET_FILTER_ENABLED:
                market_favorable = self._check_market_regime(date)

                # If market turns unfavorable, close all positions
                if not market_favorable and not in_cash_mode:
                    in_cash_mode = True
                    for symbol in list(positions.keys()):
                        pos = positions[symbol]
                        df = scanner_service.data_cache[symbol]
                        if date not in df.index:
                            continue
                        current_price = df.loc[date, 'close']
                        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']

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
                            exit_reason='market_regime',
                            days_held=(date - pd.Timestamp(pos['entry_date'])).days,
                            dwap_at_entry=pos.get('dwap_at_entry', 0)
                        ))
                        capital += pos['shares'] * current_price
                    positions.clear()

                elif market_favorable:
                    in_cash_mode = False

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

                if use_momentum_strategy:
                    # Update high water mark and trailing stop
                    if current_price > pos.get('high_water_mark', pos['entry_price']):
                        pos['high_water_mark'] = current_price
                        pos['trailing_stop'] = current_price * (1 - self.trailing_stop_pct)

                    # Check trailing stop
                    if current_price <= pos.get('trailing_stop', 0):
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
                            exit_reason='trailing_stop',
                            days_held=(date - pd.Timestamp(pos['entry_date'])).days,
                            dwap_at_entry=pos.get('dwap_at_entry', 0)
                        ))
                        capital += pos['shares'] * current_price
                        symbols_to_close.append(symbol)
                else:
                    # Legacy DWAP strategy: fixed stop loss and profit target
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

            # Skip new entries if in cash mode (unfavorable market)
            if in_cash_mode:
                position_value = sum(
                    pos['shares'] * scanner_service.data_cache[sym].loc[date, 'close']
                    for sym, pos in positions.items()
                    if sym in scanner_service.data_cache and date in scanner_service.data_cache[sym].index
                )
                equity_curve.append({'date': date_str, 'equity': capital + position_value})
                continue

            # Rebalancing logic (v2: weekly; legacy: daily)
            should_rebalance = (
                self._should_rebalance(date, last_rebalance)
                if use_momentum_strategy
                else True
            )

            # Look for new entries if we have room and it's rebalance time
            if len(positions) < self.max_positions and should_rebalance:
                candidates = []

                if use_momentum_strategy:
                    # Momentum-based ranking
                    for symbol in symbols:
                        if symbol in positions:
                            continue
                        score_data = self._calculate_momentum_score(symbol, date)
                        if score_data and score_data['passes_quality']:
                            candidates.append(score_data)

                    # Sort by composite score (highest first)
                    candidates.sort(key=lambda x: -x['composite_score'])

                    # Enter positions up to max
                    for cand in candidates:
                        if len(positions) >= self.max_positions:
                            break

                        position_value = self.initial_capital * self.position_size_pct
                        if position_value > capital:
                            continue

                        shares = position_value / cand['price']
                        capital -= shares * cand['price']

                        position_id += 1
                        entry_price = cand['price']
                        positions[cand['symbol']] = {
                            'id': position_id,
                            'entry_price': entry_price,
                            'entry_date': date_str,
                            'shares': round(shares, 2),
                            'high_water_mark': entry_price,
                            'trailing_stop': round(entry_price * (1 - self.trailing_stop_pct), 2),
                            'dwap_at_entry': 0,  # Not used in momentum strategy
                            'pct_above_dwap_at_entry': 0,
                            'composite_score': cand['composite_score']
                        }

                    if candidates:
                        last_rebalance = date

                else:
                    # Legacy DWAP strategy
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

                        if (pct_above_dwap >= self.dwap_threshold_pct and
                            volume >= self.min_volume and
                            price >= self.min_price):

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

                    candidates.sort(key=lambda x: (not x['is_strong'], -x['pct_above_dwap']))

                    for cand in candidates:
                        if len(positions) >= self.max_positions:
                            break

                        position_value = self.initial_capital * self.position_size_pct
                        if position_value > capital:
                            continue

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

            # Handle both momentum (trailing_stop) and legacy (stop_loss/profit_target) positions
            stop_loss = pos.get('stop_loss') or pos.get('trailing_stop', 0)
            profit_target = pos.get('profit_target', 0)

            final_positions.append(SimulatedPosition(
                id=pos['id'],
                symbol=symbol,
                shares=pos['shares'],
                entry_price=round(pos['entry_price'], 2),
                entry_date=pos['entry_date'],
                current_price=round(current_price, 2),
                stop_loss=stop_loss,
                profit_target=profit_target,
                pnl_pct=round(pnl_pct * 100, 2),
                pnl_dollars=round((current_price - pos['entry_price']) * pos['shares'], 2),
                days_held=(today - pd.Timestamp(pos['entry_date'])).days,
                dwap_at_entry=pos.get('dwap_at_entry', 0),
                pct_above_dwap_at_entry=pos.get('pct_above_dwap_at_entry', 0)
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
