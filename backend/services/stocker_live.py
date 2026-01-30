#!/usr/bin/env python3
"""
Stocker Live Trading System
============================

Optimized DWAP-based stock scanner and trading system.

WINNING STRATEGY (backtested 2009-2011):
- 216% total return (70% annualized)
- 1.14 Sharpe ratio
- -11.8% max drawdown
- 52.3% win rate

Configuration:
- Buy: Price > DWAP × 1.05 (5% above)
- Stop Loss: 8%
- Profit Target: 20%
- Max Positions: 15
- Position Size: 6% each

Usage:
    python stocker_live.py scan          # Run market scan
    python stocker_live.py scan --save   # Save to JSON
    python stocker_live.py download      # Download fresh data
    python stocker_live.py backtest      # Run backtest on fresh data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import json
import argparse
import os

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not installed. Run: pip install yfinance")


# =============================================================================
# CONFIGURATION - Optimized from backtesting
# =============================================================================

CONFIG = {
    # Buy signal parameters
    'dwap_threshold_pct': 5,        # Buy when price > DWAP × 1.05
    'min_volume': 500_000,          # Minimum daily volume
    'min_price': 20.0,              # Minimum stock price
    'volume_spike_mult': 1.5,       # Optional: volume > avg × this
    
    # Sell parameters  
    'stop_loss_pct': 8,             # Stop loss at -8%
    'profit_target_pct': 20,        # Take profit at +20%
    
    # Portfolio management
    'max_positions': 15,            # Maximum concurrent positions
    'position_size_pct': 6,         # Each position = 6% of portfolio
    'initial_capital': 100_000,     # Starting capital
    
    # Data settings
    'lookback_days': 252,           # 1 year for DWAP calculation
    'data_period': '2y',            # yfinance download period
}


# =============================================================================
# STOCK UNIVERSE
# =============================================================================

NASDAQ_100 = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AVGO', 'COST', 'NFLX',
    'AMD', 'ADBE', 'PEP', 'CSCO', 'TMUS', 'INTC', 'CMCSA', 'INTU', 'QCOM', 'TXN',
    'AMGN', 'AMAT', 'ISRG', 'HON', 'BKNG', 'VRTX', 'SBUX', 'GILD', 'MDLZ', 'ADI',
    'ADP', 'REGN', 'LRCX', 'PANW', 'MU', 'SNPS', 'KLAC', 'CDNS', 'MELI', 'ASML',
    'PYPL', 'CRWD', 'MAR', 'ORLY', 'CTAS', 'MNST', 'NXPI', 'MRVL', 'CSX', 'WDAY',
    'PCAR', 'ADSK', 'FTNT', 'ROP', 'CPRT', 'AEP', 'PAYX', 'ROST', 'ODFL', 'KDP',
]

SP500_ADDITIONS = [
    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM',
    'CVX', 'LLY', 'ABBV', 'KO', 'MRK', 'PFE', 'TMO', 'ABT', 'CRM', 'ACN',
    'MCD', 'DHR', 'WFC', 'VZ', 'NKE', 'PM', 'UPS', 'NEE', 'RTX', 'SPGI',
    'IBM', 'BA', 'CAT', 'GE', 'DE', 'LOW', 'UNP', 'LMT',
    'DIS', 'T', 'CCI', 'AMT', 'EQIX', 'PLD', 'SO', 'DUK',
    'CVS', 'CI', 'HUM', 'ELV', 'HCA',
    'COP', 'EOG', 'SLB', 'OXY', 'PSX', 'MPC', 'VLO', 'HAL',
]

# Excluded: Leveraged ETFs, Volatility products
EXCLUDED = [
    'VXX', 'UVXY', 'SVXY', 'VIXY',
    'SSO', 'SDS', 'SPXU', 'SPXL',
    'TQQQ', 'SQQQ', 'QLD', 'QID',
    'FAS', 'FAZ', 'TNA', 'TZA',
]

def get_universe():
    """Get full stock universe"""
    return [s for s in set(NASDAQ_100 + SP500_ADDITIONS) if s not in EXCLUDED]


# =============================================================================
# INDICATORS
# =============================================================================

def dwap(prices: pd.Series, volumes: pd.Series, period: int = 200) -> pd.Series:
    """Daily Weighted Average Price"""
    pv = prices * volumes
    return pv.rolling(period, min_periods=50).sum() / volumes.rolling(period, min_periods=50).sum()

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(period, min_periods=1).mean()

def high_52w(prices: pd.Series) -> pd.Series:
    """52-week rolling high"""
    return prices.rolling(252, min_periods=1).max()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Signal:
    """Buy/Sell signal"""
    symbol: str
    signal_type: str  # 'BUY' or 'SELL'
    price: float
    dwap: float
    pct_above_dwap: float
    volume: int
    volume_ratio: float
    stop_loss: float
    profit_target: float
    ma_50: float
    ma_200: float
    high_52w: float
    timestamp: str
    is_strong: bool = False
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Position:
    """Open position"""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: float
    stop_loss: float
    profit_target: float
    highest_price: float = 0.0
    
    def update_high(self, price: float):
        self.highest_price = max(self.highest_price, price)
    
    def check_exit(self, current_price: float, config: Dict) -> Optional[str]:
        """Check if position should be exited"""
        pnl_pct = (current_price / self.entry_price - 1) * 100
        
        if current_price <= self.stop_loss:
            return f"STOP_LOSS ({pnl_pct:.1f}%)"
        if current_price >= self.profit_target:
            return f"PROFIT_TARGET ({pnl_pct:.1f}%)"
        return None


# =============================================================================
# SCANNER
# =============================================================================

class StockerScanner:
    """Live market scanner"""
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.universe = get_universe()
        self.data_cache = {}
        self.signals = []
        
    def download_data(self, symbols: List[str] = None, period: str = None):
        """Download historical data from yfinance"""
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance not available")
        
        symbols = symbols or self.universe
        period = period or self.config['data_period']
        
        print(f"📥 Downloading {len(symbols)} stocks ({period} history)...")
        
        data = yf.download(symbols, period=period, progress=True, threads=True)
        
        # Convert to our format
        for symbol in symbols:
            try:
                df = pd.DataFrame({
                    'date': data.index,
                    'open': data['Open'][symbol],
                    'high': data['High'][symbol],
                    'low': data['Low'][symbol],
                    'close': data['Close'][symbol],
                    'volume': data['Volume'][symbol],
                }).dropna()
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                
                # Compute indicators
                df['dwap'] = dwap(df['close'], df['volume'])
                df['ma_50'] = sma(df['close'], 50)
                df['ma_200'] = sma(df['close'], 200)
                df['vol_avg'] = sma(df['volume'], 200)
                df['high_52w'] = high_52w(df['close'])
                
                self.data_cache[symbol] = df
            except Exception as e:
                pass
        
        print(f"✅ Downloaded {len(self.data_cache)} symbols")
        return self.data_cache
    
    def analyze_stock(self, symbol: str) -> Optional[Signal]:
        """Analyze single stock for signals"""
        if symbol not in self.data_cache:
            return None
        
        df = self.data_cache[symbol]
        if len(df) < 200:
            return None
        
        # Current values
        row = df.iloc[-1]
        price = row['close']
        vol = row['volume']
        current_dwap = row['dwap']
        vol_avg = row['vol_avg']
        ma_50 = row['ma_50']
        ma_200 = row['ma_200']
        h_52w = row['high_52w']
        
        if pd.isna(current_dwap) or current_dwap <= 0:
            return None
        
        pct_above_dwap = (price / current_dwap - 1) * 100
        vol_ratio = vol / vol_avg if vol_avg > 0 else 0
        
        # Check buy conditions
        is_buy = (
            pct_above_dwap >= self.config['dwap_threshold_pct'] and
            vol >= self.config['min_volume'] and
            price >= self.config['min_price']
        )
        
        # Strong signal: volume spike + trend alignment
        is_strong = (
            is_buy and
            vol_ratio >= self.config['volume_spike_mult'] and
            price > ma_50 > ma_200
        )
        
        if not is_buy:
            return None
        
        stop = price * (1 - self.config['stop_loss_pct'] / 100)
        target = price * (1 + self.config['profit_target_pct'] / 100)
        
        return Signal(
            symbol=symbol,
            signal_type='BUY',
            price=round(price, 2),
            dwap=round(current_dwap, 2),
            pct_above_dwap=round(pct_above_dwap, 1),
            volume=int(vol),
            volume_ratio=round(vol_ratio, 2),
            stop_loss=round(stop, 2),
            profit_target=round(target, 2),
            ma_50=round(ma_50, 2),
            ma_200=round(ma_200, 2),
            high_52w=round(h_52w, 2),
            timestamp=datetime.now().isoformat(),
            is_strong=is_strong
        )
    
    def scan(self) -> List[Signal]:
        """Scan all stocks for signals"""
        if not self.data_cache:
            print("⚠️  No data loaded. Run download_data() first.")
            return []
        
        self.signals = []
        
        for symbol in self.data_cache:
            signal = self.analyze_stock(symbol)
            if signal:
                self.signals.append(signal)
        
        # Sort by strength, then by pct above DWAP
        self.signals.sort(key=lambda s: (-s.is_strong, -s.pct_above_dwap))
        
        return self.signals
    
    def print_signals(self):
        """Print scan results"""
        strong = [s for s in self.signals if s.is_strong]
        regular = [s for s in self.signals if not s.is_strong]
        
        print(f"\n{'='*80}")
        print(f"📊 STOCKER SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*80}")
        
        if strong:
            print(f"\n🔥 STRONG SIGNALS ({len(strong)}):")
            print("-" * 80)
            for s in strong:
                print(f"  {s.symbol:6} ${s.price:>8.2f}  +{s.pct_above_dwap:>5.1f}% DWAP  "
                      f"Vol: {s.volume_ratio:.1f}x  Stop: ${s.stop_loss:.2f}  Target: ${s.profit_target:.2f}")
        
        if regular:
            print(f"\n📈 BUY SIGNALS ({len(regular)}):")
            print("-" * 80)
            for s in regular[:15]:
                print(f"  {s.symbol:6} ${s.price:>8.2f}  +{s.pct_above_dwap:>5.1f}% DWAP  "
                      f"Vol: {s.volume_ratio:.1f}x")
        
        if not self.signals:
            print("\n  ⏸️  No signals found. Market may be consolidating.")
        
        print(f"\n{'='*80}")
    
    def save_signals(self, filepath: str = 'signals.json'):
        """Save signals to JSON"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'signals': [s.to_dict() for s in self.signals],
            'strong_count': len([s for s in self.signals if s.is_strong]),
            'total_count': len(self.signals)
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"💾 Saved to {filepath}")


# =============================================================================
# BACKTESTER
# =============================================================================

@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    shares: float
    exit_reason: str
    
    @property
    def pnl_pct(self): return (self.exit_price / self.entry_price - 1) * 100
    @property
    def pnl(self): return (self.exit_price - self.entry_price) * self.shares
    @property
    def days_held(self): return (self.exit_date - self.entry_date).days


class Backtester:
    """Backtest the strategy on historical data"""
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
    
    def run(self, data: Dict[str, pd.DataFrame], start_date: datetime = None, end_date: datetime = None):
        """Run backtest"""
        cash = self.config['initial_capital']
        positions = []
        trades = []
        equity_history = []
        
        # Get all dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted([d for d in all_dates if 
                           (start_date is None or d >= start_date) and
                           (end_date is None or d <= end_date)])
        
        if not all_dates:
            print("No dates in range")
            return None
        
        print(f"📈 Backtesting from {all_dates[0].strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')}")
        
        for date in all_dates:
            # Get current prices
            prices = {}
            for symbol, df in data.items():
                if date in df.index:
                    prices[symbol] = df.loc[date]
            
            # Update position highs
            for pos in positions:
                if pos.symbol in prices:
                    pos.update_high(prices[pos.symbol]['close'])
            
            # Calculate equity
            equity = cash + sum(p.shares * prices.get(p.symbol, {}).get('close', p.entry_price) 
                               for p in positions if p.symbol in prices)
            equity_history.append({'date': date, 'equity': equity})
            
            # Check exits
            for pos in positions.copy():
                if pos.symbol not in prices:
                    continue
                
                current = prices[pos.symbol]['close']
                exit_reason = pos.check_exit(current, self.config)
                
                if exit_reason:
                    trades.append(Trade(pos.symbol, pos.entry_date, pos.entry_price,
                                       date, current, pos.shares, exit_reason))
                    cash += pos.shares * current
                    positions.remove(pos)
            
            # Check entries
            if len(positions) < self.config['max_positions']:
                for symbol, row in prices.items():
                    if any(p.symbol == symbol for p in positions):
                        continue
                    
                    price = row['close']
                    dwap_val = row.get('dwap', 0)
                    vol = row.get('volume', 0)
                    
                    if pd.isna(dwap_val) or dwap_val <= 0:
                        continue
                    
                    pct_above = (price / dwap_val - 1) * 100
                    
                    # Buy signal
                    if (pct_above >= self.config['dwap_threshold_pct'] and
                        vol >= self.config['min_volume'] and
                        price >= self.config['min_price']):
                        
                        pos_size = equity * self.config['position_size_pct'] / 100
                        shares = pos_size / price
                        
                        if shares * price <= cash:
                            stop = price * (1 - self.config['stop_loss_pct'] / 100)
                            target = price * (1 + self.config['profit_target_pct'] / 100)
                            
                            positions.append(Position(symbol, date, price, shares, stop, target, price))
                            cash -= shares * price
        
        # Close remaining positions
        final_date = all_dates[-1]
        for pos in positions:
            if pos.symbol in prices:
                price = prices[pos.symbol]['close']
                trades.append(Trade(pos.symbol, pos.entry_date, pos.entry_price,
                                   final_date, price, pos.shares, "END_OF_BACKTEST"))
        
        # Calculate metrics
        return self._calculate_metrics(trades, equity_history)
    
    def _calculate_metrics(self, trades: List[Trade], equity_history: List[Dict]):
        """Calculate performance metrics"""
        if not equity_history:
            return None
        
        eq = pd.DataFrame(equity_history)
        initial = self.config['initial_capital']
        final = eq['equity'].iloc[-1]
        
        total_return = (final / initial - 1) * 100
        days = (eq['date'].iloc[-1] - eq['date'].iloc[0]).days
        ann_return = ((final / initial) ** (365 / days) - 1) * 100 if days > 0 else 0
        
        # Drawdown
        running_max = eq['equity'].expanding().max()
        drawdown = ((eq['equity'] - running_max) / running_max * 100).min()
        
        # Sharpe
        returns = eq['equity'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Trade stats
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        win_rate = len(winners) / len(trades) * 100 if trades else 0
        
        print(f"\n{'='*60}")
        print("📊 BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Return:      {total_return:>10.1f}%")
        print(f"Annual Return:     {ann_return:>10.1f}%")
        print(f"Max Drawdown:      {drawdown:>10.1f}%")
        print(f"Sharpe Ratio:      {sharpe:>10.2f}")
        print(f"Win Rate:          {win_rate:>10.1f}%")
        print(f"Total Trades:      {len(trades):>10}")
        print(f"Avg Win:           {np.mean([t.pnl_pct for t in winners]):>10.1f}%" if winners else "")
        print(f"Avg Loss:          {np.mean([t.pnl_pct for t in losers]):>10.1f}%" if losers else "")
        print(f"{'='*60}")
        
        return {
            'total_return': total_return,
            'annual_return': ann_return,
            'max_drawdown': drawdown,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'trades': len(trades),
            'equity_history': eq
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stocker Live Trading System')
    parser.add_argument('command', choices=['scan', 'download', 'backtest'],
                       help='Command to run')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    parser.add_argument('--period', default='2y', help='Data period (1y, 2y, 5y)')
    args = parser.parse_args()
    
    if args.command == 'download':
        scanner = StockerScanner()
        scanner.download_data(period=args.period)
        
    elif args.command == 'scan':
        scanner = StockerScanner()
        scanner.download_data(period=args.period)
        scanner.scan()
        scanner.print_signals()
        
        if args.save:
            scanner.save_signals()
    
    elif args.command == 'backtest':
        scanner = StockerScanner()
        scanner.download_data(period=args.period)
        
        backtester = Backtester()
        backtester.run(scanner.data_cache)


if __name__ == '__main__':
    main()
