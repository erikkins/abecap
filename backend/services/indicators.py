"""
RigaCap Technical Indicators Library

This module implements all technical indicators from the original RigaCap SQL Server database,
optimized for vectorized operations with pandas/numpy.

Each function is designed to match the exact logic of the corresponding SQL function.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# MOVING AVERAGES
# =============================================================================

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average - matches fn50DMA / fn200DMA logic"""
    return series.rolling(window=period, min_periods=1).mean()


def ma_50(prices: pd.Series) -> pd.Series:
    """50-day Simple Moving Average (fn50DMA)"""
    return sma(prices, 50)


def ma_200(prices: pd.Series) -> pd.Series:
    """200-day Simple Moving Average (fn200DMA)"""
    return sma(prices, 200)


def volume_ma_50(volume: pd.Series) -> pd.Series:
    """50-day Volume Moving Average (fn50Volume)"""
    return sma(volume, 50)


def volume_ma_200(volume: pd.Series) -> pd.Series:
    """200-day Volume Moving Average (fn200Volume)"""
    return sma(volume, 200)


# =============================================================================
# DWAP (Daily Weighted Average Price)
# =============================================================================

def dwap(prices: pd.Series, volumes: pd.Series, period: int = 200) -> pd.Series:
    """
    Daily Weighted Average Price - matches fnDWAP logic
    
    DWAP = Σ(price × volume) / Σ(volume) over rolling period
    
    This is a volume-weighted price that gives more weight to high-volume days.
    It represents the "fair value" based on where most trading occurred.
    
    Original SQL logic:
        select @tempdwa = (@volume/convert(float,@totalvol))*@price
        -- summed across all days in period
    """
    # Calculate price × volume for weighted sum
    pv = prices * volumes
    
    # Rolling sums
    rolling_pv = pv.rolling(window=period, min_periods=1).sum()
    rolling_vol = volumes.rolling(window=period, min_periods=1).sum()
    
    # DWAP = weighted price sum / total volume
    return rolling_pv / rolling_vol


def dwap_50(prices: pd.Series, volumes: pd.Series) -> pd.Series:
    """50-day DWAP (fnDWAP50)"""
    return dwap(prices, volumes, 50)


def dwap_200(prices: pd.Series, volumes: pd.Series) -> pd.Series:
    """200-day DWAP (default fnDWAP period based on dateOf200)"""
    return dwap(prices, volumes, 200)


# =============================================================================
# 52-WEEK HIGH/LOW
# =============================================================================

def high_52_week(prices: pd.Series) -> pd.Series:
    """Rolling 52-week (252 trading days) high - matches fn52WeekHigh"""
    return prices.rolling(window=252, min_periods=1).max()


def high_52_week_date(prices: pd.Series) -> pd.Series:
    """Date of 52-week high - matches fn52WeekHighDate"""
    def get_max_date(window):
        if len(window) == 0:
            return pd.NaT
        return window.idxmax()
    
    return prices.rolling(window=252, min_periods=1).apply(
        lambda x: x.argmax(), raw=True
    )


def is_new_52_week_high(prices: pd.Series) -> pd.Series:
    """Check if current price is a new 52-week high"""
    rolling_max = prices.rolling(window=252, min_periods=1).max()
    return prices >= rolling_max


# =============================================================================
# TREND ANALYSIS
# =============================================================================

def slope(prices: pd.Series, period: int = 30) -> pd.Series:
    """
    Linear regression slope of price - matches fnSlope
    
    Original SQL:
        select @ret = (count(cnt)*sum(cnt*price)-sum(cnt)*sum(price))/
                      (count(cnt)*sum(cnt*cnt) - sum(cnt)*sum(cnt))
    
    Positive slope = uptrend, negative = downtrend
    """
    def calc_slope(y):
        n = len(y)
        if n < 2:
            return np.nan
        x = np.arange(n)
        # Linear regression slope formula
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                (n * np.sum(x ** 2) - np.sum(x) ** 2)
        return slope
    
    return prices.rolling(window=period, min_periods=2).apply(calc_slope, raw=True)


def slope_50dma(prices: pd.Series, period: int = 30) -> pd.Series:
    """Slope of 50-day moving average - matches fnSlope50"""
    ma = ma_50(prices)
    return slope(ma, period)


def slope_dwap(prices: pd.Series, volumes: pd.Series, period: int = 30) -> pd.Series:
    """Slope of DWAP - matches fnSlopeDWAP"""
    d = dwap(prices, volumes)
    return slope(d, period)


# =============================================================================
# SIGNAL DETECTION
# =============================================================================

def is_accumulation(prices: pd.Series) -> pd.Series:
    """
    Up day detection - matches IsAccumulation
    
    Returns True if today's price > yesterday's price
    """
    return prices > prices.shift(1)


def is_distribution(prices: pd.Series) -> pd.Series:
    """Down day detection (inverse of accumulation)"""
    return prices < prices.shift(1)


def is_price_cross_dwap(
    prices: pd.Series, 
    volumes: pd.Series,
    min_price: float = 25.0,
    min_avg_volume: int = 500_000
) -> pd.Series:
    """
    DWAP crossover detection - matches fnIsPriceCrossDWAP
    
    Returns True when:
    1. Previous price was BELOW DWAP
    2. Current price is ABOVE DWAP
    3. Price > minimum threshold
    4. Average volume > minimum threshold
    
    Original SQL logic:
        if @lastprice < @DWAP and @currprice > @DWAP
           and @lastprice > @minprice and @currprice > @minprice 
           and @avgvol > @minavgvol
    """
    d = dwap_200(prices, volumes)
    avg_vol = volume_ma_200(volumes)
    
    # Price conditions
    prev_below_dwap = prices.shift(1) < d.shift(1)
    curr_above_dwap = prices > d
    price_above_min = (prices > min_price) & (prices.shift(1) > min_price)
    
    # Volume condition - use 300K for lower-priced stocks
    volume_threshold = np.where(prices < 50, 300_000, min_avg_volume)
    volume_ok = avg_vol > volume_threshold
    
    return prev_below_dwap & curr_above_dwap & price_above_min & volume_ok


def is_50dma_cross_dwap(prices: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    50DMA crosses above DWAP - matches fnIs50DayCrossDWAP
    """
    ma = ma_50(prices)
    d = dwap_200(prices, volumes)
    
    prev_below = ma.shift(1) < d.shift(1)
    curr_above = ma > d
    
    return prev_below & curr_above


# =============================================================================
# VOLATILITY
# =============================================================================

def std_dev_1_month(prices: pd.Series) -> pd.Series:
    """1-month (21 trading days) standard deviation - matches StdDev1Month"""
    return prices.rolling(window=21, min_periods=1).std()


def std_dev_3_month(prices: pd.Series) -> pd.Series:
    """3-month (63 trading days) standard deviation - matches StdDev3Month"""
    return prices.rolling(window=63, min_periods=1).std()


def std_dev_1_year(prices: pd.Series) -> pd.Series:
    """1-year (252 trading days) standard deviation - matches StdDev1Year"""
    return prices.rolling(window=252, min_periods=1).std()


# =============================================================================
# VOLUME ANALYSIS
# =============================================================================

def is_volume_above_average(
    volumes: pd.Series, 
    multiplier: float = 1.5,
    period: int = 200
) -> pd.Series:
    """
    Volume spike detection - matches IsVolumeAboveAverage
    
    Returns True when current volume > average volume × multiplier
    """
    avg_vol = sma(volumes, period)
    return volumes > (avg_vol * multiplier)


def is_7_of_10_high_volume(volumes: pd.Series, multiplier: float = 1.0) -> pd.Series:
    """
    7 out of last 10 days have above-average volume - matches fn7of10HighVolume
    """
    avg_vol = volume_ma_200(volumes)
    above_avg = volumes > (avg_vol * multiplier)
    return above_avg.rolling(window=10).sum() >= 7


# =============================================================================
# HELPER DATACLASS FOR STOCK DATA
# =============================================================================

@dataclass
class StockIndicators:
    """Container for all calculated indicators for a stock"""
    
    # Raw data
    date: pd.DatetimeIndex
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    volume: pd.Series
    
    # Moving averages
    ma_50: pd.Series = None
    ma_200: pd.Series = None
    vol_ma_50: pd.Series = None
    vol_ma_200: pd.Series = None
    
    # DWAP
    dwap: pd.Series = None
    dwap_50: pd.Series = None
    
    # Highs/Lows
    high_52w: pd.Series = None
    
    # Trend
    slope_price: pd.Series = None
    slope_ma50: pd.Series = None
    slope_dwap: pd.Series = None
    
    # Signals
    is_accumulation: pd.Series = None
    dwap_cross: pd.Series = None
    new_52w_high: pd.Series = None
    
    @classmethod
    def from_ohlcv(cls, df: pd.DataFrame) -> 'StockIndicators':
        """
        Calculate all indicators from OHLCV dataframe
        
        Expected columns: date, open, high, low, close, volume
        """
        indicators = cls(
            date=df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['date']),
            open=df['open'] if 'open' in df.columns else df['close'],
            high=df['high'] if 'high' in df.columns else df['close'],
            low=df['low'] if 'low' in df.columns else df['close'],
            close=df['close'],
            volume=df['volume']
        )
        
        # Calculate all indicators
        indicators.ma_50 = ma_50(indicators.close)
        indicators.ma_200 = ma_200(indicators.close)
        indicators.vol_ma_50 = volume_ma_50(indicators.volume)
        indicators.vol_ma_200 = volume_ma_200(indicators.volume)
        
        indicators.dwap = dwap_200(indicators.close, indicators.volume)
        indicators.dwap_50 = dwap_50(indicators.close, indicators.volume)
        
        indicators.high_52w = high_52_week(indicators.close)
        
        indicators.slope_price = slope(indicators.close)
        indicators.slope_ma50 = slope_50dma(indicators.close)
        indicators.slope_dwap = slope_dwap(indicators.close, indicators.volume)
        
        indicators.is_accumulation = is_accumulation(indicators.close)
        indicators.dwap_cross = is_price_cross_dwap(indicators.close, indicators.volume)
        indicators.new_52w_high = is_new_52_week_high(indicators.close)
        
        return indicators
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export all indicators to a DataFrame"""
        data = {
            'close': self.close,
            'volume': self.volume,
            'ma_50': self.ma_50,
            'ma_200': self.ma_200,
            'vol_ma_50': self.vol_ma_50,
            'vol_ma_200': self.vol_ma_200,
            'dwap': self.dwap,
            'dwap_50': self.dwap_50,
            'high_52w': self.high_52w,
            'slope': self.slope_price,
            'slope_ma50': self.slope_ma50,
            'slope_dwap': self.slope_dwap,
            'is_accumulation': self.is_accumulation,
            'dwap_cross': self.dwap_cross,
            'new_52w_high': self.new_52w_high,
        }
        return pd.DataFrame(data, index=self.date)


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_against_sql(
    python_result: float, 
    sql_result: float, 
    tolerance: float = 0.01
) -> Tuple[bool, float]:
    """
    Compare Python calculation against SQL result
    Returns (is_match, difference)
    """
    if pd.isna(python_result) and pd.isna(sql_result):
        return True, 0.0
    if pd.isna(python_result) or pd.isna(sql_result):
        return False, float('inf')
    
    diff = abs(python_result - sql_result)
    pct_diff = diff / abs(sql_result) if sql_result != 0 else diff
    
    return pct_diff <= tolerance, pct_diff


if __name__ == "__main__":
    # Quick test with sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    
    # Simulate price data with trend + noise
    trend = np.linspace(100, 150, 500)
    noise = np.random.randn(500) * 5
    prices = pd.Series(trend + noise, index=dates)
    volumes = pd.Series(np.random.randint(500_000, 5_000_000, 500), index=dates)
    
    print("Sample Indicator Calculations:")
    print(f"Last Price: ${prices.iloc[-1]:.2f}")
    print(f"50-day MA: ${ma_50(prices).iloc[-1]:.2f}")
    print(f"200-day MA: ${ma_200(prices).iloc[-1]:.2f}")
    print(f"DWAP: ${dwap_200(prices, volumes).iloc[-1]:.2f}")
    print(f"52-week High: ${high_52_week(prices).iloc[-1]:.2f}")
    print(f"Price Slope (30d): {slope(prices).iloc[-1]:.4f}")
    print(f"DWAP Cross Today: {is_price_cross_dwap(prices, volumes).iloc[-1]}")
