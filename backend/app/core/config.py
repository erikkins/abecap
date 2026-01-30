"""
Configuration settings for the Stocker API
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Stocker API"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://stocker:stocker@localhost:5432/stocker"
    )
    
    # Redis (for caching)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://stocker.yourdomain.com",
    ]
    
    # Trading strategy (optimized from backtesting)
    DWAP_THRESHOLD_PCT: float = 5.0
    STOP_LOSS_PCT: float = 8.0
    PROFIT_TARGET_PCT: float = 20.0
    MAX_POSITIONS: int = 15
    POSITION_SIZE_PCT: float = 6.0
    MIN_VOLUME: int = 500_000
    MIN_PRICE: float = 20.0
    VOLUME_SPIKE_MULT: float = 1.5
    
    # Data
    DATA_LOOKBACK_DAYS: int = 252
    SCAN_INTERVAL_MINUTES: int = 15
    
    # AWS
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "stocker-data")
    
    class Config:
        env_file = ".env"


settings = Settings()


# Stock universe
NASDAQ_100 = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AVGO', 'COST', 'NFLX',
    'AMD', 'ADBE', 'PEP', 'CSCO', 'TMUS', 'INTC', 'CMCSA', 'INTU', 'QCOM', 'TXN',
    'AMGN', 'AMAT', 'ISRG', 'HON', 'BKNG', 'VRTX', 'SBUX', 'GILD', 'MDLZ', 'ADI',
    'ADP', 'REGN', 'LRCX', 'PANW', 'MU', 'SNPS', 'KLAC', 'CDNS', 'MELI', 'ASML',
    'PYPL', 'CRWD', 'MAR', 'ORLY', 'CTAS', 'MNST', 'NXPI', 'MRVL', 'CSX', 'WDAY',
]

SP500_ADDITIONS = [
    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM',
    'CVX', 'LLY', 'ABBV', 'KO', 'MRK', 'PFE', 'TMO', 'ABT', 'CRM', 'ACN',
    'MCD', 'DHR', 'WFC', 'VZ', 'NKE', 'PM', 'UPS', 'NEE', 'RTX', 'SPGI',
]

EXCLUDED_SYMBOLS = [
    'VXX', 'UVXY', 'SVXY', 'SSO', 'SDS', 'SPXU', 'TQQQ', 'SQQQ',
    'QLD', 'QID', 'FAS', 'FAZ', 'TNA', 'TZA',
]

def get_universe():
    """Get tradeable stock universe"""
    all_symbols = set(NASDAQ_100 + SP500_ADDITIONS)
    return [s for s in all_symbols if s not in EXCLUDED_SYMBOLS]
