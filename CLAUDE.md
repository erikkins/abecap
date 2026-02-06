# CLAUDE.md - Stocker Trading System

> This file provides context for Claude Code CLI. It captures the full history and decisions from our development session.

## Project Overview

**Stocker** is a momentum-based stock trading system with a React dashboard and FastAPI backend, designed for deployment on AWS.

## Trading Strategy v2 (Momentum)

The current strategy uses momentum-based ranking with market regime filtering:

```
BUY SIGNAL (Momentum Ranking):
- 10-day momentum (short-term)
- 60-day momentum (long-term)
- Composite score: short_mom × 0.5 + long_mom × 0.3 - volatility × 0.2
- Quality filter: Price > MA20 and MA50 (uptrend)
- Breakout filter: Within 5% of 50-day high
- Volume > 500,000
- Price > $20

SELL RULES:
- Trailing Stop: 15% from high water mark
- Market Regime Exit: SPY < 200-day MA → close all positions

PORTFOLIO:
- Max 5 positions
- 18% of portfolio per position
- Weekly rebalancing (Fridays)
```

**Backtest Results (2011-2026, 15 years):**
- 263% total return (9% annualized)
- 1.15 Sharpe ratio
- -14.2% max drawdown
- 49% win rate

**Recent Performance (2021-2026, 5 years):**
- 95% total return (14% annualized)
- 1.19 Sharpe ratio
- -10.5% max drawdown
- 47% win rate

## Legacy Strategy (DWAP)

The original DWAP strategy is still available for backward compatibility:

```
BUY SIGNAL:
- Price > DWAP × 1.05 (5% above 200-day Daily Weighted Average Price)
- Volume > 500,000
- Price > $20

SELL RULES:
- Stop Loss: -8%
- Profit Target: +20%
```

## Key Strategy Improvements (v1 → v2)

| Aspect | v1 (DWAP) | v2 (Momentum) |
|--------|-----------|---------------|
| Entry | DWAP threshold | Momentum ranking |
| Positions | 15 @ 6.6% | 5 @ 18% |
| Stop Loss | Fixed 8% | 15% trailing |
| Profit Target | Fixed 20% | Let winners run |
| Market Filter | None | SPY > 200MA |
| Rebalancing | Daily | Weekly |
| Sharpe | 0.19 | 1.48 |

## Architecture

```
Frontend (React + Vite + TailwindCSS)
    ↓
API Gateway / CloudFront
    ↓
Backend (FastAPI on Lambda)
    ↓
PostgreSQL (RDS) + Redis (ElastiCache)
    ↓
yfinance (data source - free, 20+ years history)
```

## Project Structure

```
stocker-app/
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main dashboard with charts, auth, trade flow
│   │   ├── main.jsx         # React entry point
│   │   └── index.css        # Tailwind styles
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── Dockerfile
├── backend/
│   ├── main.py              # FastAPI app with all endpoints
│   ├── services/
│   │   ├── scanner.py       # DWAP signal generation
│   │   ├── indicators.py    # Technical indicators
│   │   └── stocker_live.py  # Live scanning system
│   ├── requirements.txt
│   └── Dockerfile
├── infrastructure/
│   └── terraform/
│       └── main.tf          # AWS: S3, CloudFront, Lambda, API Gateway, RDS
├── docker-compose.yml       # Local development stack
└── .github/workflows/
    └── deploy.yml           # CI/CD pipeline
```

## Current Features

### Dashboard
- [x] Live signals panel with DWAP crossover detection
- [x] Open positions table with real-time P&L
- [x] Interactive 2-year stock charts (click signal or position)
- [x] Trade history tab with 2 years of historical trades
- [x] Authentication (Google, Apple, email)
- [x] Fluid buy/sell flow: Signal → Position → Trade History

### Backend
- [x] `/api/signals/scan` - Run market scan
- [x] `/api/signals/latest` - Get latest signals
- [x] `/api/portfolio/positions` - CRUD for positions
- [x] `/api/portfolio/trades` - Trade history
- [x] Mock data for demo mode

### Infrastructure
- [x] Docker Compose for local dev
- [x] Terraform for AWS deployment
- [x] GitHub Actions CI/CD

## TODO / Next Steps

### High Priority
- [ ] Connect frontend to real backend API (currently using mock data)
- [ ] Implement real yfinance data fetching in backend
- [ ] Add database persistence (PostgreSQL)
- [ ] Implement real OAuth (Google, Apple) - currently mocked

### Features
- [ ] Email/SMS alerts for new signals
- [ ] TradingView chart integration (instead of Recharts)
- [ ] Broker API integration (Alpaca, Interactive Brokers)
- [ ] Paper trading mode
- [ ] Watchlist for stocks approaching DWAP threshold
- [ ] Mobile responsive design improvements

### Infrastructure
- [ ] Deploy to AWS (terraform apply)
- [ ] Set up daily cron job for scanner (4 PM ET)
- [ ] Add CloudWatch monitoring/alerts
- [ ] Implement Redis caching for API responses

## Data Sources

**Primary: yfinance (Yahoo Finance)**
- Free, unlimited, no API key
- 20+ years historical daily OHLCV
- All US stocks, ETFs
- Split/dividend adjusted

```python
import yfinance as yf
df = yf.download("AAPL", period="2y")
```

**Stock Universe:**
- NASDAQ-100 (~100 stocks)
- S&P 500 additions (~50 stocks)
- Excludes: VXX, UVXY, TQQQ, SQQQ, FAS, FAZ, etc.

## Commands

### Local Development
```bash
# Docker (easiest)
docker-compose up

# Manual
cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
uvicorn main:app --reload --port 8000

cd frontend && npm install && npm run dev
```

### Lambda Container Deployment

**CRITICAL: AWS Lambda requires single-platform linux/amd64 images.**

Docker BuildKit creates multi-platform manifest lists with attestations that Lambda rejects with:
> "The image manifest, config or layer media type is not supported"

Always use `--provenance=false --sbom=false` when building Lambda container images:

```bash
# Build Lambda-compatible image
cd backend
docker buildx build --platform linux/amd64 --provenance=false --sbom=false -f Dockerfile.lambda -t rigacap-prod-api:latest --load .

# Or use the deploy script which handles this automatically:
./scripts/deploy-container.sh
```

**Note:** The Lambda init phase has a 10-second timeout. Database connections are initialized lazily on first request to avoid this timeout.

### Invoking Lambda Directly (IMPORTANT)

**API Gateway has a 29-second timeout.** For long-running operations (walk-forward simulations, backtests, AI optimization), you MUST invoke Lambda directly:

```bash
# Create payload file
cat > /tmp/payload.json << 'EOF'
{
  "walk_forward_job": {
    "start_date": "2024-02-06",
    "end_date": "2026-02-06",
    "frequency": "biweekly",
    "min_score_diff": 10.0,
    "enable_ai": false,
    "max_symbols": 100
  }
}
EOF

# Invoke Lambda directly (bypasses API Gateway timeout)
aws lambda invoke \
  --function-name rigacap-prod-api \
  --region us-east-1 \
  --invocation-type RequestResponse \
  --payload fileb:///tmp/payload.json \
  --cli-read-timeout 600 \
  /tmp/result.json

# Check results
cat /tmp/result.json | python3 -m json.tool
```

**DO NOT:**
- Call API Gateway endpoints for long operations (will timeout at 29s)
- Try to run simulations locally (no database/data available)

**Lambda payload types:**
- `walk_forward_job`: Walk-forward simulation
- `backtest_job`: Single backtest run
- `ai_optimization_job`: AI parameter optimization

### Deployment
```bash
cd infrastructure/terraform
terraform init
terraform apply -var="db_password=YourSecurePassword"
```

## Original Context

This project is a rebuild of a legacy Azure SQL stock prediction tool. The original system had:
- 48 database tables
- 45 technical indicators
- 46 trading rules in 127 stored procedures
- Daily Weighted Average Price (DWAP) as primary indicator

We ported the best-performing rules to Python, backtested extensively, and built a modern React + FastAPI stack.

## Session History Summary

1. Analyzed legacy model.xml (1.4MB) with full database schema
2. Extracted 46 trading rules from stored procedures
3. Implemented indicators in Python (indicators.py)
4. Built backtesting engine (backtester.py)
5. Ran comprehensive optimization (35+ combinations)
6. Identified winning strategy: DWAP 5% / Stop 8% / Target 20%
7. Built React dashboard with charts and auth
8. Created AWS infrastructure (Terraform)
9. Set up CI/CD (GitHub Actions)
10. **Upgraded to Momentum Strategy v2** (Sharpe 1.48)
    - Momentum-based ranking (10/60 day)
    - Trailing stops (15%)
    - Weekly rebalancing
    - Market regime filter (SPY > 200MA)

## Code Style

- Python: Black formatter, type hints preferred
- JavaScript: ESLint + Prettier, functional components, hooks
- Use Tailwind CSS for styling (no separate CSS files)
- Keep components in single files when under 500 lines
