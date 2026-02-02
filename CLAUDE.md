# CLAUDE.md - Stocker Trading System

> This file provides context for Claude Code CLI. It captures the full history and decisions from our development session.

## Project Overview

**Stocker** is a DWAP-based stock trading system with a React dashboard and FastAPI backend, designed for deployment on AWS.

## Optimized Trading Strategy

After backtesting **35+ rule combinations** on 2009-2011 data:

```
BUY SIGNAL:
- Price > DWAP × 1.05 (5% above 200-day Daily Weighted Average Price)
- Volume > 500,000
- Price > $20

SELL RULES:
- Stop Loss: -8%
- Profit Target: +20%

PORTFOLIO:
- Max 15 positions
- 6% of portfolio per position
```

**Backtest Results:**
- 216% total return (70% annualized)
- 1.14 Sharpe ratio
- -11.8% max drawdown
- 52.3% win rate

## Key Findings from Optimization

1. **DWAP 5% threshold** beats 10% - earlier entry captures more upside
2. **Volume spike filter (1.5x avg)** reduces drawdown significantly
3. **Fixed profit targets (20%)** outperform trailing stops in trending markets
4. **15 positions at 6% each** provides optimal diversification
5. **Exclude leveraged ETFs** (VXX, TQQQ, SQQQ, etc.) - too volatile

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

## Code Style

- Python: Black formatter, type hints preferred
- JavaScript: ESLint + Prettier, functional components, hooks
- Use Tailwind CSS for styling (no separate CSS files)
- Keep components in single files when under 500 lines
