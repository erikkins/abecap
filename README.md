# 🚀 Stocker Trading System

A DWAP-based stock trading system with React dashboard and AWS backend.

**Optimized Strategy Results:**
- 📈 **216% total return** (70% annualized)
- 📊 **1.14 Sharpe ratio**
- 🛡️ **-11.8% max drawdown**
- 🎯 **52.3% win rate**

---

## 📁 Project Structure

```
stocker-app/
├── frontend/           # React + Vite + TailwindCSS
│   ├── src/
│   │   ├── components/ # React components
│   │   ├── hooks/      # Custom hooks
│   │   ├── pages/      # Page components
│   │   └── utils/      # Utilities
│   └── package.json
├── backend/            # FastAPI backend
│   ├── api/            # API endpoints
│   ├── services/       # Business logic
│   ├── models/         # Database models
│   └── requirements.txt
├── infrastructure/     # AWS deployment
│   ├── cdk/           # AWS CDK (TypeScript)
│   └── terraform/     # Terraform alternative
├── scripts/           # Helper scripts
└── .github/workflows/ # CI/CD
```

---

## 🛠️ Quick Start (Local Development)

### Prerequisites
- Node.js 18+
- Python 3.11+
- Docker & Docker Compose (optional)

### 1. Clone & Setup

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/stocker-app.git
cd stocker-app

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Setup frontend
cd ../frontend
npm install
```

### 2. Configure Environment

```bash
# Backend (.env)
cp backend/.env.example backend/.env
# Edit with your settings

# Frontend (.env)
cp frontend/.env.example frontend/.env
```

### 3. Run Locally

```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

Open http://localhost:5173

---

## 🐳 Docker Setup

```bash
# Build and run everything
docker-compose up --build

# Or run in background
docker-compose up -d
```

---

## ☁️ AWS Deployment

### Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ CloudFront  │────▶│     S3      │     │   Route53   │
│    (CDN)    │     │  (Frontend) │     │   (DNS)     │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ API Gateway │────▶│   Lambda    │────▶│    RDS      │
│   (REST)    │     │  (FastAPI)  │     │ (PostgreSQL)│
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│ EventBridge │────▶│   Lambda    │  (Daily scan at 4PM ET)
│  (Cron)     │     │  (Scanner)  │
└─────────────┘     └─────────────┘
```

### Option A: AWS CDK (Recommended)

```bash
# Install CDK
npm install -g aws-cdk

# Deploy
cd infrastructure/cdk
npm install
cdk bootstrap  # First time only
cdk deploy --all
```

### Option B: Terraform

```bash
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
```

---

## 📱 VS Code Setup

### 1. Create New Project

```bash
# Create folder
mkdir stocker-app && cd stocker-app

# Initialize git
git init

# Copy files from this template
# Or clone directly from GitHub
```

### 2. Recommended Extensions

Create `.vscode/extensions.json`:
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "bradlc.vscode-tailwindcss",
    "hashicorp.terraform",
    "ms-azuretools.vscode-docker"
  ]
}
```

### 3. VS Code Settings

Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "./backend/venv/bin/python",
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  }
}
```

---

## 🐙 GitHub Setup

### 1. Create Repository

```bash
# On GitHub: Create new repository "stocker-app"

# In your local folder:
git remote add origin https://github.com/YOUR_USERNAME/stocker-app.git
git branch -M main
git add .
git commit -m "Initial commit"
git push -u origin main
```

### 2. Add Secrets for CI/CD

Go to GitHub → Settings → Secrets → Actions, add:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret |
| `AWS_REGION` | e.g., `us-east-1` |

### 3. Enable GitHub Actions

The workflow files in `.github/workflows/` will automatically:
- Run tests on push
- Deploy to AWS on merge to `main`

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/signals/scan` | GET | Run market scan |
| `/api/signals/latest` | GET | Get latest signals |
| `/api/signals/watchlist` | GET | Get watchlist |
| `/api/portfolio/positions` | GET | Get open positions |
| `/api/portfolio/positions` | POST | Open new position |
| `/api/portfolio/trades` | GET | Get trade history |
| `/api/backtest/run` | POST | Run backtest |
| `/api/stocks/{symbol}` | GET | Get stock data |

---

## 🎯 Trading Strategy

```python
# Buy Signal
if price > dwap * 1.05 and volume > 500_000 and price > 20:
    buy()

# Sell Rules
if price <= entry_price * 0.92:  # -8% stop loss
    sell("STOP_LOSS")
if price >= entry_price * 1.20:  # +20% profit target
    sell("PROFIT_TARGET")
```

---

## 📈 Dashboard Features

- **Live Signals**: Real-time buy signals from scanner
- **Portfolio**: Track open positions with P&L
- **Equity Curve**: Interactive performance chart
- **Trade History**: Complete trade log
- **Watchlist**: Stocks approaching signal threshold
- **Settings**: Customize strategy parameters

---

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test

# E2E tests
npm run test:e2e
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request
