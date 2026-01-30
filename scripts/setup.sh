#!/bin/bash
# Stocker App - Quick Setup Script

set -e

echo "🚀 Stocker App Setup"
echo "===================="

# Check prerequisites
echo ""
echo "📋 Checking prerequisites..."

command -v node >/dev/null 2>&1 || { echo "❌ Node.js required. Install from https://nodejs.org"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 required. Install from https://python.org"; exit 1; }
command -v git >/dev/null 2>&1 || { echo "❌ Git required. Install from https://git-scm.com"; exit 1; }

echo "✅ All prerequisites found"

# Setup backend
echo ""
echo "📦 Setting up backend..."
cd backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt --quiet

# Create .env if not exists
if [ ! -f ".env" ]; then
    cat > .env << EOF
DATABASE_URL=postgresql://stocker:stocker@localhost:5432/stocker
REDIS_URL=redis://localhost:6379
DEBUG=true
EOF
    echo "Created backend/.env"
fi

cd ..

# Setup frontend
echo ""
echo "📦 Setting up frontend..."
cd frontend
npm install --silent

# Create .env if not exists
if [ ! -f ".env" ]; then
    cat > .env << EOF
VITE_API_URL=http://localhost:8000/api
EOF
    echo "Created frontend/.env"
fi

cd ..

# Done
echo ""
echo "✅ Setup complete!"
echo ""
echo "To start development:"
echo ""
echo "  Option 1 - Docker (recommended):"
echo "    docker-compose up"
echo ""
echo "  Option 2 - Manual:"
echo "    Terminal 1: cd backend && source venv/bin/activate && uvicorn main:app --reload"
echo "    Terminal 2: cd frontend && npm run dev"
echo ""
echo "Then open http://localhost:5173"
