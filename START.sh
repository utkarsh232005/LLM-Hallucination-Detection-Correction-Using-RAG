#!/bin/bash

# HalluciGuard - Local Development Startup Script
# This script starts all services locally for development

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=================================="
echo "🚀 HalluciGuard - Local Startup"
echo "=================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.11+"
    exit 1
fi

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not installed. Please install from: https://ollama.ai"
    echo "   Models needed: smollm2:360m, llama3.2:latest, nomic-embed-text"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js not installed. API logging won't work."
    echo "   Install Node.js to enable logging: https://nodejs.org"
fi

echo ""
echo "📋 Setup Steps:"
echo "   1. Starting Ollama..."
echo "   2. Starting Flask Backend (port 8080)..."
echo "   3. Starting Node API (port 3001)..."
echo "   4. Starting Frontend (port 5500)..."
echo ""

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "✓ Activating Python virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -q -r backend/requirements.txt

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env 2>/dev/null || cat > .env << 'EOF'
OLLAMA_BASE_URL=http://localhost:11434
PINECONE_API_KEY=your-key-here
GOOGLE_SEARCH_API_KEY=your-key-here
SERPAPI_KEY=your-key-here
EOF
fi

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "🎯 Starting Services..."
echo ""

# Terminal multiplexer detection (tmux/screen or just serial)
if command -v tmux &> /dev/null; then
    # Using tmux for parallel execution
    SESSION="halluciguard"
    
    # Kill existing session
    tmux kill-session -t $SESSION 2>/dev/null || true
    
    # Create new session
    tmux new-session -d -s $SESSION -x 200 -y 50
    
    # Window 1: Ollama
    tmux new-window -t $SESSION -n "ollama"
    tmux send-keys -t $SESSION:ollama "echo '🦙 Starting Ollama...' && ollama serve" C-m
    
    # Window 2: Flask Backend
    tmux new-window -t $SESSION -n "backend"
    tmux send-keys -t $SESSION:backend "sleep 3 && echo '🐍 Starting Flask Backend...' && cd backend && python server.py" C-m
    
    # Window 3: Node API
    tmux new-window -t $SESSION -n "api"
    tmux send-keys -t $SESSION:api "sleep 5 && echo '📡 Starting Node API...' && cd api && node server.js" C-m
    
    # Window 4: Frontend
    tmux new-window -t $SESSION -n "frontend"
    tmux send-keys -t $SESSION:frontend "sleep 2 && echo '🎨 Starting Frontend...' && cd frontend && python -m http.server 5500" C-m
    
    echo "✅ All services started in tmux session: $SESSION"
    echo ""
    echo "📺 View logs:"
    echo "   tmux attach-session -t $SESSION"
    echo ""
    echo "   Switch windows: Ctrl+B then [1-4]"
    echo "   Kill session: tmux kill-session -t $SESSION"
else
    # Manual startup instructions
    echo "📌 Start services in separate terminals:"
    echo ""
    echo "Terminal 1 - Ollama:"
    echo "  ollama serve"
    echo ""
    echo "Terminal 2 - Flask Backend (wait 3 sec):"
    echo "  cd backend && python server.py"
    echo ""
    echo "Terminal 3 - Node API (wait 5 sec):"
    echo "  cd api && node server.js"
    echo ""
    echo "Terminal 4 - Frontend:"
    echo "  cd frontend && python -m http.server 5500"
    echo ""
fi

echo ""
echo "=================================="
echo "🌐 Access Your App:"
echo "=================================="
echo "  Frontend:  http://localhost:5500"
echo "  API:       http://localhost:8080"
echo "  API Docs:  http://localhost:8080/api/docs"
echo ""
echo "=================================="
