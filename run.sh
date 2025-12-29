#!/bin/bash

# Deployment-ready startup script for Universal File Compression Suite

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads downloads temp logs

# Check if port is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8000 is already in use. Stopping existing server..."
    pkill -f "uvicorn main:app" || true
    sleep 2
fi

# Start the server
echo "🚀 Starting Universal File Compression Suite..."
echo "📝 Logs: logs/app.log"
echo "🌐 Web UI: http://127.0.0.1:8000"
echo "📚 API Docs: http://127.0.0.1:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run with proper error handling
uvicorn main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --reload \
    --log-level info

