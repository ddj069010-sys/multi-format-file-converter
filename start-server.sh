#!/bin/bash

# File Compression Suite Startup Script

# Navigate to project directory
cd ~/DocFormatting

# Activate virtual environment
source venv/bin/activate

# Start the server in background
echo "🚀 Starting File Compression Suite..."
uvicorn main:app --host 127.0.0.1 --port 8000 > /tmp/compression-server.log 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > /tmp/compression-server.pid

# Wait for server to start
sleep 3

# Open in browser
echo "🌐 Opening browser..."
xdg-open http://127.0.0.1:8000

# Keep script running
wait $SERVER_PID
