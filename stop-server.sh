#!/bin/bash

if [ -f /tmp/compression-server.pid ]; then
    PID=$(cat /tmp/compression-server.pid)
    kill $PID 2>/dev/null
    echo "✅ Server stopped"
fi
