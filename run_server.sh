#!/usr/bin/env bash
# ==============================================================================
# Aalo Gateway - Start Local Server
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Ensure venv exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi

# Ensure .env exists
if [ ! -f ".env" ]; then
    echo ".env file not found!"
    echo "Please create a .env file with HOST, PORT, OLLAMA_BASE_URL, API_KEY and ALLOWED_ORIGINS."
fi

# Read port and host from .env or defaults
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

echo "=========================================================="
echo " Starting Aalo on http://$HOST:$PORT"
echo " Health check: http://localhost:$PORT/health"
echo "=========================================================="

exec .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
