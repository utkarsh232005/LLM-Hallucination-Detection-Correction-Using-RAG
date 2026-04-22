#!/usr/bin/env bash
set -euo pipefail

# scripts/ lives inside project root — resolve the parent
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/venv"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
USER_AGENT_VALUE="${USER_AGENT:-RAGDetector/1.0}"
INSTALL_DEPS="false"

for arg in "$@"; do
  case "$arg" in
    --install)
      INSTALL_DEPS="true"
      ;;
    --help|-h)
      cat <<'EOF'
Usage: ./scripts/start_backend.sh [--install]

Options:
  --install   Install/refresh Python dependencies before start
  -h, --help  Show this help

Environment variables:
  PORT        Backend port (default: 8080)
  HOST        Backend host (default: 0.0.0.0)
  USER_AGENT  User agent for web requests (default: RAGDetector/1.0)
EOF
      exit 0
      ;;
  esac
done

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ "$INSTALL_DEPS" == "true" ]]; then
  echo "[setup] Installing dependencies from backend/requirements.txt"
  pip install -r "$ROOT_DIR/backend/requirements.txt"
fi

if ! command -v lsof >/dev/null 2>&1; then
  echo "[warn] lsof not found, skipping port conflict check"
else
  if lsof -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[info] Port $PORT already in use. Stop it with: lsof -ti:$PORT | xargs kill -9"
    exit 1
  fi
fi

export USER_AGENT="$USER_AGENT_VALUE"
export PORT
export HOST

echo "[run] Starting backend on http://127.0.0.1:$PORT"
exec python "$ROOT_DIR/backend/server.py"
