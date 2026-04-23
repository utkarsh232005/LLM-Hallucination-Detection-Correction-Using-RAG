#!/bin/sh
set -eu

OLLAMA_URL="http://${OLLAMA_HOST:-ollama:11434}"

echo "Waiting for Ollama server at $OLLAMA_URL..."

# Wait up to 120s for the Ollama server to become available
MAX_RETRIES=60
RETRY=0
until ollama list > /dev/null 2>&1; do
  RETRY=$((RETRY + 1))
  if [ "$RETRY" -ge "$MAX_RETRIES" ]; then
    echo "ERROR: Ollama server did not start within $((MAX_RETRIES * 2))s"
    exit 1
  fi
  sleep 2
done

echo "✓ Ollama server is ready."

# ── Pull models only if not already present ───────────────────────────────────
pull_if_missing() {
  model="$1"
  if ollama list 2>/dev/null | grep -q "$model"; then
    echo "✓ $model already present — skipping pull."
  else
    echo "⬇ Pulling $model (this may take a few minutes on first run)..."
    ollama pull "$model"
    echo "✓ $model pulled successfully."
  fi
}

pull_if_missing "smollm2:360m"
pull_if_missing "gemma2:2b"
pull_if_missing "nomic-embed-text"

echo "✅ Ollama model initialization complete."
