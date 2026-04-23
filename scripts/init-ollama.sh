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

<<<<<<< HEAD
echo "✓ Ollama server is ready."
=======
if ollama list | grep -q "llama3.2:latest"; then
	echo "llama3.2:latest already present, skipping pull."
else
	echo "Pulling llama3.2:latest..."
	ollama pull llama3.2:latest
fi

if ollama list | grep -q "smollm2:360m"; then
	echo "smollm2:360m already present, skipping pull."
else
	echo "Pulling smollm2:360m..."
	ollama pull smollm2:360m
fi
>>>>>>> b87bf50d (feat: update model configurations and enhance hallucination detection)

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
pull_if_missing "nomic-embed-text"

echo "✅ Ollama model initialization complete."
