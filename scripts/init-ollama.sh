#!/bin/sh
set -eu

echo "Initializing Ollama models..."

# Give the service a moment to settle after health check passes.
sleep 5

if ollama list | grep -q "nomic-embed-text"; then
	echo "nomic-embed-text already present, skipping pull."
else
	echo "Pulling nomic-embed-text..."
	ollama pull nomic-embed-text
fi

if ollama list | grep -q "llama3.2:1b"; then
	echo "llama3.2:1b already present, skipping pull."
else
	echo "Pulling llama3.2:1b..."
	ollama pull llama3.2:1b
fi

echo "Ollama model initialization complete."
