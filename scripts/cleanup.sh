#!/bin/bash

# Cleanup script for Ollama models and Streamlit
echo "🧹 Starting cleanup process..."
echo ""

# Remove Ollama models
echo "📦 Removing Ollama models..."
echo "  - Removing nomic-embed-text..."
ollama rm nomic-embed-text

echo "  - Removing llama3.2:latest..."
ollama rm llama3.2:latest

echo "  - Removing smollm2:360m..."
ollama rm smollm2:360m

echo ""

# Uninstall Streamlit
echo "📦 Uninstalling Streamlit..."
pip uninstall -y streamlit

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Summary:"
echo "  - Removed Ollama models: nomic-embed-text, llama3.2:latest, smollm2:360m"
echo "  - Uninstalled Streamlit"
