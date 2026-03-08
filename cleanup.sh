#!/bin/bash

# Cleanup script for Ollama models and Streamlit
echo "🧹 Starting cleanup process..."
echo ""

# Remove Ollama models
echo "📦 Removing Ollama models..."
echo "  - Removing nomic-embed-text..."
ollama rm nomic-embed-text

echo "  - Removing llama3.2:1b..."
ollama rm llama3.2:1b

echo ""

# Uninstall Streamlit
echo "📦 Uninstalling Streamlit..."
pip uninstall -y streamlit

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Summary:"
echo "  - Removed Ollama models: nomic-embed-text, llama3.2:1b"
echo "  - Uninstalled Streamlit"
