"""
Configuration for the Hallucination Detection RAG system.
All values can be overridden via environment variables or .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM models ────────────────────────────────────────────────────────────────
# Override in .env:  LLM_MODEL=llama3.2:3b  RAG_LLM_MODEL=qwen2.5:3b

LLM_MODEL        = os.getenv("LLM_MODEL",        "smollm2:360m")
RAG_LLM_MODEL    = os.getenv("RAG_LLM_MODEL",    LLM_MODEL)
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL",  "nomic-embed-text")

LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.7"))   # creative chat
RAG_TEMPERATURE  = float(os.getenv("RAG_TEMPERATURE", "0.1"))   # accurate RAG

# ── Hallucination thresholds (used by app.py Streamlit UI) ───────────────────

HALLUCINATION_THRESHOLD      = 35.0   # below → hallucinated
MILD_HALLUCINATION_THRESHOLD = 55.0   # below → moderately hallucinated
