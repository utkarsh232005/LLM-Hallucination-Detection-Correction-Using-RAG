"""
Configuration for the Hallucination Detection RAG system.
All values can be overridden via environment variables or .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM models ────────────────────────────────────────────────────────────────
# Override in .env:  LLM_MODEL=smollm2:360m  RAG_LLM_MODEL=llama3.2:latest

LLM_MODEL        = os.getenv("LLM_MODEL",        "smollm2:360m")
<<<<<<< HEAD
RAG_LLM_MODEL    = os.getenv("RAG_LLM_MODEL",    LLM_MODEL)
=======
RAG_LLM_MODEL    = os.getenv("RAG_LLM_MODEL",    "llama3.2:latest")
>>>>>>> b87bf50d (feat: update model configurations and enhance hallucination detection)
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL",  "nomic-embed-text")

LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.7"))   # creative chat
RAG_TEMPERATURE  = float(os.getenv("RAG_TEMPERATURE", "0.1"))   # accurate RAG

# Generation controls to reduce latency/token usage.
LLM_NUM_PREDICT  = int(os.getenv("LLM_NUM_PREDICT", "128"))
RAG_NUM_PREDICT  = int(os.getenv("RAG_NUM_PREDICT", "192"))
LLM_NUM_CTX      = int(os.getenv("LLM_NUM_CTX", "2048"))
RAG_NUM_CTX      = int(os.getenv("RAG_NUM_CTX", "3072"))
RAG_CONTEXT_CHAR_LIMIT = int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", "3000"))
RETRIEVAL_K      = int(os.getenv("RETRIEVAL_K", "8"))

# ── Hallucination thresholds (used by app.py Streamlit UI) ───────────────────

HALLUCINATION_THRESHOLD      = 35.0   # below → hallucinated
MILD_HALLUCINATION_THRESHOLD = 55.0   # below → moderately hallucinated
