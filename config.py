"""
Configuration settings for Hallucination Detection RAG System
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== API KEYS ====================
# Loaded from .env file

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# ==================== PINECONE SETTINGS ====================

INDEX_NAME = "rag-embedds"
PINECONE_NAMESPACE = "web-rag-records"


# ==================== LLM SETTINGS ====================

# Main model used by chat generation.
# You can switch models quickly by setting this in your `.env` file:
# LLM_MODEL=llama3.2:1b
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")

# Optional dedicated model for RAG correction. Falls back to LLM_MODEL.
# Example in `.env`: RAG_LLM_MODEL=qwen2.5:3b
RAG_LLM_MODEL = os.getenv("RAG_LLM_MODEL", LLM_MODEL)

EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")

# Temperature settings
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))  # For general chat - more creative
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.1"))  # For RAG responses - more accurate


# ==================== HALLUCINATION DETECTION THRESHOLDS ====================

# Main threshold - below this confidence level, answer is considered hallucinated
HALLUCINATION_THRESHOLD = 50.0

# Mild hallucination threshold - between this and HALLUCINATION_THRESHOLD
MILD_HALLUCINATION_THRESHOLD = 55.0

# High confidence threshold
HIGH_CONFIDENCE_THRESHOLD = 75.0

# Weights for final hallucination score
EMBEDDING_WEIGHT = 0.65  # Weight for semantic similarity
TOKEN_WEIGHT = 0.35  # Weight for token-level matching


# ==================== CHUNKING PARAMETERS ====================
# Optimized for completeness

# Chunk size - larger for more complete context
CHUNK_SIZE = 1500

# Overlap between chunks
CHUNK_OVERLAP = 150

# Maximum number of chunks to index per query
MAX_CHUNKS = 30


# ==================== WEB SCRAPING SETTINGS ====================

# Maximum number of URLs to scrape
MAX_URLS = 3

# Maximum concurrent page loads
MAX_CONCURRENT_LOADS = 3


# ==================== RETRIEVAL SETTINGS ====================

# Number of chunks to retrieve for context
RETRIEVAL_K = 8

# Minimum relevance score for retrieved chunks
RELEVANCE_THRESHOLD = 0.30

# Relaxed threshold for fallback retrieval
RELAXED_THRESHOLD = 0.20


# ==================== UI SETTINGS ====================

# Chat history limit
MAX_CHAT_HISTORY = 50

# Sentence preview length in analysis
SENTENCE_PREVIEW_LENGTH = 100

# Maximum sentences to display in analysis
MAX_SENTENCE_DISPLAY = 5


# ==================== CACHING SETTINGS ====================

# TTL for embedding cache (in seconds)
EMBEDDING_CACHE_TTL = 3600  # 1 hour


# ==================== COLOR SCHEME ====================

COLORS = {
    "green": "#00c853",   # High confidence
    "orange": "#ff9800",  # Medium confidence
    "yellow": "#ffc107",  # Low confidence
    "red": "#f44336",     # Hallucinated
    "blue": "#0066ff",    # User messages
    "dark_green": "#2d5a2d",  # RAG messages
    "dark_gray": "#333333",   # LLM messages
}


# ==================== PROMPT TEMPLATES ====================

LLM_PROMPT_TEMPLATE = """
You are a helpful AI assistant named Llama. Answer the user's question directly and concisely.

Rules:
- Be informative and accurate
- Keep responses clear and well-structured
- Use 4-8 sentences for most answers
- Do not include any URLs or source citations

Question: {question}

Answer:
"""

RAG_PROMPT_TEMPLATE = """
You are a precise AI assistant. Answer using ONLY the provided context.

Rules:
- Use ONLY information from the context below
- Do not add information not present in context
- Be direct and factual
- Keep response clear and structured (4-8 sentences)
- Do not include URLs in your response

Question: {question}

Context:
{context}

Answer:
"""


# ==================== LOGGING ====================

ENABLE_LOGGING = True
LOG_LEVEL = "INFO"
