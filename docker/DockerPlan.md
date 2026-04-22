# Docker Containerization Plan

## 🎯 Objective
Containerize the LLM Hallucination Detection RAG Assistant to enable easy deployment, portability, and consistent runtime environment across different systems.

---

## 📐 Architecture Overview

### Container Architecture
```
───────────────────────────────────────────────────────────────
│                    Docker Network (rag-network)              │
│                                                               │
│  ┌───────────────────────────────┐   ┌──────────────────────┐│
│  │  Ollama Service          │   │  RAG App Container       ││
│  │  (ollama/ollama)         │◄──┤  (Streamlit)             ││
│  │                          │   │                          ││
│  │  - nomic-embed-text      │   │  - Python 3.11           ││
│  │  - llama3.2:1b           │   │  - Streamlit             ││
│  │  Port: 11434             │   │  - Dependencies          ││
│  │  Volume: ollama-data     │   │  Port: 8501              ││
│  └───────────────────────────────┘   └──────────────────────┘│
│                                           │                  │
│                                           │ (External APIs)  │
│                                           ▼                  │
│                                  ┌──────────────────────┐      │
│                                  │ Pinecone API       │      │
│                                  │ SerpAPI            │      │
│                                  └──────────────────────┘      │
└───────────────────────────────────────────────────────────────
```

---

## 📋 Component Breakdown

### 1. Ollama Service Container
**Purpose:** Provides LLM inference and embeddings
**Base Image:** `ollama/ollama:latest`
**Responsibilities:**
- Run nomic-embed-text model for embeddings
- Run llama3.2:1b model for text generation
- Expose API on port 11434

### 2. RAG Application Container
**Purpose:** Main Streamlit application
**Base Image:** `python:3.11-slim`
**Responsibilities:**
- Run Streamlit web interface
- Connect to Ollama service
- Connect to external APIs (Pinecone, SerpAPI)
- Expose UI on port 8501

---

## 🐳 Dockerfile Specification

### Dockerfile for RAG Application
