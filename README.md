# 🎯 LLM Hallucination Detection & Correction Using RAG

A real-time hallucination detection system that verifies LLM responses against live web sources and auto-corrects using Retrieval-Augmented Generation (RAG).

---

## 📸 Sample Output

| State | Screenshot |
|---|---|
| Light theme UI — welcome screen | `docs/screenshots/welcome.png` |
| Query in progress (SSE stream) | `docs/screenshots/streaming.png` |
| Hallucination detected + RAG correction | `docs/screenshots/hallucinated.png` |
| Not hallucinated result | `docs/screenshots/not_hallucinated.png` |
| MySQL Workbench — chat_logs table | `docs/screenshots/mysql_logs.png` |

> Add your screenshots to `docs/screenshots/` and replace the filenames above.

---

## 🧠 How It Works

```
User Query
    │
    ▼
┌──────────────┐    ┌────────────────────┐    ┌──────────────────────┐
│ Llama 3.2:1b │ →  │ SerpAPI + Web Scrape│ →  │ Hallucination Detect │
│  (LLM answer)│    │ Pinecone VectorDB   │    │ CrossEncoder (NLI)   │
└──────────────┘    └────────────────────┘    └──────────────────────┘
                                                         │
                                         ┌───────────────┴───────────────┐
                                         ▼                               ▼
                                  NOT HALLUCINATED              HALLUCINATED
                                  (trust LLM)               → RAG correction
                                                             → MySQL log saved
```

---

## 🤖 Model Information

### Hallucination Detection Model
- **Model:** [`Shreyash03Chimote/Hallucination_Detection`](https://huggingface.co/Shreyash03Chimote/Hallucination_Detection)
- **Type:** CrossEncoder (NLI — Natural Language Inference)
- **Hosted on:** HuggingFace 🤗 (no download needed — loaded automatically via `sentence-transformers`)
- **Task:** Given a (context, claim) pair → predicts Entailment / Contradiction / Neutral

### LLM (Chat + RAG)
- **Model:** `llama3.2:1b` via [Ollama](https://ollama.ai)
- **Local inference** — no API key required for the LLM

### Embeddings
- **Model:** `nomic-embed-text` via Ollama
- **Stored in:** Pinecone vector database

> ⚠️ No model weights need to be downloaded manually. All models load automatically on first run.

---

## 📊 Dataset

This project does **not use a static dataset**. It is a live RAG pipeline:

| Component | Source |
|---|---|
| Web context | [SerpAPI](https://serpapi.com) — real-time Google search results |
| Web content | Scraped via `langchain`'s `WebBaseLoader` |
| Vector index | Pinecone — rebuilt per query (ephemeral namespace) |
| Chat logs | MySQL (`rag_app.chat_logs`) |

If you want to test with a fixed dataset, you can pre-populate the Pinecone index manually using the vector store utilities in `backend/server.py`.

---

## 🗂️ Project Structure

```
hallucination-rag/
├── backend/                  # Python Flask AI backend
│   ├── server.py             # Main Flask app + RAG + hallucination detection
│   ├── config.py             # Model/threshold configuration
│   └── requirements.txt      # Python dependencies
│
├── api/                      # Node.js MySQL REST API
│   ├── server.js             # Express server (port 3001)
│   └── db.js                 # MySQL connection
│
├── frontend/                 # Static HTML/JS UI
│   ├── index.html            # Single-page chat interface
│   └── public/               # Favicons, web manifest
│
├── docs/                     # Documentation & diagrams
│   ├── README.md             # Project README (this file)
│   ├── plan.md               # Technical planning notes
│   └── Flow_of_rag/          # Architecture diagram
│
├── scripts/                  # Shell scripts
│   ├── start_backend.sh      # Start Python Flask server
│   └── cleanup.sh            # Clean up processes
│
├── docker/                   # Docker configs
├── .env.example              # Template for environment variables
└── docker-compose.yml
```

---

## ⚙️ Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Flask backend |
| Node.js | 18+ | MySQL API |
| Ollama | Latest | Local LLM + embeddings |
| MySQL | 8.0+ | Chat log persistence |
| [SerpAPI key](https://serpapi.com) | — | Web search |
| [Pinecone key](https://pinecone.io) | — | Vector database |
| [HuggingFace token](https://huggingface.co/settings/tokens) | — | CrossEncoder model |

---

## 🚀 Setup & Execution

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/hallucination-rag.git
cd hallucination-rag
```

### 2. Configure environment variables
```bash
cp .env.example .env
# Fill in your API keys (see .env.example for all required keys)
```

### 3. Install Ollama models
```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### 4. Set up MySQL database
```sql
CREATE DATABASE rag_app;
USE rag_app;

CREATE TABLE chat_logs (
  id                  INT AUTO_INCREMENT PRIMARY KEY,
  query               TEXT NOT NULL,
  llm_response        TEXT,
  rag_response        TEXT,
  is_hallucinated     TINYINT(1)   DEFAULT 0,
  hallucination_score FLOAT,
  classification      VARCHAR(50),
  sentence_count      INT          DEFAULT 0,
  sources_count       INT          DEFAULT 0,
  sources             JSON,
  model_id            VARCHAR(150),
  response_time_ms    INT,
  created_at          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);
```

### 5. Start the Python Flask backend
```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

python backend/server.py
# → Running at http://127.0.0.1:8080
```

### 6. Start the Node.js MySQL API
```bash
npm install
node api/server.js
# → Running at http://127.0.0.1:3001
```

### 7. Open the UI
Open `frontend/index.html` in VS Code with **Live Server**, or visit:
```
http://127.0.0.1:8080/static/
```

---

## 🌐 API Endpoints

### Flask (`:8080`)
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/chat/stream?q=<query>` | SSE stream — LLM + RAG + hallucination |
| `GET` | `/api/models` | Currently configured model names |

### Node.js (`:3001`)
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/save` | Save a chat log to MySQL |
| `GET` | `/api/history` | Fetch last 100 chat logs |
| `DELETE` | `/api/history/:id` | Delete a specific log |

---

## 🔑 Environment Variables

See [`.env.example`](.env.example) for the full list. Required:

```env
SERPAPI_API_KEY=your_serpapi_key
PINECONE_API_KEY=your_pinecone_key
HF_API_TOKEN=your_huggingface_token
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (`llama3.2:1b`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector DB | Pinecone |
| Web Search | SerpAPI |
| Hallucination Detection | HuggingFace CrossEncoder (NLI) |
| Backend (AI) | Python / Flask / LangChain |
| Backend (Logging) | Node.js / Express |
| Database | MySQL 8 |
| Frontend | Vanilla HTML / CSS / JS |

---

## 📝 License

MIT
