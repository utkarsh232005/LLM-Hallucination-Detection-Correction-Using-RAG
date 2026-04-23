# 🚀 Quick Start - Local Development

## Prerequisites

Install these first:
1. **Python 3.11+** - [python.org](https://www.python.org/downloads/)
2. **Ollama** - [ollama.ai](https://ollama.ai)
3. **Node.js** (optional, for logging) - [nodejs.org](https://nodejs.org/)

## Fastest Setup (One Command)

```bash
chmod +x START.sh
./START.sh
```

The script will:
- Create Python virtual environment
- Install all dependencies
- Start Ollama, Backend, API, and Frontend

---

## Manual Setup (If Script Doesn't Work)

### 1️⃣ Terminal 1 - Start Ollama (LLM Service)
```bash
ollama serve
```
First run will pull models: `llama3.2:1b` and `nomic-embed-text`

### 2️⃣ Terminal 2 - Start Flask Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python server.py
```
Runs on: **http://localhost:8080**

### 3️⃣ Terminal 3 - Start Node API (Optional - For Logging)
```bash
cd api
npm install
node server.js
```
Runs on: **http://localhost:3001**

### 4️⃣ Terminal 4 - Start Frontend
```bash
cd frontend
python -m http.server 5500
```
Runs on: **http://localhost:5500**

---

## 🌐 Access Your App

| Service | URL |
|---------|-----|
| **Chat App** | http://localhost:5500 |
| **API** | http://localhost:8080 |
| **API Docs** | http://localhost:8080/api/docs |

---

## ⚙️ Configuration

Edit `.env` file in project root:

```env
OLLAMA_BASE_URL=http://localhost:11434
PINECONE_API_KEY=your-api-key
GOOGLE_SEARCH_API_KEY=your-api-key
SERPAPI_KEY=your-api-key
```

---

## 🆘 Troubleshooting

**Ollama not starting?**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Ensure models are pulled
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

**Port already in use?**
```bash
# Change port in backend/server.py and update .env
# Change port in frontend with: python -m http.server 5500
```

**Python dependencies failing?**
```bash
# Try installing without cache
pip install --no-cache-dir -r backend/requirements.txt
```

---

## 📂 Project Structure

```
hallucination-rag/
├── backend/          # Flask API (Python)
├── api/              # Logging API (Node.js)
├── frontend/         # Chat UI (HTML/JS)
├── .env              # Configuration
├── START.sh          # Auto startup script
└── README.md         # Full documentation
```

---

## ✨ Features

✅ Real-time hallucination detection  
✅ RAG-based correction  
✅ NLI CrossEncoder scoring  
✅ Local LLM (Ollama)  
✅ Web-grounded responses  
✅ MySQL audit logging  

---
