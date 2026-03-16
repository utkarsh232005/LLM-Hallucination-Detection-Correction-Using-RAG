# LLM Hallucination Detection & Correction Using RAG

A Retrieval-Augmented Generation (RAG) system with **real-time hallucination detection** that automatically validates LLM-generated answers by comparing them against retrieved web content. When hallucination is detected, the system automatically provides corrected answers with source links.

## 🎯 New UI Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        HALLUCINATION DETECTION RAG SYSTEM                               │
├─────────────────────────────────┬──────────────────────────────────────────────────────┤
│                                 │                                                      │
│  🎯 HALLUCINATION DETECTION     │         💬 USER QUESTION ASKED TO LLM                │
│         REPORT                  │                                                      │
│                                 │    ┌──────────────────────────────────────────────┐  │
│  Overall Confidence: 87% ✅     │    │  User: What is machine learning?             │  │
│  Classification: WELL-GROUNDED  │    │                                              │  │
│                                 │    │  🦙 Llama: Machine learning is a subset...   │  │
│  Confidence Meter:              │    │                                              │  │
│  [████████████████████░░░] 87%  │    │  User: Next question...                      │  │
│                                 │    │                                              │  │
│  📊 Detailed Sentence Analysis: │    │  🤖 RAG: (shown when LLM hallucinated)       │  │
│  1. ✅ 94% - "Machine learning.."│    └──────────────────────────────────────────────┘  │
│  2. ✅ 86% - "It uses algo..."   │                                                      │
│  3. ✅ 81% - "Common apps..."    │         📥 INPUT SECTION                            │
│                                 │    ┌──────────────────────────────────────────────┐  │
│  💡 Interpretation:             │    │  [Ask anything...]  [🚀 Send] [🗑️ Clear]     │  │
│  HIGH CONFIDENCE - Well-grounded│    └──────────────────────────────────────────────┘  │
│                                 │                                                      │
├─────────────────────────────────┤                                                      │
│  📦 RAG UNIT                    │                                                      │
│                                 │                                                      │
│  🔗 Web Scrapped Links:         │                                                      │
│  1. wikipedia.org/...           │                                                      │
│  2. sciencedirect.com/...       │                                                      │
│                                 │                                                      │
│  📝 Textual Content (excerpts)  │                                                      │
│  & More...                      │                                                      │
└─────────────────────────────────┴──────────────────────────────────────────────────────┘
                                        ▲
                                        │ CONTINUOUS LOOP
                                        ▼
              ┌─────────────────────────────────────────────────────────┐
              │  IF ANALYZED LLM OUTPUT IS HALLUCINATED THEN            │
              │  GENERATE RAG BASED ANSWER OTHERWISE IGNORE RAG         │
              └─────────────────────────────────────────────────────────┘
```

## How It Works

```
┌─────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│ User Question   │ ──► │ LLM (Llama 3.2)    │ ──► │ Generate Answer  │
│ (right side UI) │     │ Direct Response    │     │                  │
└─────────────────┘     └────────────────────┘     └────────┬─────────┘
                                                            │
                        ┌───────────────────────────────────┘
                        │
                        ▼
     ┌──────────────────────────────────────────────────────────────┐
     │              HALLUCINATION ANALYSIS ENGINE                    │
     │  ┌──────────────┐    ┌──────────────────┐                    │
     │  │ LLM Output   │───►│ Token Matching   │                    │
     │  │ Tokens       │    │ with VDB Chunks  │                    │
     │  └──────────────┘    └────────┬─────────┘                    │
     │                               │                              │
     │  ┌──────────────┐    ┌────────▼─────────┐                    │
     │  │ Pinecone VDB │◄──►│ Embedding        │                    │
     │  │ (web chunks) │    │ Similarity       │                    │
     │  └──────────────┘    └────────┬─────────┘                    │
     │                               │                              │
     │                      ┌────────▼─────────┐                    │
     │                      │ Confidence Score │                    │
     │                      │ & Classification │                    │
     │                      └────────┬─────────┘                    │
     └───────────────────────────────┼──────────────────────────────┘
                                     │
                    ┌────────────────┴─────────────────┐
                    │                                  │
          ┌─────────▼─────────┐              ┌─────────▼─────────┐
          │ Confidence < 35%  │              │ Confidence >= 35% │
          │ = HALLUCINATED    │              │ = WELL-GROUNDED   │
          └─────────┬─────────┘              └─────────┬─────────┘
                    │                                  │
          ┌─────────▼─────────┐              ┌─────────▼─────────┐
          │ 🤖 RAG ACTIVATES  │              │ ✅ USE LLM ANSWER │
          │ • Scrape web      │              │ (No RAG needed)   │
          │ • Get sources     │              │                   │
          │ • Generate new    │              │                   │
          │   grounded answer │              │                   │
          └───────────────────┘              └───────────────────┘
```

### Step-by-Step Process

1. **User Input**: User enters a question in the chat interface (right side)
2. **LLM Response**: Llama 3.2 generates an immediate response
3. **Background Processing**: System automatically:
   - Searches web with SerpAPI (8 URLs)
   - Scrapes content from discovered URLs
   - Chunks documents (1800 chars, 250 overlap)
   - Embeds and stores in Pinecone
4. **Hallucination Detection**:
   - Embeds LLM answer sentences
   - Compares against Pinecone VDB embeddings
   - Calculates confidence scores per sentence
   - Token-level matching for additional validation
5. **Decision Logic**:
   - If confidence < 35% → **RAG provides corrected answer with sources**
   - If confidence >= 35% → **LLM answer is used as-is**
6. **Report Display**: Left panel shows detailed analysis

## Features

### Core RAG Capabilities
- **Question-first workflow** - No manual URL entry required
- **Automated web search** - Uses SerpAPI to discover fresh sources
- **Related image preview** - Visual context for user understanding
- **Persistent vector storage** - Pinecone-backed embeddings
- **Local LLM inference** - Ollama models for embeddings and generation
- **Cached model loading** - Streamlit resource caching for performance

### 🎯 Hallucination Detection System
- **Real-time validation** - Analyzes every generated answer automatically
- **Sentence-level analysis** - Granular scoring for each statement
- **Semantic similarity matching** - Uses cosine similarity between answer and source embeddings
- **Confidence scoring** - Overall confidence percentage (0-100%)
- **Visual reporting** - Color-coded confidence meter and detailed breakdown
- **Classification system**:
  - ✅ **WELL-GROUNDED** (75-100%): High confidence, strongly supported by sources
  - ⚠️ **PARTIALLY GROUNDED** (50-74%): Moderate confidence, partial support
  - ⚡ **WEAKLY GROUNDED** (30-49%): Low confidence, weak support
  - ❌ **LIKELY HALLUCINATED** (0-29%): Very low confidence, potential fabrication
- **Technical transparency** - Exposes metrics, processing time, and methodology
- **Interpretation guidance** - Context-aware explanations of confidence scores

## Prerequisites

- Python 3.10+
- Ollama installed and running
- Pinecone account and API key
- SerpAPI account and API key

## How Hallucination Detection Works

The hallucination detection system validates LLM-generated answers through semantic similarity analysis:

### 1. **Sentence Splitting**
The generated answer is split into individual sentences using regex patterns that handle abbreviations and punctuation correctly.

### 2. **Embedding Generation**
- Each sentence is converted to a 768-dimensional vector using the same `nomic-embed-text` model
- Retrieved context chunks are also embedded (cached for performance)

### 3. **Similarity Calculation**
For each sentence:
- Calculate cosine similarity with all retrieved context chunks
- Take the maximum similarity score (best match)
- Score ranges from 0 (unrelated) to 1 (identical)

### 4. **Confidence Aggregation**
- Average all sentence scores to get overall confidence
- Convert to percentage (0-100%)
- Classify based on thresholds

### 5. **Visual Reporting**
The system displays:
- **Overall Confidence**: Single score with emoji indicator
- **Confidence Meter**: Color-coded progress bar (green/orange/yellow/red)
- **Sentence Breakdown**: Individual scores for each statement
- **Interpretation Guide**: Context-aware explanation of results
- **Technical Details**: Processing time, model info, metadata

### Mathematical Foundation

```
Cosine Similarity = (A · B) / (||A|| × ||B||)

Where:
- A = sentence embedding vector
- B = context chunk embedding vector
- Result ∈ [-1, 1], normalized to [0, 1]

Overall Confidence = (Σ max_similarity_per_sentence / n_sentences) × 100%
```

### Why This Works

**High Similarity → Low Hallucination Risk**
- If an answer sentence is semantically similar to retrieved sources, it's likely grounded in factual content

**Low Similarity → High Hallucination Risk**
- If an answer sentence has low similarity with all sources, the LLM may be fabricating information from its training data



## Local Setup

### 1) Clone and enter project

```bash
git clone https://github.com/Rishiraj-Pathak-27/LLM-Hallucination-Detection-Correction-Using-RAG.git
cd LLM-Hallucination-Detection-Correction-Using-RAG
```

### 2) Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `streamlit` - Web UI framework
- `langchain` ecosystem - RAG orchestration
- `pinecone` - Vector database
- `google-search-results` - SerpAPI wrapper
- `numpy` - Numerical computations
- `scikit-learn` - Cosine similarity calculations
- `nltk` - Natural language processing utilities

### 4) Install and start Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
ollama pull nomic-embed-text
ollama serve
```

**Note**: The app uses `llama3.2:1b` for answer generation and `nomic-embed-text` for embeddings.

### 5) Create Pinecone index

Create an index with:

- Index name: rag-embedds
- Dimension: 768
- Metric: cosine

Important: The embedding model nomic-embed-text outputs 768-d vectors. If index dimension is different, upsert fails.

### 6) Configure API keys

Edit `config.py` or set environment variables:

```bash
export SERPAPI_API_KEY="your-serpapi-key"
export PINECONE_API_KEY="your-pinecone-key"
```

Or edit `config.py` directly:
```python
SERPAPI_API_KEY = "your-serpapi-key-here"
PINECONE_API_KEY = "your-pinecone-key-here"
```

### 7) Run app

**Backend + Web UI (Recommended):**
```bash
./start_backend.sh
```

Open: `http://localhost:8080`

**Streamlit app (optional):**
```bash
streamlit run app.py
```

## Usage

### New UI (`app.py`)

1. **Chat Interface (Right Side)**:
   - Type your question in the input box
   - Click "🚀 Send" to submit
   - LLM (Llama 3.2) generates a response

2. **Automatic Analysis**:
   - System automatically searches web and builds context
   - LLM output is analyzed against Pinecone VDB embeddings
   - Confidence score is calculated

3. **Hallucination Detection Report (Left Side)**:
   - Overall confidence percentage
   - Classification (WELL-GROUNDED / HALLUCINATED)
   - Visual confidence meter
   - Sentence-by-sentence analysis

4. **RAG Activation (When Needed)**:
   - If LLM answer is hallucinated (confidence < 35%)
   - RAG provides corrected answer with sources
   - Web scrapped links are displayed
   - Source context snippets are shown

5. **RAG Unit**:
   - Shows all web scrapped links
   - Displays extracted textual content
   - Shows number of chunks retrieved

### Example Flow

```
User: "What is quantum computing?"
         │
         ▼
┌─────────────────────────────────────────────┐
│ 🦙 LLM Answer: "Quantum computing uses..."  │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ 🎯 Analysis: Confidence 82% ✅ WELL-GROUNDED│
│ → LLM answer is used (no RAG needed)        │
└─────────────────────────────────────────────┘

User: "Who invented the flux capacitor?"
         │
         ▼
┌─────────────────────────────────────────────┐
│ 🦙 LLM Answer: "The flux capacitor was..."  │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ 🎯 Analysis: Confidence 18% ❌ HALLUCINATED │
│ → RAG activates with corrected answer       │
│ → Sources provided                          │
└─────────────────────────────────────────────┘
```

### Example Output

```
💬 USER QUESTION ASKED TO LLM

User: What is machine learning?

🦙 Llama: Machine learning is a subset of artificial intelligence...

───────────────────────────────────────────────

🎯 HALLUCINATION DETECTION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Confidence: 87% ✅
Classification: WELL-GROUNDED

Confidence Meter: [████████████████░░░░] 87%

📊 Detailed Sentence Analysis:
  1. ✅ 94% - "Machine learning is a subset..."
  2. ✅ 86% - "It uses algorithms to identify..."
  3. ✅ 81% - "Common applications include..."

💡 Interpretation: HIGH CONFIDENCE
The answer is strongly supported by retrieved context.
```

## Project Structure

```
├── server.py               # Flask backend + SSE streaming endpoints
├── static/index.html       # Web UI served by Flask
├── app.py                  # Streamlit version of the app
├── config.py               # Runtime/model configuration
├── start_backend.sh        # Local backend startup helper
├── Dockerfile              # Container image for rag-app service
├── docker-compose.yml      # Multi-service stack (rag-app + ollama + init)
├── init-ollama.sh          # One-time Ollama model bootstrap script
├── .env.example            # Environment variable template
├── .dockerignore           # Docker build ignore patterns
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── DockerPlan.md           # Docker containerization plan
├── plan.md                 # Hallucination detection implementation plan
├── cleanup.sh              # Utility script
└── Flow_of_rag/            # Documentation folder
   └── document-export-08-03-2026-16_48_09.md
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **UI** | Streamlit | Interactive web application |
| **Search** | SerpAPI | Web links + image results |
| **Scraping** | WebBaseLoader | Load web page content |
| **Splitter** | RecursiveCharacterTextSplitter | Chunk long text (2000/100) |
| **Embeddings** | nomic-embed-text (Ollama) | 768-d text vectorization |
| **Vector DB** | Pinecone | Store/retrieve embeddings |
| **LLM** | llama3.2:1b (Ollama) | Context-grounded answer generation |
| **Orchestration** | LangChain | Pipeline composition |
| **Similarity** | scikit-learn (cosine_similarity) | Compare embedding vectors |
| **NLP** | NumPy + regex | Sentence splitting & numerical ops |
| **Detection** | Custom implementation | Hallucination validation system |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Vector dimension mismatch** | Recreate Pinecone index with dimension 768 |
| **Pinecone API key missing** | Set PINECONE_API_KEY correctly in code or environment |
| **SerpAPI errors** | Check SERPAPI_API_KEY and account quota |
| **No scraped docs** | Query may return blocked/unreachable URLs |
| **Model not found** | Run `ollama pull llama3.2:1b` and `ollama pull nomic-embed-text` |
| **Ollama connection refused** | Ensure `ollama serve` is running |
| **Low confidence scores** | Normal for topics with poor web sources or LLM paraphrasing |
| **Slow hallucination detection** | First run caches embeddings; subsequent runs are faster |
| **Import errors (sklearn, numpy)** | Run `pip install -r requirements.txt` to install all dependencies |
| **Sentence splitting issues** | Check for unusual punctuation or formatting in answer |

### Performance Notes

- **First run**: Slower due to model loading and embedding caching
- **Subsequent runs**: Faster due to Streamlit's `@st.cache_resource` and `@st.cache_data`
- **Hallucination detection**: Adds 1-3 seconds per answer (depends on length)
- **Cache TTL**: Embeddings cached for 1 hour (configurable)

## Understanding Confidence Scores

### What the Scores Mean

| Range | Classification | Meaning | Action |
|-------|---------------|---------|--------|
| **75-100%** | ✅ WELL-GROUNDED | Answer closely matches sources | High trust, safe to use |
| **50-74%** | ⚠️ PARTIALLY GROUNDED | Moderate support from sources | Review for critical info |
| **30-49%** | ⚡ WEAKLY GROUNDED | Weak support, possible inference | Verify with sources |
| **0-29%** | ❌ LIKELY HALLUCINATED | Little to no source support | High risk, check sources |

### Factors Affecting Scores

**Lower Scores Don't Always Mean Hallucination:**
- LLM may paraphrase correctly but use different words
- Answer may synthesize information across multiple chunks
- Retrieved sources may lack coverage of certain aspects
- Model may add reasonable context or explanations

**Higher Scores Don't Guarantee Perfect Accuracy:**
- LLM might copy incorrect information from sources
- Source material itself could be inaccurate
- High similarity with low-quality sources is still problematic

### Best Practices

1. **Critical Information**: Always verify facts for important decisions
2. **Low Confidence**: Check original sources when scores < 50%
3. **Sentence-Level Review**: Expand individual sentences to see which statements are well-supported
4. **Source Quality**: Consider the quality of retrieved web pages
5. **Context Matters**: Use confidence scores as guidance, not absolute truth

## Docker Deployment

For containerized deployment instructions, see [DockerPlan.md](DockerPlan.md).

Quick start with Docker:
```bash
# Create env file from template
cp .env.example .env

# Add your API keys to .env, then start backend container
# (default uses host Ollama at http://host.docker.internal:11434)
docker compose up -d --build

# Check service status
docker compose ps

# Tail logs
docker compose logs -f rag-app

# App URL
# http://localhost:8080
```

### Docker Services

- `rag-app`: Flask API + static frontend on port `8080`
- `ollama` (optional): Local model runtime on port `11434` (profile: `with-ollama`)
- `ollama-init` (optional): Pulls `nomic-embed-text` and `llama3.2:1b` once (profile: `with-ollama`)

### Common Docker Commands

```bash
# Start/rebuild services
docker compose up -d --build

# Start with Docker-managed Ollama too (optional)
docker compose --profile with-ollama up -d --build

# Stop services
docker compose down

# Stop and remove Ollama model cache volume
docker compose down -v

# Verify models inside the Docker Ollama service
docker compose --profile with-ollama exec ollama ollama list
```

### Docker Notes

- Use `docker compose up -d --build` (not `docker compose up -d build`).
- Use `docker compose ...` (not `docker up ...`).
- Default compose run does not pull `ollama/ollama` images.
- Host Ollama models and Docker Ollama models are separate stores.
- Docker models are persisted in the `ollama-data` volume and reused across restarts.

## Future Enhancements

Potential improvements documented in [plan.md](plan.md):

- [ ] Adaptive confidence thresholds based on topic/domain
- [ ] Citation linking to specific source chunks
- [ ] Alternative answer generation when confidence is low
- [ ] Fact extraction and cross-referencing
- [ ] Historical confidence tracking and analytics
- [ ] PDF report export functionality
- [ ] Multi-language support
- [ ] Fine-tuned embeddings for specific domains

## Security Note

**⚠️ Important Security Practices:**

1. **Never commit API keys** to version control
2. **Use environment variables** in production
3. **Rotate keys regularly** if exposed
4. **Use .gitignore** to exclude sensitive files:
   ```
   .env
   .env.local
   *_key.txt
   ```
5. **Consider using secrets management** for production deployments (AWS Secrets Manager, Azure Key Vault, etc.)

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution

- Improving hallucination detection accuracy
- Adding support for more embedding models
- Implementing citation linking
- Enhancing UI/UX
- Writing tests
- Documentation improvements
- Performance optimizations

For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **Ollama** - Local LLM inference
- **Pinecone** - Vector database
- **LangChain** - RAG framework
- **Streamlit** - Web UI framework
- **SerpAPI** - Web search capabilities

---

