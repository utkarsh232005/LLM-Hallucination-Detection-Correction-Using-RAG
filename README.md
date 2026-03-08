# LLM Hallucination Detection & Correction Using RAG

A Retrieval-Augmented Generation (RAG) assistant with **real-time hallucination detection** that validates and scores LLM-generated answers by comparing them against retrieved web content. The app searches the web, scrapes relevant pages, stores content in Pinecone, generates context-grounded answers, and provides detailed confidence scores showing how well each sentence is supported by the retrieved sources.

## How It Works

```
┌─────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│ Ask a Question  │ ──► │ Google Search      │ ──► │ Scrape Top URLs  │
│ (chat input)    │     │ (SerpAPI, top 5)   │     │ (WebBaseLoader)  │
└─────────────────┘     └────────────────────┘     └────────┬─────────┘
                                                             │
                                                             ▼
┌──────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│ Pinecone Store   │ ◄── │ Chunk + Embed      │ ◄── │ Split Documents  │
│ (rag-embedds)    │     │ (nomic-embed-text) │     │ (2000 / 100)     │
└────────┬─────────┘     └────────────────────┘     └──────────────────┘
         │
         ▼
┌──────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│ Retrieve Top 3   │ ──► │ Build Prompt       │ ──► │ Generate Answer  │
│ similar chunks   │     │ with context       │     │ (llama3.2:1b)    │
└────────┬─────────┘     └────────────────────┘     └────────┬─────────┘
         │                                                    │
         │                                                    ▼
         │                                          ┌──────────────────┐
         │                                          │ Split Answer     │
         │                                          │ into Sentences   │
         │                                          └────────┬─────────┘
         │                                                   │
         └───────────────────────────────────────────────────┘
                                  │
                                  ▼
                       ┌────────────────────────┐
                       │ Hallucination Detection│
                       │ (Cosine Similarity)    │
                       └────────┬───────────────┘
                                │
                                ▼
                       ┌────────────────────────┐
                       │ Display Confidence     │
                       │ Score & Report         │
                       └────────────────────────┘
```

### Step-by-Step Process

1. **User Input**: User enters a question in Streamlit chat.
2. **Web Search**: App fetches related Google results using SerpAPI and keeps top 5 links.
3. **Image Preview**: App fetches related Google Images (top 4) and displays them.
4. **Scraping**: App scrapes content from discovered URLs with WebBaseLoader.
5. **Chunking**: Documents are split into chunks (chunk size 2000, overlap 100).
6. **Embedding**: Chunks are embedded with nomic-embed-text (768 dimensions).
7. **Storage**: Embedded chunks are inserted into Pinecone index rag-embedds.
8. **Retrieval**: Similarity search retrieves top 3 relevant chunks.
9. **Generation**: llama3.2:1b generates a detailed response using retrieved context.
10. **Hallucination Detection**: 
    - Answer is split into individual sentences
    - Each sentence is embedded and compared against retrieved chunks
    - Cosine similarity scores measure grounding in source material
    - Overall confidence score is calculated (0-100%)
11. **Report Display**: Comprehensive report shows:
    - Overall confidence percentage and classification
    - Visual confidence meter with color coding
    - Sentence-by-sentence analysis with individual scores
    - Interpretation guide and technical details

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

Edit `rag_scrapper.py` and replace the placeholder values:

```python
SERPAPI_API_KEY = "your-serpapi-key-here"
PINECONE_API_KEY = "your-pinecone-key-here"
```

**Security Best Practice**: For production, use environment variables instead:

```bash
export SERPAPI_API_KEY="your-serpapi-key"
export PINECONE_API_KEY="your-pinecone-key"
```

Then modify the code to read from environment:
```python
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
```

### 7) Run app

```bash
streamlit run rag_scrapper.py
```

## Usage

1. **Ask a question** in the chat input box
2. **View sources**: App displays related images and discovered source URLs
3. **Wait for processing**: App scrapes, chunks, embeds, and stores content in Pinecone
4. **Read the answer**: App retrieves relevant chunks and generates a grounded response
5. **Check confidence**: Hallucination detection report shows:
   - Overall confidence score and classification
   - Color-coded confidence meter
   - Individual sentence scores (expandable)
   - Interpretation guide explaining the results
   - Technical details (processing time, model info)

### Example Output

```
👤 User: What is machine learning?

🤖 Assistant: Machine learning is a subset of artificial intelligence...

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
├── rag_scrapper.py          # Main application with RAG + hallucination detection
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── plan.md                 # Hallucination detection implementation plan
├── DockerPlan.md          # Docker containerization guide
├── cleanup.sh             # Utility script
└── Flow_of_rag/           # Documentation folder
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
# Build image
docker build -t rag-assistant .

# Run container
docker run -p 8501:8501 \
  -e SERPAPI_API_KEY=your_key \
  -e PINECONE_API_KEY=your_key \
  rag-assistant
```

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

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{rag_hallucination_detection,
  title = {LLM Hallucination Detection & Correction Using RAG},
  author = {Rishiraj Pathak},
  year = {2026},
  url = {https://github.com/Rishiraj-Pathak-27/LLM-Hallucination-Detection-Correction-Using-RAG}
}
```

---

