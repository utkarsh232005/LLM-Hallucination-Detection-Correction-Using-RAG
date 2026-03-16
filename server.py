"""
Flask backend for the Hallucination Detection RAG Web Application.
Exposes SSE streaming endpoint so the frontend gets live step-by-step updates.
"""

import json
import os
import re
import time
import uuid

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from sklearn.metrics.pairwise import cosine_similarity

try:
    from langchain_pinecone import PineconeVectorStore
    HAS_LANGCHAIN_PINECONE = True
except ImportError:
    from langchain_community.vectorstores import Pinecone as PineconeVectorStore
    HAS_LANGCHAIN_PINECONE = False

from config import (
    EMBEDDINGS_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RAG_LLM_MODEL,
    RAG_TEMPERATURE,
)

# ── env ──────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME       = "rag-embedds"
PINECONE_NS      = "web-rag-records"

HALLUCINATION_THRESHOLD      = 35.0
MILD_HALLUCINATION_THRESHOLD = 55.0
CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 300
MAX_CHUNKS    = 80
MAX_URLS      = 10
RETRIEVAL_K   = 12

# ── lazy singletons ───────────────────────────────────────────────────────────
_embeddings   = None
_llm          = None
_llm_rag      = None
_vector_store = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL)
    return _embeddings


def get_llm():
    global _llm
    if _llm is None:
        _llm = OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE, base_url=OLLAMA_BASE_URL)
    return _llm


def get_llm_rag():
    global _llm_rag
    if _llm_rag is None:
        _llm_rag = OllamaLLM(model=RAG_LLM_MODEL, temperature=RAG_TEMPERATURE, base_url=OLLAMA_BASE_URL)
    return _llm_rag


def missing_required_env_vars():
    """List required API keys that are currently missing."""
    missing = []
    if not SERPAPI_API_KEY:
        missing.append("SERPAPI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    return missing


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        emb = get_embeddings()
        if HAS_LANGCHAIN_PINECONE:
            _vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=emb)
        else:
            _vector_store = PineconeVectorStore.from_existing_index(
                index_name=INDEX_NAME, embedding=emb, text_key="text"
            )
    return _vector_store


# ── prompt templates ──────────────────────────────────────────────────────────
LLM_PROMPT = """You are a helpful AI assistant named Llama. Answer the user's question directly and concisely.

Rules:
- Be informative and accurate
- Keep responses clear and well-structured
- Use 4-8 sentences for most answers
- Do not include any URLs or source citations

Question: {question}

Answer:"""

RAG_PROMPT = """You are a precise AI assistant. Answer using ONLY the provided context.

Rules:
- Use ONLY information from the context below
- Do not add information not present in context
- Be direct and factual
- Keep response clear and structured (4-8 sentences)
- Do not include URLs in your response

Question: {question}

Context:
{context}

Answer:"""

SUMMARY_PROMPT = """Based on the web-scraped data below, provide a comprehensive summary with all key facts, numbers, and details that answer the question.

Question: {question}

Web Data:
{context}

Detailed Summary:"""


# ── helpers ───────────────────────────────────────────────────────────────────

def clear_namespace(namespace: str):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pc.Index(INDEX_NAME).delete(delete_all=True, namespace=namespace)
        time.sleep(0.5)
    except Exception:
        pass


def search_web(query: str, max_results: int = MAX_URLS):
    try:
        search   = SerpAPIWrapper()
        results  = search.results(query)
        urls     = []
        for r in results.get("organic_results", [])[:max_results]:
            urls.append({"url": r["link"], "title": r.get("title", ""), "snippet": r.get("snippet", "")})
        return urls
    except Exception:
        return []


def load_pages(urls):
    documents = []
    for url_info in urls[:3]:
        try:
            url  = url_info["url"] if isinstance(url_info, dict) else url_info
            docs = WebBaseLoader(url, requests_kwargs={"timeout": 5}).load()
            for doc in docs:
                if isinstance(url_info, dict):
                    doc.metadata["title"]   = url_info.get("title", "")
                    doc.metadata["snippet"] = url_info.get("snippet", "")
            documents.extend(docs)
            if len(documents) >= 2:
                break
        except Exception:
            continue
    return documents


def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)[:MAX_CHUNKS]


def index_documents(docs, vector_store, namespace: str):
    texts, metas, ids = [], [], []
    ts = int(time.time())
    for i, doc in enumerate(docs):
        texts.append(doc.page_content)
        metas.append({"source": doc.metadata.get("source", ""), "title": doc.metadata.get("title", ""),
                      "chunk_text": doc.page_content[:500], "created_at": ts})
        ids.append(f"chunk-{ts}-{i}-{uuid.uuid4().hex[:8]}")
    vector_store.add_texts(texts=texts, metadatas=metas, ids=ids, namespace=namespace)


def retrieve_documents(query: str, vector_store, namespace: str):
    try:
        raw = vector_store.similarity_search_with_relevance_scores(query, k=RETRIEVAL_K, namespace=namespace)
        docs = [(doc, float(sc)) for doc, sc in raw if sc >= 0.30]
        return docs or [(doc, float(sc)) for doc, sc in raw[:RETRIEVAL_K]]
    except Exception:
        raw = vector_store.similarity_search(query, k=RETRIEVAL_K, namespace=namespace)
        return [(doc, 1.0) for doc in raw]


def format_context(docs_with_scores):
    blocks = [re.sub(r"\s+", " ", doc.page_content.strip()) for doc, _ in docs_with_scores]
    return "\n\n---\n\n".join(blocks)


def sanitize_answer(text: str) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sentences(text: str):
    pattern   = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def embed_text(text: str, emb_model):
    return np.array(emb_model.embed_query(text))


def calc_cosine(v1, v2) -> float:
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.clip(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0], 0, 1))


def tokenize(text: str):
    stop = {"the","and","for","that","this","with","from","into","have","has","had","are","was","were",
            "will","would","shall","could","should","about","your","their","there","which","when","where",
            "what","who","why","how","does","did","can","also","than","then","them","they","its","our","you","been"}
    return [t for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-']+\b", text.lower())
            if len(t) >= 3 and t not in stop]


def extract_color_terms(text: str):
    """Extract common color words from text for contradiction checks."""
    color_terms = {
        "black", "white", "gray", "grey", "red", "green", "blue", "yellow", "orange",
        "purple", "pink", "brown", "teal", "cyan", "magenta", "silver", "gold",
        "light", "dark"
    }
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    return tokens.intersection(color_terms)


def token_support_analysis(answer: str, docs):
    ans_tokens = tokenize(answer)
    ctx_tokens = set()
    for doc in docs:
        ctx_tokens.update(tokenize(doc.page_content))
    unique = list(dict.fromkeys(ans_tokens))
    if not unique:
        return {"support_pct": 100.0, "matched": [], "unmatched": []}
    matched   = [t for t in unique if t in ctx_tokens]
    unmatched = [t for t in unique if t not in ctx_tokens]
    return {"support_pct": len(matched) / len(unique) * 100, "matched": matched[:15], "unmatched": unmatched[:15]}


def detect_hallucination(answer: str, retrieved_docs, emb_model, token_support_pct: float = 0.0):
    sentences = split_into_sentences(answer)
    if not sentences:
        return {"overall_confidence": 0, "is_hallucinated": True, "classification": "UNKNOWN",
                "sentence_scores": [], "metadata": {"processing_time": 0}}
    ctx_embs = []
    for doc in retrieved_docs:
        try:
            ctx_embs.append(embed_text(doc.page_content, emb_model))
        except Exception:
            pass
    if not ctx_embs:
        return {"overall_confidence": 0, "is_hallucinated": True, "classification": "NO CONTEXT",
                "sentence_scores": [], "metadata": {"processing_time": 0}}
    scores = []
    for sentence in sentences:
        try:
            s_emb = embed_text(sentence, emb_model)
            mx    = max(calc_cosine(s_emb, ce) for ce in ctx_embs)
            pct   = mx * 100
            if pct >= 75:   status = "NOT HALLUCINATED"
            elif pct >= 50: status = "SLIGHTLY HALLUCINATED"
            elif pct >= 30: status = "MODERATELY HALLUCINATED"
            else:           status = "HALLUCINATED"
            scores.append({"text": sentence[:100] + ("..." if len(sentence) > 100 else ""),
                           "score": mx, "score_pct": pct, "status": status})
        except Exception:
            pass
    embedding_confidence = float(np.mean([s["score"] for s in scores]) * 100) if scores else 0

    # Blend semantic similarity with token support to reduce false "safe" answers.
    combined_confidence = (embedding_confidence * 0.55) + (token_support_pct * 0.45)

    # Contradiction penalty for explicit color claims unsupported by retrieved context.
    answer_colors = extract_color_terms(answer)
    context_text = " ".join(doc.page_content for doc in retrieved_docs[:8])
    context_colors = extract_color_terms(context_text)
    color_penalty = 0.0
    if answer_colors and context_colors:
        mismatched_colors = [c for c in answer_colors if c not in context_colors]
        if mismatched_colors:
            color_penalty = min(40.0, float(len(mismatched_colors) * 18.0))

    confidence = max(0.0, combined_confidence - color_penalty)
    is_hall = confidence < HALLUCINATION_THRESHOLD
    if confidence >= 75:                          classification = "NOT HALLUCINATED"
    elif confidence >= MILD_HALLUCINATION_THRESHOLD: classification = "SLIGHTLY HALLUCINATED"
    elif confidence >= HALLUCINATION_THRESHOLD:   classification = "MODERATELY HALLUCINATED"
    else:                                         classification = "HIGHLY HALLUCINATED"
    return {"overall_confidence": confidence, "is_hallucinated": is_hall,
            "classification": classification, "sentence_scores": scores,
            "metadata": {
                "sentence_count": len(sentences),
                "chunks_analyzed": len(ctx_embs),
                "embedding_confidence": embedding_confidence,
                "token_support_pct": token_support_pct,
                "color_penalty": color_penalty,
            }}


def generate_llm_answer(question: str) -> str:
    chain = ChatPromptTemplate.from_template(LLM_PROMPT) | get_llm()
    return sanitize_answer(chain.invoke({"question": question}))


def generate_rag_answer(question: str, context: str) -> str:
    chain = ChatPromptTemplate.from_template(RAG_PROMPT) | get_llm_rag()
    return sanitize_answer(chain.invoke({"question": question, "context": context}))


def summarize_rag_context(question: str, context: str) -> str:
    try:
        chain = ChatPromptTemplate.from_template(SUMMARY_PROMPT) | get_llm_rag()
        return str(chain.invoke({"question": question, "context": context[:4000]})).strip()
    except Exception:
        return ""


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


@app.route("/")
def index():
    response = send_from_directory("static", "index.html")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/health", methods=["GET"])
def health_check():
    """Lightweight container health endpoint."""
    return jsonify({"status": "ok"}), 200


@app.after_request
def add_no_cache_headers(response):
    if request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.route("/api/chat/stream", methods=["GET"])
def chat_stream():
    """
    SSE endpoint.  Client calls:
      GET /api/chat/stream?q=<question>

    Events emitted (all JSON in the `data:` field):
      step     – {step, label, status}
      llm      – {text}
      analysis – {confidence, is_hallucinated, classification, sentence_scores, token_analysis}
      rag      – {text, sources, summary, is_hallucinated}
      done     – {}
      error    – {message}
    """
    question = request.args.get("q", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    missing = missing_required_env_vars()
    if missing:
        return jsonify({"error": "Missing required environment variables", "missing": missing}), 500

    def event(kind: str, payload: dict) -> str:
        return f"data: {json.dumps({'type': kind, **payload})}\n\n"

    def stream():
        try:
            vs  = get_vector_store()
            emb = get_embeddings()

            # ── Step 1: LLM ─────────────────────────────────────────────────
            yield event("step", {"step": 1, "label": "Generating LLM response…", "status": "running"})
            llm_answer = generate_llm_answer(question)
            yield event("llm",  {"text": llm_answer})
            yield event("step", {"step": 1, "label": "LLM response generated ✓", "status": "complete"})

            # ── Step 2: RAG context ─────────────────────────────────────────
            yield event("step", {"step": 2, "label": "Searching & scraping web sources…", "status": "running"})
            clear_namespace(PINECONE_NS)
            url_results = search_web(question)
            documents   = load_pages(url_results) if url_results else []
            chunks      = split_text(documents) if documents else []
            docs_ws     = []
            context     = ""
            docs        = []
            if chunks:
                index_documents(chunks, vs, PINECONE_NS)
                docs_ws = retrieve_documents(question, vs, PINECONE_NS)
                context = format_context(docs_ws)
                docs    = [d for d, _ in docs_ws]
            yield event("step", {"step": 2,
                                 "label": f"Found {len(url_results)} sources, {len(docs)} chunks ✓",
                                 "status": "complete"})

            # ── Step 3: Hallucination detection ─────────────────────────────
            yield event("step", {"step": 3, "label": "Analysing for hallucination…", "status": "running"})
            if docs:
                tok_an = token_support_analysis(llm_answer, docs)
                result = detect_hallucination(llm_answer, docs, emb, tok_an.get("support_pct", 0.0))
            else:
                result = {"overall_confidence": 20, "is_hallucinated": True,
                          "classification": "NO CONTEXT AVAILABLE", "sentence_scores": [],
                          "metadata": {"processing_time": 0}}
                tok_an = {"support_pct": 0, "matched": [], "unmatched": []}
            yield event("analysis", {
                "confidence":      result["overall_confidence"],
                "is_hallucinated": result["is_hallucinated"],
                "classification":  result["classification"],
                "sentence_scores": result["sentence_scores"],
                "token_analysis":  tok_an,
                "metadata":        result.get("metadata", {}),
                "source_count":    len(url_results),
                "retrieved_chunks": len(docs),
            })
            hall_pct = 100 - result["overall_confidence"]
            label    = f"Hallucination: {hall_pct:.0f}% — {'❌ HALLUCINATED' if result['is_hallucinated'] else '✅ NOT HALLUCINATED'}"
            yield event("step", {"step": 3, "label": label, "status": "complete"})

            # ── Step 4: Decision ─────────────────────────────────────────────
            yield event("step", {"step": 4, "label": "Deciding…", "status": "running"})
            if result["is_hallucinated"] and context:
                yield event("step", {"step": 4,
                                     "label": "Hallucination detected → generating RAG response…",
                                     "status": "running"})
                rag_answer = generate_rag_answer(question, context)
                summary    = summarize_rag_context(question, context)
                sources    = [
                    {"url": u["url"], "title": u.get("title", u["url"])[:80]}
                    for u in url_results[:3]
                    if isinstance(u, dict)
                ]
                yield event("rag", {"text": rag_answer, "sources": sources,
                                    "summary": summary, "is_hallucinated": True,
                                    "context_text": context[:20000],
                                    "llm_answer": llm_answer})
                yield event("step", {"step": 4, "label": "RAG response generated ✓", "status": "complete"})
            else:
                sources = [
                    {"url": u["url"], "title": u.get("title", u["url"])[:80]}
                    for u in url_results[:3]
                    if isinstance(u, dict)
                ]
                yield event("rag", {"text": llm_answer, "sources": sources,
                                    "summary": "", "is_hallucinated": False,
                                    "context_text": context[:20000],
                                    "llm_answer": llm_answer})
                yield event("step", {"step": 4,
                                     "label": "✅ LLM answer verified — no RAG needed",
                                     "status": "complete"})

            yield event("done", {})

        except Exception as exc:
            yield event("error", {"message": str(exc)})

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/models", methods=["GET"])
def list_models():
    """Return currently configured model names."""
    return jsonify({"llm": LLM_MODEL, "rag": RAG_LLM_MODEL, "embeddings": EMBEDDINGS_MODEL})


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    app.run(host=host, port=port, debug=False, threaded=True)
