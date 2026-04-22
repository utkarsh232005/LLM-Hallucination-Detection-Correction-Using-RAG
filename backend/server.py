"""
Flask backend for the Hallucination Detection RAG Web Application.
Exposes SSE streaming endpoint so the frontend gets live step-by-step updates.
"""

import json
import logging
import os
import re
import time
import uuid

import threading
import numpy as np
import requests
from sentence_transformers import CrossEncoder
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
os.environ.setdefault("USER_AGENT", "hallucination-rag/1.0")
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME       = "rag-embedds"
PINECONE_NS      = "web-rag-records"
HALLUCINATION_MODEL_ID = os.getenv(
    "HALLUCINATION_MODEL_ID",
    "Shreyash03Chimote/Hallucination_Detection",
)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))
HF_INFERENCE_TIMEOUT_SEC = float(os.getenv("HF_INFERENCE_TIMEOUT_SEC", "30"))

HALLUCINATION_THRESHOLD      = 35.0
CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 300
MAX_CHUNKS    = 80
MAX_URLS      = 10
RETRIEVAL_K   = 12

_embeddings   = None
_llm          = None
_llm_rag      = None
_vector_store = None
_hallucination_model = None
_hf_session   = None


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


def get_hf_session():
    global _hf_session
    if _hf_session is None:
        _hf_session = requests.Session()
    return _hf_session


def get_hallucination_model():
    global _hallucination_model
    if _hallucination_model is None:
        if HF_API_TOKEN:
            os.environ.setdefault("HF_TOKEN", HF_API_TOKEN)
        _hallucination_model = CrossEncoder(
            HALLUCINATION_MODEL_ID,
            device="cpu",
            token=HF_API_TOKEN or None,
        )
        # Log the label mapping so we know which index maps to which label
        try:
            id2label = _hallucination_model.model.config.id2label
            logging.info("CrossEncoder id2label mapping: %s", id2label)
        except Exception:
            pass
    return _hallucination_model


def _get_label_indices(model):
    """Return (contradiction_idx, entailment_idx, neutral_idx) based on model's id2label.
    Falls back to (0, 1, 2) if mapping cannot be determined."""
    try:
        id2label = model.model.config.id2label
        label_to_idx = {v.lower(): int(k) for k, v in id2label.items()}
        con_idx = next((label_to_idx[k] for k in label_to_idx if 'contra' in k or k.endswith('_0')), None)
        ent_idx = next((label_to_idx[k] for k in label_to_idx if 'entail' in k or k.endswith('_1')), None)
        neu_idx = next((label_to_idx[k] for k in label_to_idx if 'neutral' in k or k.endswith('_2')), None)
        if con_idx is not None and ent_idx is not None:
            return con_idx, ent_idx, (neu_idx if neu_idx is not None else -1)
    except Exception:
        pass
    return 0, 1, 2  # safe default


# ── prompt templates ──────────────────────────────────────────────────────────
LLM_PROMPT = """You are a helpful AI assistant named Llama. Answer the user's question directly and concisely.

Rules:
- Be informative and accurate
- Keep responses clear and well-structured
- Use 4-8 sentences for most answers
- Do not include any URLs or source citations

Question: {question}

Answer:"""

RAG_PROMPT = """You are a fact-extraction AI that corrects LLM hallucinations using web-scraped context.

Your job:
1. Read the CONTEXT carefully — it contains scraped web data that is more recent and accurate than the LLM.
2. Extract the direct, specific answer to the QUESTION from the context.
3. State the answer CONFIDENTLY and CLEARLY. Do NOT say "the context doesn't mention" if the answer is present.
4. Provide a DETAILED response (5-8 sentences) including:
   - The direct answer (name, number, fact)
   - Relevant background information (dates, roles, history)
   - Any additional context that helps the user understand the topic fully
5. Do not add URLs. Write in clear, informative prose.

If the context genuinely does not contain the answer, say: "Based on available web sources, I could not find a definitive answer."

Question: {question}

Context:
{context}

Detailed Fact-corrected Answer:"""



# ── helpers ───────────────────────────────────────────────────────────────────


MYSQL_API_URL = os.getenv("MYSQL_API_URL", "http://localhost:3001")


def _save_to_mysql(**kwargs):
    """POST chat result to the Node.js/MySQL API in a background thread (non-blocking)."""
    def _post():
        try:
            requests.post(f"{MYSQL_API_URL}/api/save", json=kwargs, timeout=5)
        except Exception as exc:
            logging.warning("MySQL save failed (non-fatal): %s", exc)
    threading.Thread(target=_post, daemon=True).start()


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
    for url_info in urls[:2]:  # Max 2 URLs for speed
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


def _extract_three_way_scores(raw_json):
    """Normalize HF inference response into contradiction/entailment/neutral scores."""
    if isinstance(raw_json, dict) and raw_json.get("error"):
        raise RuntimeError(raw_json.get("error"))

    rows = raw_json
    if isinstance(rows, list) and rows and isinstance(rows[0], list):
        rows = rows[0]
    if not isinstance(rows, list):
        raise RuntimeError("Unexpected verifier response format")

    contradiction = 0.0
    entailment = 0.0
    neutral = 0.0

    for item in rows:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        if "contradiction" in label or label.endswith("_0"):
            contradiction = max(contradiction, score)
        elif "entailment" in label or label.endswith("_1"):
            entailment = max(entailment, score)
        elif "neutral" in label or label.endswith("_2"):
            neutral = max(neutral, score)

    total = contradiction + entailment + neutral
    if total <= 0:
        neutral = 1.0
        total = 1.0

    return contradiction / total, entailment / total, neutral / total


def verify_with_hf_api(premise: str, hypothesis: str):
    """Run NLI verification via Hugging Face hosted inference endpoint."""
    url = f"https://router.huggingface.co/hf-inference/models/{HALLUCINATION_MODEL_ID}"
    headers = {"Content-Type": "application/json"}
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is required for hosted verifier mode")
    headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    payload = {
        "inputs": {
            "text": premise[:2500],
            "text_pair": hypothesis[:800],
        },
        "options": {
            "wait_for_model": True,
        },
    }

    session = get_hf_session()
    response = session.post(url, headers=headers, json=payload, timeout=HF_INFERENCE_TIMEOUT_SEC)
    response.raise_for_status()
    return _extract_three_way_scores(response.json())


def detect_hallucination_fallback(answer: str, retrieved_docs, emb_model):
    """Fallback detector using embedding support if hosted verifier is unavailable."""
    sentences = split_into_sentences(answer)
    if not sentences:
        return {
            "overall_confidence": 0.0,
            "is_hallucinated": True,
            "classification": "UNKNOWN",
            "sentence_scores": [],
            "metadata": {"mode": "embedding_fallback", "model_id": HALLUCINATION_MODEL_ID},
        }

    ctx_embs = []
    for doc in retrieved_docs:
        try:
            ctx_embs.append(embed_text(doc.page_content, emb_model))
        except Exception:
            continue

    if not ctx_embs:
        return {
            "overall_confidence": 0.0,
            "is_hallucinated": True,
            "classification": "NO CONTEXT",
            "sentence_scores": [],
            "metadata": {"mode": "embedding_fallback", "model_id": HALLUCINATION_MODEL_ID},
        }

    scores = []
    for sentence in sentences:
        try:
            s_emb = embed_text(sentence, emb_model)
            mx = max(calc_cosine(s_emb, ce) for ce in ctx_embs)
            pct = mx * 100.0
            if pct >= 75:
                status = "NOT HALLUCINATED"
            elif pct >= 50:
                status = "SLIGHTLY HALLUCINATED"
            elif pct >= 30:
                status = "MODERATELY HALLUCINATED"
            else:
                status = "HALLUCINATED"
            scores.append({
                "text": sentence[:100] + ("..." if len(sentence) > 100 else ""),
                "score": mx,
                "score_pct": pct,
                "status": status,
            })
        except Exception:
            continue

    overall = float(np.mean([s["score_pct"] for s in scores])) if scores else 0.0
    return {
        "overall_confidence": overall,
        "is_hallucinated": overall < HALLUCINATION_THRESHOLD,
        "classification": "HIGHLY HALLUCINATED" if overall < HALLUCINATION_THRESHOLD else "NOT HALLUCINATED",
        "sentence_scores": scores,
        "metadata": {"mode": "embedding_fallback", "model_id": HALLUCINATION_MODEL_ID},
    }


def _score_sentence(contradiction: float, entailment: float) -> tuple:
    """
    Map NLI scores to a (confidence_pct, status) pair.

    Formula: confidence = (1 - contradiction) + entailment * 0.15
      - contradiction=1, entailment=0  →   0%  HALLUCINATED
      - contradiction=0, entailment=0  → 100%  not contradicted (neutral context)
      - contradiction=0, entailment=1  → 115%  clamped to 100%  strongly supported
    """
    pct = max(0.0, min(1.0, (1.0 - contradiction) + entailment * 0.15)) * 100.0
    if pct >= 75:
        status = "NOT HALLUCINATED"
    elif pct >= 50:
        status = "SLIGHTLY HALLUCINATED"
    elif pct >= 30:
        status = "MODERATELY HALLUCINATED"
    else:
        status = "HALLUCINATED"
    return pct, status


def detect_hallucination(question: str, answer: str, retrieved_docs, emb_model,
                         doc_scores: list = None):
    """
    Detect hallucinations using three signals:
      1. NLI contradiction  — did web context actively contradict the answer?
      2. NLI entailment ratio — how much of the answer does relevant context support?
      3. Pinecone relevance score — is the retrieved context actually on-topic?

    Returns a dict with overall_confidence (0-100), is_hallucinated (bool),
    classification string, and per-sentence scores.
    """
    sentences = split_into_sentences(answer)
    if not sentences:
        sentences = [answer.strip()] if answer.strip() else []

    if not sentences:
        return {"overall_confidence": 0.0, "is_hallucinated": True, "classification": "UNKNOWN",
                "sentence_scores": [], "metadata": {"processing_time": 0, "model_id": HALLUCINATION_MODEL_ID}}

    context_chunks = [doc.page_content.strip() for doc in retrieved_docs if doc.page_content.strip()]
    if not context_chunks:
        return {"overall_confidence": 0.0, "is_hallucinated": True, "classification": "NO CONTEXT",
                "sentence_scores": [], "metadata": {"processing_time": 0, "model_id": HALLUCINATION_MODEL_ID}}

    # Build a compact premise once for hosted fallback.
    context_premise = "\n\n".join(context_chunks[:3])
    sentence_scores = []

    # Primary path: local cross-encoder verifier.
    try:
        model = get_hallucination_model()
        con_idx, ent_idx, neu_idx = _get_label_indices(model)
        logging.info("Label indices — contradiction:%d entailment:%d neutral:%d", con_idx, ent_idx, neu_idx)
        for sentence in sentences:
            # Pure NLI format: premise = context chunk, hypothesis = answer sentence.
            hypothesis = sentence[:800]
            pairs = [(chunk[:2500], hypothesis) for chunk in context_chunks[:8]]
            scores = model.predict(pairs, convert_to_numpy=True, apply_softmax=True)

            if scores.ndim == 1:
                scores = np.expand_dims(scores, 0)

            # ── Best-pair scoring (label-aware) ──────────────────────────────
            # Pick the chunk that gives the HIGHEST entailment score (most supportive).
            n_labels = scores.shape[-1]
            if n_labels >= 2:
                _ent_col = min(ent_idx, n_labels - 1)
                _con_col = min(con_idx, n_labels - 1)
                best_idx      = int(np.argmax(scores[:, _ent_col]))
                entailment    = float(scores[best_idx, _ent_col])
                contradiction = float(scores[best_idx, _con_col])
                if neu_idx >= 0 and neu_idx < n_labels:
                    neutral = float(scores[best_idx, neu_idx])
                else:
                    neutral = max(0.0, 1.0 - entailment - contradiction)
            else:
                entailment    = float(np.max(scores))
                contradiction = max(0.0, 1.0 - entailment)
                neutral       = 0.0

            # Sentence confidence = how NOT contradicted the sentence is,
            # with a small bonus for strong entailment.
            sentence_confidence, status = _score_sentence(contradiction, entailment)

            sentence_scores.append({
                "text":      sentence[:100] + ("..." if len(sentence) > 100 else ""),
                "score":     sentence_confidence / 100.0,
                "score_pct": sentence_confidence,
                "status":    status,
                "detail":    {"contradiction": contradiction, "entailment": entailment,
                              "neutral": neutral, "hallucination_score": contradiction},
            })
    except Exception as _ce:
        logging.warning("CrossEncoder primary path failed: %s", _ce, exc_info=True)
        sentence_scores = []

    # Secondary path: hosted HF inference if local load/predict fails.
    if not sentence_scores:
        try:
            for sentence in sentences:
                # Plain NLI hypothesis — no Q/A prefix
                hypothesis = sentence[:800]
                contradiction, entailment, neutral = verify_with_hf_api(context_premise, hypothesis)

                sentence_confidence, status = _score_sentence(contradiction, entailment)

                sentence_scores.append({
                    "text": sentence[:100] + ("..." if len(sentence) > 100 else ""),
                    "score": sentence_confidence / 100.0,
                    "score_pct": sentence_confidence,
                    "status": status,
                    "detail": {
                        "contradiction": contradiction,
                        "entailment": entailment,
                        "neutral": neutral,
                        "hallucination_score": sentence_hallucination,
                    },
                })
        except Exception as exc:
            fallback = detect_hallucination_fallback(answer, retrieved_docs, emb_model)
            fallback_meta = fallback.get("metadata", {})
            fallback_meta["hf_error"] = str(exc)
            fallback["metadata"] = fallback_meta
            return fallback

    # ── Context relevance (Pinecone cosine scores) ───────────────────────────
    RELEVANCE_THRESHOLD = 0.40
    if doc_scores and len(doc_scores) > 0:
        context_relevance = float(np.mean([float(s) for s in doc_scores]))
    else:
        context_relevance = 0.30   # conservative default if no scores

    context_is_relevant = context_relevance >= RELEVANCE_THRESHOLD
    logging.info("Context relevance: %.3f  relevant=%s", context_relevance, context_is_relevant)

    # ── NLI aggregates ────────────────────────────────────────────────────────
    details = [s["detail"] for s in sentence_scores if isinstance(s.get("detail"), dict)]
    mean_contradiction = float(np.mean([d.get("contradiction", 0) for d in details])) if details else 0.0
    mean_entailment    = float(np.mean([d.get("entailment",    0) for d in details])) if details else 0.0
    entailed_count     = sum(1 for d in details if d.get("entailment", 0) > 0.5)
    entailment_ratio   = entailed_count / len(details) if details else 0.0

    # ── Unified confidence score ──────────────────────────────────────────────
    # This is the SINGLE number that drives the ring, verdict banner, classification,
    # and is_hallucinated — all derived from the same value so they are always consistent.
    #
    # Formula (context-relevance-weighted blend):
    #
    #   When context IS relevant (Pinecone score >= 0.40):
    #     Both contradiction and entailment matter equally.
    #     hallucination_blend = contradiction * 0.5
    #                         + (1 - min(entailment_ratio * 2, 1)) * 0.5
    #
    #     Examples:
    #       con=0, ent_ratio=0   → blend=0+0.5=0.50 → conf=50%  MODERATELY HALLUCINATED ✓
    #       con=0, ent_ratio=0.5 → blend=0+0.0=0.00 → conf=100% NOT HALLUCINATED ✓
    #       con=0.5, ent_ratio=0 → blend=0.25+0.5=0.75 → conf=25% HIGHLY HALLUCINATED ✓
    #       con=0, ent_ratio=0.3 → blend=0+0.20=0.20 → conf=80%  SLIGHTLY HALLUCINATED ✓
    #
    #   When context is NOT relevant (off-topic web pages):
    #     Only contradiction matters — lack of entailment is not informative.
    #     hallucination_blend = contradiction
    #
    #     Example (Taj Mahal correct answer, burning-brown articles retrieved):
    #       con=0, ent_ratio=0  → blend=0 → conf=100% NOT HALLUCINATED ✓
    #
    if context_is_relevant:
        hallucination_blend = (
            mean_contradiction * 0.5
            + (1.0 - min(entailment_ratio * 2.0, 1.0)) * 0.5
        )
    else:
        hallucination_blend = mean_contradiction   # only contradiction is reliable signal

    overall_confidence = max(0.0, (1.0 - hallucination_blend) * 100.0)

    logging.info(
        "Unified score — con=%.3f ent_ratio=%.2f relevant=%s "
        "blend=%.3f → confidence=%.1f%%",
        mean_contradiction, entailment_ratio, context_is_relevant,
        hallucination_blend, overall_confidence
    )

    # ── Classification ────────────────────────────────────────────────────────
    # hallPct = 100 - overall_confidence  (used by the UI ring/meter)
    # UI thresholds:  ≤25 → green (NOT HALL.)  ≤45 → orange-slight  ≤65 → orange-mod  >65 → red
    if overall_confidence >= 75:
        classification = "NOT HALLUCINATED"
    elif overall_confidence >= 55:
        classification = "SLIGHTLY HALLUCINATED"
    elif overall_confidence >= 35:
        classification = "MODERATELY HALLUCINATED"
    else:
        classification = "HIGHLY HALLUCINATED"

    # ── is_hallucinated flag ──────────────────────────────────────────────────
    # Derived directly from overall_confidence so UI and decision are always in sync.
    # Threshold = 65: anything below "NOT HALLUCINATED" band triggers RAG correction.
    is_hallucinated = overall_confidence < 65

    logging.info(
        "Final — classification=%s confidence=%.1f%% is_hallucinated=%s",
        classification, overall_confidence, is_hallucinated
    )

    return {
        "overall_confidence": overall_confidence,
        "is_hallucinated": is_hallucinated,
        "classification": classification,
        "sentence_scores": sentence_scores,
        "metadata": {
            "sentence_count": len(sentences),
            "chunks_analyzed": len(context_chunks),
            "model_id": HALLUCINATION_MODEL_ID,
            "pair_count": len(sentences),
            "mode": "local_cross_encoder",
            "mean_contradiction": mean_contradiction,
            "mean_entailment": mean_entailment,
            "entailment_ratio": entailment_ratio,
            "context_relevance": context_relevance,
            "context_is_relevant": context_is_relevant,
        },
    }


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
app = Flask(__name__, static_folder="../frontend", static_url_path="/static")
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


_PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "public")

@app.route("/public/<path:filename>")
def serve_public(filename):
    """Serve files from frontend/public/ (favicons, web manifest, etc.)."""
    return send_from_directory(_PUBLIC_DIR, filename)


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
                doc_scores = [float(sc) for _, sc in docs_ws]  # Pinecone relevance scores
                result = detect_hallucination(question, llm_answer, docs, emb,
                                             doc_scores=doc_scores)
                tok_an = {
                    "support_pct": None,
                    "matched": [],
                    "unmatched": [],
                    "model": HALLUCINATION_MODEL_ID,
                }
            else:
                result = {"overall_confidence": 20, "is_hallucinated": True,
                          "classification": "NO CONTEXT AVAILABLE", "sentence_scores": [],
                          "metadata": {"processing_time": 0, "model_id": HALLUCINATION_MODEL_ID}}
                tok_an = {"support_pct": None, "matched": [], "unmatched": [], "model": HALLUCINATION_MODEL_ID}
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
            rag_answer = None  # set only when RAG is used
            yield event("step", {"step": 4, "label": "Deciding…", "status": "running"})
            if result["is_hallucinated"] and context:
                yield event("step", {"step": 4,
                                     "label": "Hallucination detected → generating RAG response…",
                                     "status": "running"})
                # Raise context cap to 6000 chars — ensures relevant content
                # (e.g. "Keir Starmer" buried deeper in the page) isn't cut off
                rag_context = context[:6000]
                rag_answer  = generate_rag_answer(question, rag_context)
                sources     = [
                    {"url": u["url"], "title": u.get("title", u["url"])[:80]}
                    for u in url_results[:3]
                    if isinstance(u, dict)
                ]
                # Send RAG response immediately — no second LLM call for summary
                yield event("rag", {"text": rag_answer, "sources": sources,
                                    "summary": "", "is_hallucinated": True,
                                    "context_text": rag_context,
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

            # ── Save to MySQL (non-blocking) ──────────────────────────────
            rag_text = rag_answer if result.get("is_hallucinated") and context else None
            _save_to_mysql(
                query=question,
                llm_response=llm_answer,
                rag_response=rag_text,
                is_hallucinated=result["is_hallucinated"],
                hallucination_score=round(result["overall_confidence"], 2),
                classification=result["classification"],
                sentence_count=len(result.get("sentence_scores", [])),
                sources_count=len(url_results),
                sources=sources,
                model_id=HALLUCINATION_MODEL_ID,
            )
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
