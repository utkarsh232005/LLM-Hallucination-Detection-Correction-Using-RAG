"""
Hallucination Detection RAG Application

This application provides a simple LLM chat interface (right side) with automatic
hallucination detection. When the LLM output is detected as hallucinated, the RAG
system provides corrected answers with source links.

Flow:
1. User asks question to LLM (Llama 3.2)
2. LLM generates output
3. Output is analyzed against Pinecone VDB embeddings
4. If hallucinated → RAG provides corrected answer with sources
5. If not hallucinated → LLM answer is used as-is
"""

import os
import time
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import streamlit as st

from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

try:
    from langchain_pinecone import PineconeVectorStore
    HAS_LANGCHAIN_PINECONE = True
except ImportError:
    from langchain_community.vectorstores import Pinecone as PineconeVectorStore
    HAS_LANGCHAIN_PINECONE = False
from pinecone import Pinecone

from config import (
    EMBEDDINGS_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RAG_LLM_MODEL,
    RAG_TEMPERATURE,
)


# ==================== TYPEWRITER SETTINGS ====================

TYPEWRITER_DELAY = 0.008  # Delay between characters (seconds)
FAST_TYPEWRITER_DELAY = 0.004  # Faster for longer texts


# ==================== CONFIGURATION ==

SERPAPI_API_KEY = "3b5e8c37d4769cf12f42df01df5baa17f207836ee859d08f62d66607cd06cfb4"
PINECONE_API_KEY = "pcsk_4EeaiW_PxmXpizoWmimbi8q9Cn3NTEMQJK9Xz14epbTWVwJGyWbyRp6cQy5BeEuE3AP9ws"

os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME = "rag-embedds"
PINECONE_NAMESPACE = "web-rag-records"

# Hallucination Detection Thresholds
HALLUCINATION_THRESHOLD = 35.0  # Below this confidence = hallucinated
MILD_HALLUCINATION_THRESHOLD = 55.0  # Between 35-55 = mild hallucination

# Optimized Chunking Parameters (for better accuracy)
CHUNK_SIZE = 2000  # Larger chunks for more context
CHUNK_OVERLAP = 300  # More overlap to preserve context
MAX_CHUNKS = 80  # More chunks for comprehensive coverage
MAX_URLS = 10  # More sources for better grounding
RETRIEVAL_K = 12  # Retrieve more relevant chunks
CONTENT_PREVIEW_LENGTH = 500  # Longer content previews


# ==================== MODEL INITIALIZATION ====================

@st.cache_resource
def get_embeddings():
    """Initialize embeddings model."""
    return OllamaEmbeddings(model=EMBEDDINGS_MODEL)


@st.cache_resource
def get_llm():
    """Initialize LLM for chat."""
    return OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


@st.cache_resource
def get_llm_rag():
    """Initialize LLM for RAG (lower temperature for accuracy)."""
    return OllamaLLM(model=RAG_LLM_MODEL, temperature=RAG_TEMPERATURE)


@st.cache_resource
def get_pinecone_client():
    """Initialize Pinecone client."""
    return Pinecone(api_key=PINECONE_API_KEY)


def clear_namespace(namespace):
    """Clear all vectors from the namespace before new query."""
    try:
        pc = get_pinecone_client()
        index = pc.Index(INDEX_NAME)
        # Delete all vectors in the namespace
        index.delete(delete_all=True, namespace=namespace)
        time.sleep(0.5)  # Wait for deletion to complete
    except Exception:
        pass  # Ignore errors if namespace doesn't exist


@st.cache_resource
def get_vector_store(_embeddings):
    """Initialize Pinecone vector store."""
    if HAS_LANGCHAIN_PINECONE:
        return PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=_embeddings
        )
    else:
        return PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=_embeddings,
            text_key="text"
        )


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


# ==================== WEB SCRAPING FUNCTIONS ====================

def search_web(query, max_results=MAX_URLS):
    """Search web using SerpAPI and return URLs."""
    try:
        search = SerpAPIWrapper()
        results = search.results(query)
        urls = []
        
        if "organic_results" in results:
            for r in results["organic_results"][:max_results]:
                urls.append({
                    "url": r["link"],
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", "")
                })
        
        return urls
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []


def load_pages(urls):
    """Load web pages with timeout for faster processing."""
    documents = []
    
    for url_info in urls[:3]:  # Limit to 3 URLs for speed
        try:
            url = url_info["url"] if isinstance(url_info, dict) else url_info
            loader = WebBaseLoader(url, requests_kwargs={"timeout": 5})  # 5 second timeout
            docs = loader.load()
            
            # Add title to metadata
            for doc in docs:
                if isinstance(url_info, dict):
                    doc.metadata["title"] = url_info.get("title", "")
                    doc.metadata["snippet"] = url_info.get("snippet", "")
            
            documents.extend(docs)
            
            # Early exit if we have enough content
            if len(documents) >= 2:
                break
        except Exception:
            continue
    
    return documents


def split_text(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, max_chunks=MAX_CHUNKS):
    """Split documents into chunks optimized for embedding matching."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    return chunks[:max_chunks]


def index_documents(docs, vector_store, namespace):
    """Index document chunks into Pinecone."""
    texts = []
    metadatas = []
    ids = []
    timestamp = int(time.time())
    
    for i, doc in enumerate(docs):
        texts.append(doc.page_content)
        metadatas.append({
            "source": doc.metadata.get("source", ""),
            "title": doc.metadata.get("title", ""),
            "chunk_text": doc.page_content[:500],
            "created_at": timestamp
        })
        ids.append(f"chunk-{timestamp}-{i}-{uuid.uuid4().hex[:8]}")
    
    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        namespace=namespace
    )
    
    return len(texts)


def retrieve_documents(query, vector_store, namespace, k=RETRIEVAL_K, relevance_threshold=0.30):
    """Retrieve relevant documents from Pinecone."""
    docs_with_scores = []
    
    try:
        raw_results = vector_store.similarity_search_with_relevance_scores(
            query,
            k=k,
            namespace=namespace
        )
        
        for doc, score in raw_results:
            if score >= relevance_threshold:
                docs_with_scores.append((doc, float(score)))
        
        # Fallback: always return something
        if not docs_with_scores and raw_results:
            docs_with_scores = [(doc, float(score)) for doc, score in raw_results[:k]]
            
    except Exception:
        raw_results = vector_store.similarity_search(query, k=k, namespace=namespace)
        docs_with_scores = [(doc, 1.0) for doc in raw_results]
    
    return docs_with_scores


# ==================== HALLUCINATION DETECTION ====================

def split_into_sentences(text):
    """Split text into sentences for analysis."""
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


@st.cache_data(ttl=3600)
def cached_embed_text(text, _embeddings_model):
    """Cache embeddings for performance."""
    return _embeddings_model.embed_query(text)


def embed_text(text, embeddings_model):
    """Convert text to embedding vector."""
    embedding = cached_embed_text(text, embeddings_model)
    return np.array(embedding)


def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return max(0.0, min(1.0, similarity))


def tokenize_text(text, min_len=3):
    """Extract meaningful tokens from text."""
    stop_words = {
        "the", "and", "for", "that", "this", "with", "from", "into", "have", "has", "had",
        "are", "was", "were", "will", "would", "shall", "could", "should", "about", "your",
        "their", "there", "which", "when", "where", "what", "who", "why", "how", "does",
        "did", "can", "also", "than", "then", "them", "they", "its", "our", "you", "been"
    }
    raw_tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-']+\b", text.lower())
    return [token for token in raw_tokens if len(token) >= min_len and token not in stop_words]


def token_support_analysis(answer_text, docs):
    """Analyze token-level support from context."""
    answer_tokens = tokenize_text(answer_text)
    context_tokens = set()
    
    for doc in docs:
        context_tokens.update(tokenize_text(doc.page_content))
    
    unique_answer_tokens = list(dict.fromkeys(answer_tokens))
    
    if not unique_answer_tokens:
        return {"support_pct": 100.0, "matched": [], "unmatched": []}
    
    matched = [t for t in unique_answer_tokens if t in context_tokens]
    unmatched = [t for t in unique_answer_tokens if t not in context_tokens]
    
    support_pct = (len(matched) / len(unique_answer_tokens)) * 100.0
    
    return {
        "support_pct": support_pct,
        "matched": matched[:15],
        "unmatched": unmatched[:15]
    }


def detect_hallucination(answer, retrieved_docs, embeddings_model):
    """
    Analyze LLM answer for hallucination by comparing with retrieved context.
    
    Returns a detailed report with confidence scores per sentence.
    """
    start_time = time.time()
    sentences = split_into_sentences(answer)
    
    if len(sentences) == 0:
        return {
            'overall_confidence': 0,
            'is_hallucinated': True,
            'classification': 'UNKNOWN',
            'sentence_scores': [],
            'metadata': {'processing_time': 0}
        }
    
    # Embed all context chunks
    context_embeddings = []
    for doc in retrieved_docs:
        try:
            context_emb = embed_text(doc.page_content, embeddings_model)
            context_embeddings.append(context_emb)
        except Exception:
            continue
    
    if not context_embeddings:
        return {
            'overall_confidence': 0,
            'is_hallucinated': True,
            'classification': 'NO CONTEXT',
            'sentence_scores': [],
            'metadata': {'processing_time': time.time() - start_time}
        }
    
    # Analyze each sentence
    sentence_scores = []
    
    for sentence in sentences:
        try:
            sentence_emb = embed_text(sentence, embeddings_model)
            
            # Find max similarity with any context chunk
            max_similarity = 0.0
            for context_emb in context_embeddings:
                similarity = calculate_cosine_similarity(sentence_emb, context_emb)
                max_similarity = max(max_similarity, similarity)
            
            # Classify sentence
            score_pct = max_similarity * 100
            if score_pct >= 75:
                status = "✅ NOT HALLUCINATED"
            elif score_pct >= 50:
                status = "⚠️ SLIGHTLY HALLUCINATED"
            elif score_pct >= 30:
                status = "⚡ MODERATELY HALLUCINATED"
            else:
                status = "❌ HALLUCINATED"
            
            sentence_scores.append({
                'text': sentence[:100] + "..." if len(sentence) > 100 else sentence,
                'score': max_similarity,
                'score_pct': score_pct,
                'status': status
            })
        except Exception:
            continue
    
    # Calculate overall confidence
    if sentence_scores:
        avg_similarity = np.mean([s['score'] for s in sentence_scores])
        overall_confidence = avg_similarity * 100
    else:
        overall_confidence = 0
    
    # Determine if hallucinated
    is_hallucinated = overall_confidence < HALLUCINATION_THRESHOLD
    
    # Classification - Using Hallucination terminology
    if overall_confidence >= 75:
        classification = "NOT HALLUCINATED"
    elif overall_confidence >= MILD_HALLUCINATION_THRESHOLD:
        classification = "SLIGHTLY HALLUCINATED"
    elif overall_confidence >= HALLUCINATION_THRESHOLD:
        classification = "MODERATELY HALLUCINATED"
    else:
        classification = "HIGHLY HALLUCINATED"
    
    return {
        'overall_confidence': overall_confidence,
        'is_hallucinated': is_hallucinated,
        'classification': classification,
        'sentence_scores': sentence_scores,
        'metadata': {
            'sentence_count': len(sentences),
            'chunks_analyzed': len(context_embeddings),
            'processing_time': time.time() - start_time
        }
    }


# ==================== RAG FUNCTIONS ====================

def format_context(docs_with_scores):
    """Format retrieved documents as context for RAG."""
    if not docs_with_scores:
        return ""
    
    blocks = []
    for idx, (doc, score) in enumerate(docs_with_scores, start=1):
        content = re.sub(r"\s+", " ", doc.page_content.strip())
        blocks.append(content)
    
    return "\n\n---\n\n".join(blocks)


def generate_llm_answer(question, llm):
    """Generate answer using only LLM (no RAG)."""
    prompt = ChatPromptTemplate.from_template(LLM_PROMPT_TEMPLATE)
    chain = prompt | llm
    return chain.invoke({"question": question})


def generate_rag_answer(question, context, llm):
    """Generate answer using RAG context."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = prompt | llm
    return chain.invoke({"question": question, "context": context})


def summarize_rag_context(question, context, llm):
    """Generate a detailed summary paragraph from RAG context."""
    summary_template = """Based on the web-scraped data below, provide a comprehensive summary with all key facts, numbers, and details that answer the question. Be thorough and include all relevant information.

Question: {question}

Web Data:
{context}

Detailed Summary (include all key facts and numbers):"""
    
    prompt = ChatPromptTemplate.from_template(summary_template)
    chain = prompt | llm
    try:
        summary = chain.invoke({"question": question, "context": context[:4000]})
        return str(summary).strip()
    except Exception:
        return None


def sanitize_answer(answer):
    """Clean up LLM answer text."""
    text = str(answer)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collect_rag_context(question, vector_store, namespace):
    """Collect context from web scraping and Pinecone."""
    # Clear previous query's data from namespace
    clear_namespace(namespace)
    
    # Search web
    url_results = search_web(question, max_results=MAX_URLS)
    
    if not url_results:
        return [], "", []
    
    # Load pages
    documents = load_pages(url_results)
    
    if not documents:
        return [], "", url_results
    
    # Split into chunks
    chunks = split_text(documents)
    
    if not chunks:
        return [], "", url_results
    
    # Index in Pinecone
    index_documents(chunks, vector_store, namespace)
    
    # Retrieve relevant chunks
    docs_with_scores = retrieve_documents(question, vector_store, namespace)
    
    # Format context
    context = format_context(docs_with_scores)
    docs = [item[0] for item in docs_with_scores]
    
    return docs, context, url_results


# ==================== TYPEWRITER EFFECT ====================

def typewriter_effect(placeholder, text, delay=TYPEWRITER_DELAY, prefix=""):
    """
    Display text with typewriter effect character by character.
    
    Args:
        placeholder: Streamlit placeholder to update
        text: Text to display
        delay: Delay between characters
        prefix: Optional prefix before the text (e.g., emoji badge)
    """
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(prefix + displayed + "▌")  # Cursor effect
        time.sleep(delay)
    
    # Final render without cursor
    placeholder.markdown(prefix + displayed)
    return displayed


def typewriter_html(placeholder, text, bg_color, badge, delay=TYPEWRITER_DELAY):
    """
    Display text with typewriter effect in a styled chat bubble.
    
    Args:
        placeholder: Streamlit placeholder
        text: Text to display
        bg_color: Background color for bubble
        badge: Badge text (e.g., '🦙 Llama')
        delay: Delay between characters
    """
    displayed = ""
    
    # Adjust delay based on text length
    if len(text) > 200:
        delay = FAST_TYPEWRITER_DELAY
    
    for char in text:
        displayed += char
        html = f"""
        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
            <div style="background-color: {bg_color}; color: white; padding: 12px 18px; 
                        border-radius: 18px 18px 18px 5px; max-width: 80%;">
                <small style="opacity: 0.7;">{badge}</small><br>
                {displayed}<span style="animation: blink 1s infinite;">▌</span>
            </div>
        </div>
        """
        placeholder.markdown(html, unsafe_allow_html=True)
        time.sleep(delay)
    
    # Final render without cursor
    final_html = f"""
    <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
        <div style="background-color: {bg_color}; color: white; padding: 12px 18px; 
                    border-radius: 18px 18px 18px 5px; max-width: 80%;">
            <small style="opacity: 0.7;">{badge}</small><br>
            {displayed}
        </div>
    </div>
    """
    placeholder.markdown(final_html, unsafe_allow_html=True)
    return displayed


def render_step_indicator(placeholder, step_num, step_text, status="running"):
    """
    Render a step indicator with status.
    
    Args:
        placeholder: Streamlit placeholder
        step_num: Step number
        step_text: Description of the step
        status: 'running', 'complete', 'pending'
    """
    if status == "running":
        icon = "🔄"
        color = "#ff9800"
    elif status == "complete":
        icon = "✅"
        color = "#00c853"
    else:
        icon = "⏳"
        color = "#888888"
    
    html = f"""
    <div style="padding: 8px 12px; margin: 5px 0; border-left: 3px solid {color}; 
                background-color: rgba(255,255,255,0.05); border-radius: 0 5px 5px 0;">
        <span style="color: {color};">{icon}</span> 
        <strong>Step {step_num}:</strong> {step_text}
    </div>
    """
    placeholder.markdown(html, unsafe_allow_html=True)


def render_shimmer_loading(placeholder, label_text):
    """Render shimmering status text while a response is being generated."""
    placeholder.markdown(
        f"""
        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
            <div style="background-color: #1f1f1f; color: #ddd; padding: 12px 18px;
                        border-radius: 18px 18px 18px 5px; max-width: 80%;">
                <small style="opacity: 0.7;">{label_text}</small><br>
                <span class="shimmer-text">Generating response...</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================== UI COMPONENTS ====================

def render_hallucination_meter(hallucination_pct):
    """Render visual hallucination meter (higher = worse)."""
    # Invert colors - red for high hallucination, green for low
    if hallucination_pct <= 25:
        color = "#00c853"  # green - low hallucination
    elif hallucination_pct <= 45:
        color = "#ff9800"  # orange - moderate
    elif hallucination_pct <= 65:
        color = "#ffc107"  # yellow - high
    else:
        color = "#f44336"  # red - very high hallucination
    
    st.markdown(f"""
        <div style="background-color: #2d2d2d; border-radius: 10px; padding: 3px; margin: 10px 0;">
            <div style="width: {hallucination_pct}%; background-color: {color}; border-radius: 8px; 
                        height: 25px; display: flex; align-items: center; justify-content: flex-end;
                        padding-right: 10px; color: white; font-weight: bold; font-size: 14px;
                        transition: width 0.5s ease; min-width: 50px;">
                {hallucination_pct:.0f}%
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_hallucination_report(result, token_analysis):
    """Render the hallucination detection report panel."""
    confidence = result['overall_confidence']
    hallucination_pct = 100 - confidence  # Invert to show hallucination %
    classification = result['classification']
    
    # Header
    st.markdown("### 🎯 HALLUCINATION DETECTION REPORT")
    st.markdown("---")
    
    # Overall metrics - Show hallucination percentage
    emoji = "✅" if hallucination_pct <= 25 else ("⚠️" if hallucination_pct <= 45 else ("⚡" if hallucination_pct <= 65 else "❌"))
    
    st.markdown(f"**Hallucination Level:** {hallucination_pct:.0f}% {emoji}")
    st.markdown(f"**Status:** {classification}")
    
    # Hallucination meter
    st.markdown("**Hallucination Meter:**")
    render_hallucination_meter(hallucination_pct)
    
    st.markdown("---")
    
    # Detailed sentence analysis
    st.markdown("### 📊 Sentence Analysis:")
    
    for i, item in enumerate(result['sentence_scores'][:6], 1):
        score_pct = item['score_pct']
        hall_pct = 100 - score_pct  # Hallucination percentage
        status = item['status']
        text = item['text']
        
        # Color based on hallucination level
        if hall_pct <= 25:
            color = "🟢"
        elif hall_pct <= 50:
            color = "🟡"
        else:
            color = "🔴"
        
        st.markdown(f"{i}. {color} **{hall_pct:.0f}% hallucinated** - \"{text}\"")
    
    st.markdown("---")
    
    # Interpretation
    st.markdown("### 💡 Result")
    
    if hallucination_pct <= 25:
        st.success("**NOT HALLUCINATED**: The LLM answer is accurate and verified.")
    elif hallucination_pct <= 45:
        st.warning("**SLIGHTLY HALLUCINATED**: Minor inaccuracies detected. LLM answer is mostly reliable.")
    elif hallucination_pct <= 65:
        st.warning("**MODERATELY HALLUCINATED**: Several inaccuracies found. RAG correction recommended.")
    else:
        st.error("**HIGHLY HALLUCINATED**: LLM answer is unreliable. RAG system providing corrected answer.")
    
    # Unsupported tokens (simplified)
    if token_analysis['unmatched']:
        st.caption(f"⚠️ Unsupported terms: {', '.join(token_analysis['unmatched'][:8])}")


def render_rag_unit(url_results, docs, context, summarized_info=None):
    """Render the RAG unit with links and summarized paragraph."""
    st.markdown("### 📦 RAG SOURCES")
    st.markdown("---")
    
    # Source links
    st.markdown("**🔗 Sources:**")
    for i, url_info in enumerate(url_results[:3], 1):
        url = url_info["url"] if isinstance(url_info, dict) else url_info
        title = url_info.get("title", "Source") if isinstance(url_info, dict) else "Source"
        st.markdown(f"[{i}. {title[:60]}]({url})")
    
    st.markdown("---")
    
    # Summarized paragraph - dark black text
    st.markdown("**📝 RAG Context Summary:**")
    if summarized_info:
        st.markdown(f"""
        <div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; margin: 10px 0; max-height: 400px; overflow-y: auto;">
            <p style="color: #000000; font-size: 15px; line-height: 1.8; margin: 0; text-align: justify;">
                {summarized_info}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback: show full context
        if context:
            clean_context = re.sub(r'\s+', ' ', context).strip()
            st.markdown(f"""
            <div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; margin: 10px 0; max-height: 400px; overflow-y: auto;">
                <p style="color: #000000; font-size: 15px; line-height: 1.8; margin: 0; text-align: justify;">
                    {clean_context}
                </p>
            </div>
            """, unsafe_allow_html=True)


def render_chat_message(role, content, is_rag=False):
    """Render a chat message bubble."""
    if role == "user":
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
            <div style="background-color: #0066ff; color: white; padding: 12px 18px; 
                        border-radius: 18px 18px 5px 18px; max-width: 80%;">
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        badge = "🤖 RAG" if is_rag else "🦙 Llama"
        bg_color = "#2d5a2d" if is_rag else "#333333"
        
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
            <div style="background-color: {bg_color}; color: white; padding: 12px 18px; 
                        border-radius: 18px 18px 18px 5px; max-width: 80%;">
                <small style="opacity: 0.7;">{badge}</small><br>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==================== MAIN APPLICATION ====================

def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="Hallucination Detection RAG",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main .block-container {
            max-width: 1600px;
            padding: 1rem 2rem;
        }
        .stTextInput > div > div > input {
            font-size: 16px;
        }
        div[data-testid="stVerticalBlock"] > div {
            padding: 0.5rem 0;
        }
        .analysis-panel {
            background-color: #1e1e1e;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 1rem;
        }
        /* Blinking cursor animation */
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        /* Step indicator animation */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        .step-running {
            animation: pulse 1s infinite;
        }
        /* Shimmer loading text */
        .shimmer-text {
            background: linear-gradient(90deg, #9aa0a6 20%, #ffffff 50%, #9aa0a6 80%);
            background-size: 300% 100%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shimmer 1.6s linear infinite;
            font-weight: 600;
            letter-spacing: 0.2px;
        }
        @keyframes shimmer {
            0% { background-position: 300% 0; }
            100% { background-position: -300% 0; }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = None
    if "last_rag_data" not in st.session_state:
        st.session_state.last_rag_data = None
    if "show_rag" not in st.session_state:
        st.session_state.show_rag = False
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0  # 0=idle, 1=LLM, 2=Search, 3=Analyze, 4=Decision, 5=Complete
    if "step_status" not in st.session_state:
        st.session_state.step_status = "Ready"
    
    # Initialize models
    embeddings = get_embeddings()
    llm = get_llm()
    llm_rag = get_llm_rag()
    vector_store = get_vector_store(embeddings)
    
    # Title
    st.markdown("""
        <h1 style="margin-bottom: 0; text-align: center;">
            🎯 Hallucination Detection RAG System
        </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; color: #888; margin-top: 5px;'>LLM responses are automatically analyzed • RAG activates when hallucination detected</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main layout: Analysis Panel (Left) | Chat Interface (Right)
    col_analysis, col_chat = st.columns([1, 1.2], gap="large")
    
    # === RIGHT SIDE: Chat Interface ===
    with col_chat:
        st.markdown("### 💬 CHAT WITH LLM")
        
        # Chat container with messages
        chat_container = st.container(height=450)
        
        with chat_container:
            # Display existing messages as Q&A bubbles
            for msg in st.session_state.messages:
                render_chat_message(msg["role"], msg["content"], msg.get("is_rag", False))
            
            # During processing, show user's question and live response
            if st.session_state.get("processing", False):
                question = st.session_state.get("current_question", "")
                # User question already added to messages, will be displayed above
        
        # === STEP INDICATORS (between chat and input) ===
        step_container = st.container()
        
        # Input section at BOTTOM
        with st.form("chat_form", clear_on_submit=True):        
            user_input = st.text_input(
                "Ask anything...",
                placeholder="Type your question here...",
                label_visibility="collapsed"
            )
            
            col_submit, col_clear = st.columns([3, 1])
            with col_submit:
                submit = st.form_submit_button("🚀 Send", use_container_width=True)
            with col_clear:
                clear = st.form_submit_button("🗑️ Clear", use_container_width=True)
        
        if clear:
            st.session_state.messages = []
            st.session_state.last_analysis = None
            st.session_state.last_rag_data = None
            st.session_state.show_rag = False
            st.session_state.processing = False
            st.session_state.current_step = 0
            st.session_state.step_status = "Ready"
            st.rerun()
        
        if submit and user_input.strip():
            question = user_input.strip()
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Set processing flag and initial step
            st.session_state.processing = True
            st.session_state.current_question = question
            st.session_state.current_step = 1
            st.session_state.step_status = "🦙 LLM Generating..."
            st.rerun()
        
        # === PROCESSING LOGIC ===
        if st.session_state.get("processing", False):
            question = st.session_state.get("current_question", "")
            
            # Re-create step placeholders for processing
            with step_container:
                step1_placeholder = st.empty()
                step2_placeholder = st.empty()
                step3_placeholder = st.empty()
                step4_placeholder = st.empty()
                response_placeholder = st.empty()
                
                # ═══════════════════════════════════════════════════════════
                # STEP 1: Generate LLM Answer
                # ═══════════════════════════════════════════════════════════
                st.session_state.current_step = 1
                st.session_state.step_status = "🦙 LLM Generating..."
                
                render_step_indicator(step1_placeholder, 1, "Generating LLM response...", "running")
                render_step_indicator(step2_placeholder, 2, "Search knowledge base", "pending")
                render_step_indicator(step3_placeholder, 3, "Detect hallucination", "pending")
                render_step_indicator(step4_placeholder, 4, "Decision & final response", "pending")
                
                # Show shimmer while LLM is generating
                with response_placeholder.container():
                    llm_display = st.empty()
                    render_shimmer_loading(llm_display, "🦙 Llama")

                # Generate LLM answer
                llm_answer = sanitize_answer(generate_llm_answer(question, llm))
                
                # Display LLM answer directly as chat bubble
                render_step_indicator(step1_placeholder, 1, "LLM response generated ✓", "complete")
                
                with response_placeholder.container():
                    # Show as chat bubble directly (no header)
                    llm_display = st.empty()
                    typewriter_html(llm_display, llm_answer, "#333333", "🦙 Llama")
                
                time.sleep(0.5)  # Brief pause before next step
                
                # ═══════════════════════════════════════════════════════════
                # STEP 2: Collect RAG Context
                # ═══════════════════════════════════════════════════════════
                st.session_state.current_step = 2
                st.session_state.step_status = "🔍 Web Scraping & Indexing..."
                
                render_step_indicator(step2_placeholder, 2, "Searching & scraping web sources...", "running")
                
                docs, context, url_results = collect_rag_context(
                    question, vector_store, PINECONE_NAMESPACE
                )
                
                render_step_indicator(step2_placeholder, 2, f"Found {len(url_results)} sources, {len(docs)} chunks ✓", "complete")
                
                time.sleep(0.3)  # Brief pause
                
                # ═══════════════════════════════════════════════════════════
                # STEP 3: Detect Hallucination
                # ═══════════════════════════════════════════════════════════
                st.session_state.current_step = 3
                st.session_state.step_status = "🧠 Analyzing for Hallucination..."
                
                render_step_indicator(step3_placeholder, 3, "Analyzing for hallucination...", "running")
                
                if docs:
                    analysis_result = detect_hallucination(llm_answer, docs, embeddings)
                    token_analysis = token_support_analysis(llm_answer, docs)
                else:
                    analysis_result = {
                        'overall_confidence': 20,
                        'is_hallucinated': True,
                        'classification': 'NO CONTEXT AVAILABLE',
                        'sentence_scores': [],
                        'metadata': {'processing_time': 0}
                    }
                    token_analysis = {"support_pct": 0, "matched": [], "unmatched": []}
                
                confidence = analysis_result['overall_confidence']
                is_hallucinated = analysis_result['is_hallucinated']
                
                status_text = f"Hallucination: {100 - confidence:.0f}% - {'❌ HALLUCINATED' if is_hallucinated else '✅ NOT HALLUCINATED'}"
                render_step_indicator(step3_placeholder, 3, status_text, "complete")
                
                # Store analysis
                st.session_state.last_analysis = {
                    "result": analysis_result,
                    "token_analysis": token_analysis
                }
                
                time.sleep(0.5)  # Pause to show analysis result
                
                # ═══════════════════════════════════════════════════════════
                # STEP 4: Decision - Use LLM or RAG
                # ═══════════════════════════════════════════════════════════
                st.session_state.current_step = 4
                st.session_state.step_status = "⚖️ Deciding..."
                
                if is_hallucinated:
                    st.session_state.step_status = "⚡ Hallucination Detected"
                    render_step_indicator(step4_placeholder, 4, "⚡ Hallucination detected → Generating RAG response...", "running")
                    st.session_state.show_rag = True
                    
                    if context:
                        # Show shimmer while RAG is generating the corrected response
                        with response_placeholder.container():
                            rag_display = st.empty()
                            render_shimmer_loading(rag_display, "🤖 RAG")

                        # Generate RAG answer and summary
                        rag_answer = sanitize_answer(generate_rag_answer(question, context, llm_rag))
                        summarized_info = summarize_rag_context(question, context, llm_rag)
                        
                        render_step_indicator(step4_placeholder, 4, "RAG response generated ✓", "complete")
                        
                        # Display RAG answer as chat bubble
                        with response_placeholder.container():
                            # Show RAG corrected response directly
                            rag_display = st.empty()
                            typewriter_html(rag_display, rag_answer, "#2d5a2d", "🤖 RAG")
                        
                        # Store messages
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": rag_answer,
                            "is_rag": True
                        })
                        
                        st.session_state.last_rag_data = {
                            "url_results": url_results,
                            "docs": docs,
                            "context": context,
                            "original_llm_answer": llm_answer,
                            "summarized_info": summarized_info
                        }
                    else:
                        render_step_indicator(step4_placeholder, 4, "No sources available - using LLM answer with warning", "complete")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": llm_answer + "\n\n⚠️ *Note: Unable to verify against external sources.*",
                            "is_rag": False
                        })
                        st.session_state.last_rag_data = None
                else:
                    st.session_state.current_step = 4
                    st.session_state.step_status = "✅ Not Hallucinated"
                    render_step_indicator(step4_placeholder, 4, "✅ LLM answer is not hallucinated - no RAG needed", "complete")
                    st.session_state.show_rag = False
                    
                    # LLM answer is already displayed, just store it
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": llm_answer,
                        "is_rag": False
                    })
                    
                    st.session_state.last_rag_data = {
                        "url_results": url_results,
                        "docs": docs,
                        "context": context
                    } if docs else None
                
                # Reset processing flag
                st.session_state.processing = False
                st.session_state.current_question = None
                
                # Mark as complete
                st.session_state.current_step = 5
                st.session_state.step_status = "✅ Complete!"
                
                # Brief pause before showing final state
                time.sleep(1.0)
                st.rerun()
    
    # === LEFT SIDE: Analysis Panel ===
    with col_analysis:
        if st.session_state.last_analysis:
            result = st.session_state.last_analysis["result"]
            token_analysis = st.session_state.last_analysis["token_analysis"]
            
            # Hallucination Detection Report
            render_hallucination_report(result, token_analysis)
            
            # Decision indicator
            st.markdown("---")
            if result['is_hallucinated']:
                st.error("⚡ **HALLUCINATED** → RAG System Providing Corrected Answer")
            else:
                st.success("✅ **NOT HALLUCINATED** → LLM Answer is Accurate")
            
            # RAG Unit (only shown when RAG is used or for reference)
            if st.session_state.last_rag_data:
                st.markdown("---")
                render_rag_unit(
                    st.session_state.last_rag_data["url_results"],
                    st.session_state.last_rag_data["docs"],
                    st.session_state.last_rag_data["context"],
                    st.session_state.last_rag_data.get("summarized_info")
                )
                
                # Show original LLM answer if RAG was used
                if result['is_hallucinated'] and "original_llm_answer" in st.session_state.last_rag_data:
                    st.markdown("---")
                    st.markdown("**🔍 Original LLM Answer (rejected as hallucinated):**")
                    st.markdown(f"<div style='color: #888; font-style: italic; padding: 10px; border-left: 3px solid #f44336; background: rgba(244,67,54,0.1);'>{st.session_state.last_rag_data['original_llm_answer']}</div>", unsafe_allow_html=True)
        else:
            # Initial state - minimal placeholder
            st.markdown("""
            ### 🎯 HALLUCINATION DETECTION REPORT
            ---
            
            *Ask a question to see the hallucination analysis...*
            
            ---
            
            ### 📦 RAG UNIT
            
            *Sources and context will appear here after analysis...*
            """)


if __name__ == "__main__":
    main()
