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


# api keys section 

SERPAPI_API_KEY = " API "
PINECONE_API_KEY = " API "

os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME = "rag-embedds"
PINECONE_NAMESPACE = "web-rag-records"


# llama models one for using llm and another for Q&A

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


@st.cache_resource
def get_llm():
    return OllamaLLM(model="llama3.2", temperature=0)


@st.cache_resource
def get_llm_simple():
    return OllamaLLM(model="llama3.2")


embeddings = get_embeddings()
llm = get_llm()
llm_simple = get_llm_simple()


# connecting pinecone VDB 

pc = Pinecone(api_key=PINECONE_API_KEY)

if HAS_LANGCHAIN_PINECONE:
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
else:
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text"
    )


# prompt template

template = """
You are a precise and reliable AI assistant.

Your task is to answer the user's question using ONLY the information
present in the provided context.

Rules:
- Do NOT use outside knowledge.
- If exact details are missing, provide the closest related answer from available context.
- Do not invent exact names, numbers, or dates not present in context.
- If context is limited, explicitly mention that the response is based on limited context.
- Keep the response clear, factual, and detailed.
- Write around {target_lines} lines.
- Use simple plain-language sentences.
- Include key facts, conditions, and short explanation where relevant.

Question:
{question}

Retrieved Context:
{context}

Answer:
"""

llm_only_template = """
You are a precise and reliable AI assistant.

Rules:
- Keep the answer detailed, factual, and direct.
- Write between {target_lines_min} and {target_lines_max} lines.
- Use simple plain-language sentences.
- Include key facts and short explanation where useful.
- Do not include links, URLs, citations, or source lists.

Question:
{question}

Answer:
"""

# serp ai for webscrapping

def search_web(query, max_results=3):

    search = SerpAPIWrapper()

    results = search.results(query)

    urls = []

    if "organic_results" in results:
        for r in results["organic_results"][:max_results]:
            urls.append(r["link"])

    return urls


# web scrapping pages loading

def load_pages(urls):

    documents = []

    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)
        except:
            pass

    return documents


# splits web scrapped things into chunks which are provided to pinecone VDB which stores these as embeddings

def split_text(documents, chunk_size=1200, chunk_overlap=100, max_chunks=18):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)
    return chunks[:max_chunks]

def index_documents(docs, namespace):

    texts = []
    metadatas = []
    ids = []

    timestamp = int(time.time())

    for i, doc in enumerate(docs):

        texts.append(doc.page_content)

        metadatas.append({
            "source": doc.metadata.get("source", ""),
            "chunk_text": doc.page_content[:500],  # preview of chunk
            "created_at": timestamp
        })

        ids.append(f"chunk-{timestamp}-{i}-{uuid.uuid4().hex[:8]}")

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        namespace=namespace
    )


# retrieve documents

def retrieve_documents(query, namespace, k=5, relevance_threshold=0.40):

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

        # Fallback 1: Relax threshold if nothing passes strict filter
        if not docs_with_scores:
            relaxed_threshold = max(0.20, relevance_threshold - 0.15)
            for doc, score in raw_results:
                if score >= relaxed_threshold:
                    docs_with_scores.append((doc, float(score)))

        # Fallback 2: Always return top-k from raw relevance results if still empty
        if not docs_with_scores and raw_results:
            docs_with_scores = [(doc, float(score)) for doc, score in raw_results[:k]]
    except Exception:
        raw_results = vector_store.similarity_search(query, k=k, namespace=namespace)
        docs_with_scores = [(doc, 1.0) for doc in raw_results]

    return docs_with_scores


def format_retrieved_context(docs_with_scores):
    if not docs_with_scores:
        return ""

    blocks = []
    for idx, (doc, score) in enumerate(docs_with_scores, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk = re.sub(r"\s+", " ", doc.page_content.strip())
        blocks.append(
            f"[Document {idx}]\n"
            f"Source: {source}\n"
            f"Relevance: {score:.3f}\n"
            f"Content: {chunk}"
        )

    return "\n\n---\n\n".join(blocks)


# answer generation

def generate_answer(question, context, target_lines):

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm

    return chain.invoke({
        "question": question,
        "context": context,
        "target_lines": target_lines
    })


def generate_llm_only_answer(question, target_lines):

    prompt = ChatPromptTemplate.from_template(llm_only_template)

    chain = prompt | llm_simple

    return chain.invoke({
        "question": question,
        "target_lines_min": max(6, target_lines - 2),
        "target_lines_max": max(8, target_lines)
    })


def estimate_answer_lines(question):
    q = question.strip().lower()
    if any(token in q for token in ["why", "how", "compare", "difference", "advantages", "disadvantages", "in detail"]):
        return 12
    if any(token in q for token in ["explain", "describe", "overview", "uses", "working", "benefits"]):
        return 10
    return 12


def infer_output_type(question):
    q = question.strip().lower()

    factual_markers = [
        "what is", "who is", "when", "where", "which", "define", "full form", "capital", "mother", "father"
    ]
    analytical_markers = [
        "why", "how", "compare", "difference", "advantages", "disadvantages", "pros", "cons", "in detail"
    ]

    if any(marker in q for marker in analytical_markers):
        return "analytical"
    if any(marker in q for marker in factual_markers):
        return "factual"
    return "descriptive"


def get_speed_profile(question):
    target_lines = estimate_answer_lines(question)
    output_type = infer_output_type(question)

    if output_type == "factual":
        return {
            "target_lines": target_lines,
            "output_type": output_type,
            "max_urls": 4,
            "max_urls_cap": 10,
            "max_chunks": 30,
            "max_chunks_cap": 90,
            "retrieval_k": 5,
            "retrieval_k_cap": 9,
            "relevance_threshold": 0.40,
            "min_relevance_threshold": 0.20,
            "chunk_size": 1400,
            "chunk_overlap": 180,
            "max_expansions": 3
        }

    if output_type == "descriptive":
        return {
            "target_lines": target_lines,
            "output_type": output_type,
            "max_urls": 5,
            "max_urls_cap": 12,
            "max_chunks": 45,
            "max_chunks_cap": 120,
            "retrieval_k": 6,
            "retrieval_k_cap": 10,
            "relevance_threshold": 0.38,
            "min_relevance_threshold": 0.18,
            "chunk_size": 1700,
            "chunk_overlap": 220,
            "max_expansions": 3
        }

    return {
        "target_lines": target_lines,
        "output_type": "analytical",
        "max_urls": 7,
        "max_urls_cap": 14,
        "max_chunks": 60,
        "max_chunks_cap": 150,
        "retrieval_k": 7,
        "retrieval_k_cap": 12,
        "relevance_threshold": 0.35,
        "min_relevance_threshold": 0.15,
        "chunk_size": 2100,
        "chunk_overlap": 260,
        "max_expansions": 4
    }


def is_context_sufficient(docs_with_scores, context, output_type):
    if not docs_with_scores or not context.strip():
        return False

    min_requirements = {
        "factual": {"docs": 1, "chars": 300},
        "descriptive": {"docs": 1, "chars": 500},
        "analytical": {"docs": 2, "chars": 900},
    }

    req = min_requirements.get(output_type, min_requirements["descriptive"])

    return len(docs_with_scores) >= req["docs"] and len(context) >= req["chars"]


def collect_adaptive_context(question, namespace, profile):
    last_docs_with_scores = []
    last_context = ""

    for attempt in range(profile["max_expansions"]):
        current_max_urls = min(profile["max_urls"] + (attempt * 2), profile["max_urls_cap"])
        current_max_chunks = min(profile["max_chunks"] + (attempt * 18), profile["max_chunks_cap"])
        current_retrieval_k = min(profile["retrieval_k"] + attempt, profile["retrieval_k_cap"])
        current_threshold = max(
            profile["min_relevance_threshold"],
            profile["relevance_threshold"] - (attempt * 0.06)
        )

        urls = search_web(question, max_results=current_max_urls)
        documents = load_pages(urls)

        if len(documents) == 0:
            continue

        chunks = split_text(
            documents,
            chunk_size=profile["chunk_size"],
            chunk_overlap=profile["chunk_overlap"],
            max_chunks=current_max_chunks
        )

        if len(chunks) == 0:
            continue

        index_documents(chunks, namespace=namespace)

        docs_with_scores = retrieve_documents(
            question,
            namespace=namespace,
            k=current_retrieval_k,
            relevance_threshold=current_threshold
        )

        context = format_retrieved_context(docs_with_scores)

        last_docs_with_scores = docs_with_scores
        last_context = context

        if is_context_sufficient(docs_with_scores, context, profile["output_type"]):
            break

    return last_docs_with_scores, last_context


def render_typewriter(text, delay=0.01):
    placeholder = st.empty()
    rendered = ""

    for ch in text:
        rendered += ch
        placeholder.markdown(rendered)
        time.sleep(delay)


def render_dual_typewriter(left_placeholder, right_placeholder, left_text, right_text, delay=0.006):
    left_rendered = ""
    right_rendered = ""
    max_len = max(len(left_text), len(right_text))

    for i in range(max_len):
        if i < len(left_text):
            left_rendered += left_text[i]
            left_placeholder.markdown(left_rendered)

        if i < len(right_text):
            right_rendered += right_text[i]
            right_placeholder.markdown(right_rendered)

        time.sleep(delay)


def sanitize_answer_text(answer):
    text = str(answer)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"\n\s*(Sources?|References?)\s*:.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fallback_no_context_answer():
    return "The provided context does not contain enough information to answer this question."


def compute_hallucination_metrics(result):
    grounding_pct = float(result["overall_confidence"])
    hallucination_pct = max(0.0, min(100.0, 100.0 - grounding_pct))

    if hallucination_pct <= 25:
        status = "Not Hallucinated"
    elif hallucination_pct <= 50:
        status = "Mild Hallucination"
    else:
        status = "Likely Hallucinated"

    return {
        "grounding_pct": grounding_pct,
        "hallucination_pct": hallucination_pct,
        "status": status
    }


def tokenize_text(text, min_len=3):
    stop_words = {
        "the", "and", "for", "that", "this", "with", "from", "into", "have", "has", "had",
        "are", "was", "were", "will", "would", "shall", "could", "should", "about", "your",
        "their", "there", "which", "when", "where", "what", "who", "why", "how", "does",
        "did", "can", "also", "than", "then", "them", "they", "its", "it's", "our", "you"
    }
    raw_tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-']+\b", text.lower())
    return [token for token in raw_tokens if len(token) >= min_len and token not in stop_words]


def token_support_from_docs(answer_text, docs, max_tokens=20):
    answer_tokens = tokenize_text(answer_text)
    context_tokens = set()

    for doc in docs:
        context_tokens.update(tokenize_text(doc.page_content))

    unique_answer_tokens = []
    seen = set()
    for token in answer_tokens:
        if token not in seen:
            seen.add(token)
            unique_answer_tokens.append(token)

    if not unique_answer_tokens:
        return {
            "token_support_pct": 100.0,
            "token_hallucination_pct": 0.0,
            "matched_tokens": [],
            "unmatched_tokens": []
        }

    matched = [token for token in unique_answer_tokens if token in context_tokens]
    unmatched = [token for token in unique_answer_tokens if token not in context_tokens]

    support_pct = (len(matched) / len(unique_answer_tokens)) * 100.0
    hallucination_pct = 100.0 - support_pct

    return {
        "token_support_pct": max(0.0, min(100.0, support_pct)),
        "token_hallucination_pct": max(0.0, min(100.0, hallucination_pct)),
        "matched_tokens": matched[:max_tokens],
        "unmatched_tokens": unmatched[:max_tokens]
    }


def merge_hallucination_scores(embedding_hallucination_pct, token_hallucination_pct):
    # Weighted blend: embeddings capture semantic support, tokens capture literal support
    return (0.65 * embedding_hallucination_pct) + (0.35 * token_hallucination_pct)


def llm_hallucination_verdict_word(llm_hallucination_pct, rag_hallucination_pct):
    # YES => LLM is hallucinating more than RAG by meaningful margin
    if (llm_hallucination_pct - rag_hallucination_pct) >= 5.0:
        return "LLM Hallucinated"
    return "LLM Not Hallucinated"


def display_source_text_snippets(docs, max_items=3, preview_chars=350):
    st.subheader("Source Context Used")

    if not docs:
        st.info("No source context retrieved.")
        return

    for i, doc in enumerate(docs[:max_items], 1):
        source = doc.metadata.get("source", "Unknown source")
        snippet = doc.page_content.strip().replace("\n", " ")
        snippet = re.sub(r"\s+", " ", snippet)
        snippet = snippet[:preview_chars] + ("..." if len(snippet) > preview_chars else "")

        with st.expander(f"Source {i}: {source}", expanded=(i == 1)):
            st.write(snippet)


# ===== HALLUCINATION DETECTION FUNCTIONS =====

def split_into_sentences(text):
    """
    Split text into individual sentences for granular analysis.
    
    Args:
        text (str): The LLM generated answer
    
    Returns:
        list[str]: List of sentences
    """
    # Use regex to split on period, exclamation, question marks
    # Avoid splitting on abbreviations
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(pattern, text)
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


@st.cache_data(ttl=3600)
def cached_embed_text(text, _embeddings_model):
    """
    Cache embeddings to avoid recomputing.
    
    Args:
        text (str): Text to embed
        _embeddings_model: Ollama embeddings instance (prefixed with _ to skip hashing)
    
    Returns:
        list[float]: Embedding vector
    """
    return _embeddings_model.embed_query(text)


def embed_text(text, embeddings_model):
    """
    Convert text to embedding vector.
    
    Args:
        text (str): Text to embed
        embeddings_model: Ollama embeddings instance
    
    Returns:
        np.array: 768-dimensional vector
    """
    embedding = cached_embed_text(text, embeddings_model)
    return np.array(embedding)


def calculate_cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1 (np.array): First embedding vector
        vec2 (np.array): Second embedding vector
    
    Returns:
        float: Similarity score (0-1)
    """
    # Handle edge case of zero vectors
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    
    # Reshape for sklearn's cosine_similarity
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    
    similarity = cosine_similarity(vec1, vec2)[0][0]
    
    # Ensure result is between 0 and 1
    return max(0.0, min(1.0, similarity))


def classify_confidence(score):
    """
    Classify confidence score into categories.
    
    Args:
        score (float): Similarity score (0-1)
    
    Returns:
        tuple: (status_text, color)
    """
    if score >= 0.75:
        return "✅ GROUNDED", "green"
    elif score >= 0.50:
        return "⚠️ PARTIAL", "orange"
    elif score >= 0.30:
        return "⚡ WEAK", "yellow"
    else:
        return "❌ HALLUCINATED", "red"


def detect_hallucination(answer, retrieved_docs, embeddings_model):
    """
    Analyze LLM answer for hallucination by comparing with retrieved context.
    
    Args:
        answer (str): LLM generated response
        retrieved_docs (list): Retrieved context documents from Pinecone
        embeddings_model: Ollama embeddings instance
    
    Returns:
        dict: {
            'overall_confidence': float (0-100),
            'classification': str (GROUNDED/PARTIAL/HALLUCINATED),
            'sentence_scores': list[dict],
            'avg_similarity': float (0-1),
            'metadata': dict
        }
    """
    start_time = time.time()
    
    # Split answer into sentences
    sentences = split_into_sentences(answer)
    
    if len(sentences) == 0:
        return {
            'overall_confidence': 0,
            'classification': 'UNKNOWN',
            'sentence_scores': [],
            'avg_similarity': 0,
            'metadata': {
                'chunks_count': len(retrieved_docs),
                'sentence_count': 0,
                'processing_time': 0
            }
        }
    
    # Embed all context chunks
    context_embeddings = []
    for doc in retrieved_docs:
        context_text = doc.page_content
        context_emb = embed_text(context_text, embeddings_model)
        context_embeddings.append(context_emb)
    
    # Analyze each sentence
    sentence_scores = []
    
    for sentence in sentences:
        # Embed sentence
        sentence_emb = embed_text(sentence, embeddings_model)
        
        # Find max similarity with any context chunk
        max_similarity = 0.0
        
        for context_emb in context_embeddings:
            similarity = calculate_cosine_similarity(sentence_emb, context_emb)
            max_similarity = max(max_similarity, similarity)
        
        # Classify this sentence
        status, color = classify_confidence(max_similarity)
        
        sentence_scores.append({
            'text': sentence,
            'score': max_similarity,
            'status': status,
            'color': color
        })
    
    # Calculate overall confidence
    avg_similarity = np.mean([s['score'] for s in sentence_scores])
    overall_confidence = avg_similarity * 100
    
    # Classify overall result
    if overall_confidence >= 75:
        classification = "✅ WELL-GROUNDED"
    elif overall_confidence >= 50:
        classification = "⚠️ PARTIALLY GROUNDED"
    elif overall_confidence >= 30:
        classification = "⚡ WEAKLY GROUNDED"
    else:
        classification = "❌ LIKELY HALLUCINATED"
    
    processing_time = time.time() - start_time
    
    return {
        'overall_confidence': overall_confidence,
        'classification': classification,
        'sentence_scores': sentence_scores,
        'avg_similarity': avg_similarity,
        'metadata': {
            'chunks_count': len(retrieved_docs),
            'sentence_count': len(sentences),
            'processing_time': processing_time
        }
    }


def get_emoji_for_confidence(confidence):
    """Get emoji representation for confidence score."""
    if confidence >= 75:
        return "✅"
    elif confidence >= 50:
        return "⚠️"
    elif confidence >= 30:
        return "⚡"
    else:
        return "❌"


def get_color_for_score(score):
    """Get color for progress bar based on score."""
    if score >= 75:
        return "#00c853"  # green
    elif score >= 50:
        return "#ff6f00"  # orange
    elif score >= 30:
        return "#ffd600"  # yellow
    else:
        return "#d32f2f"  # red


def display_hallucination_report(result):
    """
    Display comprehensive hallucination detection report in Streamlit UI.
    
    Args:
        result (dict): Detection result from detect_hallucination()
    """
    st.markdown("---")
    st.markdown("## 🎯 HALLUCINATION DETECTION REPORT")
    
    # Overall Confidence Header
    confidence = result['overall_confidence']
    classification = result['classification']
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### Overall Confidence: **{confidence:.1f}%** {get_emoji_for_confidence(confidence)}")
        st.markdown(f"**Classification:** {classification}")
    
    with col2:
        st.metric("Avg Similarity", f"{result['avg_similarity']:.3f}")
    
    # Visual Confidence Meter
    st.markdown("### Confidence Meter")
    color = get_color_for_score(confidence)
    
    # Progress bar with color
    st.markdown(f"""
        <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; height: 30px; position: relative;">
            <div style="width: {confidence}%; background-color: {color}; border-radius: 10px; height: 30px; 
                        display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {confidence:.1f}%
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Detailed Sentence Analysis
    st.markdown("### 📊 Detailed Sentence Analysis")
    
    for i, item in enumerate(result['sentence_scores'], 1):
        score_pct = item['score'] * 100
        emoji = get_emoji_for_confidence(score_pct)
        status = item['status']
        text = item['text']
        
        with st.expander(f"{emoji} **{score_pct:.1f}%** - Sentence {i}", expanded=False):
            st.markdown(f"**Text:** {text}")
            st.markdown(f"**Status:** {status}")
            st.progress(item['score'])
    
    # Interpretation Guide
    st.markdown("### 💡 Interpretation")
    
    if confidence >= 75:
        st.success("""
        ✅ **HIGH CONFIDENCE**: The answer is strongly supported by retrieved context.
        Most statements closely match information from scraped sources.
        """)
    elif confidence >= 50:
        st.warning("""
        ⚠️ **MODERATE CONFIDENCE**: The answer is partially supported.
        Some statements may include inferences or generalizations.
        Consider verifying critical information from sources.
        """)
    elif confidence >= 30:
        st.warning("""
        ⚡ **LOW CONFIDENCE**: The answer has weak support from context.
        Many statements may include reasonable inferences but lack direct support.
        Recommend reviewing source documents for important information.
        """)
    else:
        st.error("""
        ❌ **VERY LOW CONFIDENCE**: The answer has very weak support from context.
        Many statements may be fabricated or incorrectly inferred.
        Strongly recommend reviewing source documents directly.
        """)
    
    # Technical Details Expander
    with st.expander("⚙️ Technical Details"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Embedding Model", "nomic-embed-text")
            st.metric("Vector Dimensions", "768")
        
        with col2:
            st.metric("Retrieved Chunks", result['metadata']['chunks_count'])
            st.metric("Sentences Analyzed", result['metadata']['sentence_count'])
        
        with col3:
            st.metric("Processing Time", f"{result['metadata']['processing_time']:.2f}s")
            st.metric("Threshold (High)", "0.75")


# ===== END HALLUCINATION DETECTION FUNCTIONS =====


# image search function using serp api

def search_images(query):
    import requests

    params = {
        "engine": "google_images",
        "q": query,
        "api_key": os.environ["SERPAPI_API_KEY"]
    }

    response = requests.get("https://serpapi.com/search", params=params)
    results = response.json()

    images = []

    if "images_results" in results:
        for img in results["images_results"][:4]:
            images.append(img["original"])

    return images

# streamlit UI

st.title("Web Scrapper RAG Assistant")
st.subheader("RAG Grounded Response")

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 2rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        min-height: 420px;
        padding: 1.1rem 1.1rem 0.8rem 1.1rem;
        border-radius: 12px;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stMarkdownContainer"] {
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.form("rag_compare_form", clear_on_submit=False):
    question = st.text_input("Ask anything...")
    run_button = st.form_submit_button("Generate")

if run_button and question.strip():
    profile = get_speed_profile(question)
    namespace = PINECONE_NAMESPACE

    with st.spinner("Collecting adaptive context from web + Pinecone..."):
        docs_with_scores, context = collect_adaptive_context(question, namespace, profile)

    docs = [item[0] for item in docs_with_scores]

    if len(docs_with_scores) == 0:
        context = ""
        docs = []

    with st.spinner("Generating both responses..."):
        with ThreadPoolExecutor(max_workers=2) as executor:
            llm_future = executor.submit(
                generate_llm_only_answer,
                question,
                profile["target_lines"]
            )
            if context.strip():
                rag_future = executor.submit(
                    generate_answer,
                    question,
                    context,
                    profile["target_lines"]
                )
            else:
                rag_future = executor.submit(
                    generate_llm_only_answer,
                    question,
                    profile["target_lines"]
                )

            llm_answer = sanitize_answer_text(llm_future.result())
            rag_answer = sanitize_answer_text(rag_future.result())

    with st.spinner("Computing hallucination analysis..."):
        rag_result = detect_hallucination(rag_answer, docs, embeddings)
        llm_result = detect_hallucination(llm_answer, docs, embeddings)

    st.markdown("### Response Comparison")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### LLM Output")
        with st.container(border=True):
            llm_placeholder = st.empty()
            llm_status_placeholder = st.empty()
            llm_progress_placeholder = st.empty()
            llm_token_placeholder = st.empty()
            llm_support_placeholder = st.empty()

    with col2:
        st.markdown("#### RAG Output")
        with st.container(border=True):
            rag_placeholder = st.empty()
            rag_status_placeholder = st.empty()
            rag_progress_placeholder = st.empty()
            rag_token_placeholder = st.empty()
            rag_support_placeholder = st.empty()

    render_dual_typewriter(
        llm_placeholder,
        rag_placeholder,
        llm_answer,
        rag_answer,
        delay=0.0035
    )

    llm_embed_metrics = compute_hallucination_metrics(llm_result)
    rag_embed_metrics = compute_hallucination_metrics(rag_result)

    llm_token_metrics = token_support_from_docs(llm_answer, docs)
    rag_token_metrics = token_support_from_docs(rag_answer, docs)

    llm_final_hallucination = merge_hallucination_scores(
        llm_embed_metrics["hallucination_pct"],
        llm_token_metrics["token_hallucination_pct"]
    )
    rag_final_hallucination = merge_hallucination_scores(
        rag_embed_metrics["hallucination_pct"],
        rag_token_metrics["token_hallucination_pct"]
    )

    llm_status = "Not Hallucinated" if llm_final_hallucination <= 25 else ("Mild Hallucination" if llm_final_hallucination <= 50 else "Likely Hallucinated")
    rag_status = "Not Hallucinated" if rag_final_hallucination <= 25 else ("Mild Hallucination" if rag_final_hallucination <= 50 else "Likely Hallucinated")

    llm_status_placeholder.caption(f"Hallucination: {llm_final_hallucination:.1f}% | {llm_status}")
    llm_progress_placeholder.progress(llm_final_hallucination / 100.0)
    llm_token_placeholder.caption(
        "Hallucination Tokens: " + (", ".join(llm_token_metrics["unmatched_tokens"]) if llm_token_metrics["unmatched_tokens"] else "None")
    )
    llm_support_placeholder.caption(
        f"Token Match with VDB Chunks: {llm_token_metrics['token_support_pct']:.1f}%"
    )

    rag_status_placeholder.caption(f"Hallucination: {rag_final_hallucination:.1f}% | {rag_status}")
    rag_progress_placeholder.progress(rag_final_hallucination / 100.0)
    rag_token_placeholder.caption(
        "Hallucination Tokens: " + (", ".join(rag_token_metrics["unmatched_tokens"]) if rag_token_metrics["unmatched_tokens"] else "None")
    )
    rag_support_placeholder.caption(
        f"Token Match with VDB Chunks: {rag_token_metrics['token_support_pct']:.1f}%"
    )

    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.metric("LLM Hallucination %", f"{llm_final_hallucination:.1f}%")
    with stat_col2:
        st.metric("RAG Hallucination %", f"{rag_final_hallucination:.1f}%")

    final_word = llm_hallucination_verdict_word(llm_final_hallucination, rag_final_hallucination)
    st.markdown("### Final Verdict (One Word)")
    st.markdown(f"## {final_word}")