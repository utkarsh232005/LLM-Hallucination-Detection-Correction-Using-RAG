import os
import time
import re
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

SERPAPI_API_KEY = "Enter api"
PINECONE_API_KEY = "Enter api"

os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME = "rag-embedds"


# llama models one for using llm and another for Q&A

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


@st.cache_resource
def get_llm():
    return OllamaLLM(model="llama3.2:1b")


embeddings = get_embeddings()
llm = get_llm()


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
You are an expert AI assistant.

Use the provided context to answer the question in detail.

Rules:
- Explain the concept clearly.
- Use simple language.
- Provide examples if possible.
- Structure the answer in paragraphs.

Question:
{question}

Context:
{context}

Detailed Answer:
"""

# serp ai for webscrapping

def search_web(query):

    search = SerpAPIWrapper()

    results = search.results(query)

    urls = []

    if "organic_results" in results:
        for r in results["organic_results"][:5]:
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

def split_text(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)

def index_documents(docs):

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

        ids.append(f"chunk-{timestamp}-{i}")

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )


# retrieve documents

def retrieve_documents(query):

    return vector_store.similarity_search(query, k=3)


# answer generation

def generate_answer(question, context):

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm

    return chain.invoke({
        "question": question,
        "context": context
    })


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

question = st.chat_input("Ask anything...")


if question:

    st.chat_message("user").write(question)
    

    with st.spinner("Searching Google..."):

        urls = search_web(question)

    st.subheader("Sources Found")

    for url in urls:
        st.write(url)

    with st.spinner("Scraping websites..."):

        documents = load_pages(urls)

    if len(documents) == 0:
        st.error("Could not scrape any pages")
        st.stop()

    chunks = split_text(documents)

    st.subheader("Chunks Created")
    st.write(len(chunks))

    with st.spinner("Storing embeddings in Pinecone..."):

        index_documents(chunks)

    docs = retrieve_documents(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    with st.spinner("Generating answer..."):

        answer = generate_answer(question, context)

    st.chat_message("assistant").write(answer)

    # Hallucination Detection
    with st.spinner("Analyzing answer for hallucinations..."):
        detection_result = detect_hallucination(answer, docs, embeddings)
    
    display_hallucination_report(detection_result)