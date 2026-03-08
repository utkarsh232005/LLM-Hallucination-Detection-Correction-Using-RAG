import os
import time
import streamlit as st

from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain_pinecone import PineconeVectorStore
    HAS_LANGCHAIN_PINECONE = True
except ImportError:
    from langchain_community.vectorstores import Pinecone as PineconeVectorStore
    HAS_LANGCHAIN_PINECONE = False
from pinecone import Pinecone


# api keys section 

SERPAPI_API_KEY = "3b5e8c37d4769cf12f42df01df5baa17f207836ee859d08f62d66607cd06cfb4"
PINECONE_API_KEY = "pcsk_4EeaiW_PxmXpizoWmimbi8q9Cn3NTEMQJK9Xz14epbTWVwJGyWbyRp6cQy5BeEuE3AP9ws"

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