import os
import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ----------------------------
# CONFIG
# ----------------------------

api_key = "ADD API KEY HERE"

PINECONE_API_KEY = os.getenv(api_key)
INDEX_NAME = "rag-index"

# cache models to retrieve data faster

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")  # 768 dimensions

@st.cache_resource
def get_llm():
    return OllamaLLM(model="llama3.2")

embeddings = get_embeddings()
llm = get_llm()

# pinecone vector database initialization

pc = Pinecone(api_key=api_key)

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)


# prompt template

template = """
You are an assistant for question-answering tasks.
Use the retrieved context to answer the question.
If you don't know the answer, just say you don't know.
Use maximum three sentences.

Question: {question}
Context: {context}
Answer:
"""

# functions to load page, divide the website into chunks and storing its index as an embeddings into the pincone vector database

def load_page(url):
    loader = WebBaseLoader(web_path=url)
    return loader.load()

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

def index_documents(docs):
    vector_store.add_documents(docs)

def retrieve_documents(query):
    return vector_store.similarity_search(query, k=3)

def generate_answer(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    return chain.invoke({"question": question, "context": context})


# streamlit UI

st.title("Web Scrapper RAG Agent")

if "indexed_url" not in st.session_state:
    st.session_state.indexed_url = ""

with st.form("url_form"):
    url = st.text_input("Enter URL:")
    submit = st.form_submit_button("Load & Index")

if submit and url.strip():
    with st.spinner("Loading and indexing..."):
        try:
            documents = load_page(url.strip())
            chunks = split_text(documents)
            index_documents(chunks)
            st.session_state.indexed_url = url.strip()
            st.success(f"Indexed {len(chunks)} chunks into Pinecone!")
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.indexed_url:
    st.caption(f"Indexed URL: {st.session_state.indexed_url}")

question = st.chat_input("Ask a question about the page")

if question:
    if not st.session_state.indexed_url:
        st.warning("Please index a URL first.")
    else:
        st.chat_message("user").write(question)
        docs = retrieve_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        answer = generate_answer(question, context)
        st.chat_message("assistant").write(answer)