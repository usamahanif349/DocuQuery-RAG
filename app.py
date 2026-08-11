import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# 1. Page Configuration
st.set_page_config(page_title="DocuQuery | Advanced RAG Engine", page_icon="⚡", layout="wide")

st.title("⚡ DocuQuery: Multi-Doc RAG Engine")
st.caption("PyMuPDF Parsing + Hybrid Search + CrossEncoder Re-Ranking + Deep Memory + Smart Greetings")

# 2. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "faiss_retriever" not in st.session_state:
    st.session_state.faiss_retriever = None

if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Hybrid Search + Re-ranking Engine
def get_reranked_docs(query, faiss_retriever, bm25_retriever, bm25_weight, final_k):
    faiss_docs = faiss_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)

    combined = []
    seen = set()

    primary = bm25_docs if bm25_weight >= 0.5 else faiss_docs
    secondary = faiss_docs if bm25_weight >= 0.5 else bm25_docs

    for doc in primary + secondary:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    if not combined:
        return []

    reranker = load_reranker()
    pairs = [(query, doc.page_content) for doc in combined]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(combined, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    reranked_docs = [doc for doc, score in scored_docs[:final_k]]
    return reranked_docs

# 3. Sidebar Configuration
with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    model_name = st.selectbox("LLM Model", ["llama-3.3-70b-versatile", "llama3-8b-8192"])

    st.divider()
    st.header("⚙️ RAG Parameters")
    chunk_size = st.slider("Chunk Size", 200, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
    final_k = st.slider("Final Context Chunks (k)", 1, 10, 4)
    bm25_weight = st.slider("Keyword Priority (BM25)", 0.0, 1.0, 0.5, 0.1)
    memory_depth = st.slider("Memory Depth (Turns)", 2, 20, 6, 2)

    st.divider()
    st.header("📁 Document Ingestion")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Process Documents", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Parsing PDFs with PyMuPDF & indexing..."):
                all_docs = []
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(file.read())
                        tmp_path = tmp_file.name

                    loader = PyMuPDFLoader(tmp_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = file.name
                    all_docs.extend(docs)
                    os.remove(tmp_path)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                splits = text_splitter.split_documents(all_docs)

                embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                vector_store = FAISS.from_documents(splits, embeddings)
                st.session_state.faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 12})

                bm25 = BM25Retriever.from_documents(splits)
                bm25.k = 12
                st.session_state.bm25_retriever = bm25

                st.success(f"Parsed & indexed {len(splits)} chunks with PyMuPDF + Re-Ranking!")

    if st.session_state.messages:
        st.divider()
        st.header("📥 Export Chat")
        chat_export_text = ""
        for m in st.session_state.messages:
            chat_export_text += f"{m['role'].upper()}:\n{m['content']}\n\n"
            if "sources" in m and m["sources"]:
                chat_export_text += "Sources:\n"
                for src in m["sources"]:
                    chat_export_text += f"- {src['file']} (Page {src['page']})\n"
                chat_export_text += "\n" + "-"*40 + "\n\n"

        st.download_button(
            label="Download Chat History (.txt)",
            data=chat_export_text,
            file_name="docuquery_chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )

# 4. Display Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {idx}:** {src['file']} (Page {src['page']})")
                    st.caption(src['content'])

# 5. User Input & Query Execution
if prompt := st.chat_input("Ask a question about your documents..."):
    if not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        st.stop()

    if not st.session_state.faiss_retriever or not st.session_state.bm25_retriever:
        st.error("Please upload and process documents first.")
        st.stop()

    # Render User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Hybrid Retrieval + Re-ranking Execution
    retrieved_docs = get_reranked_docs(
        prompt, 
        st.session_state.faiss_retriever, 
        st.session_state.bm25_retriever, 
        bm25_weight, 
        final_k
    )

    context_text = "\n\n".join([f"[Source: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', 0)+1}]\n{doc.page_content}" for doc in retrieved_docs])

    sources_info = [
        {
            "file": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 0) + 1,
            "content": doc.page_content
        }
        for doc in retrieved_docs
    ]

    history_slice = st.session_state.messages[-(memory_depth + 1):-1]
    history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history_slice])

    # UPDATED PROMPT: Allows natural friendly greetings without breaking the RAG constraints
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert document assistant. 
    If the user's input is a casual greeting, pleasantry, or general conversational remark (e.g., "Hi", "Hello", "How are you?"), respond in a friendly, conversational manner and offer assistance with the uploaded documents.
    For all other questions, answer accurately using ONLY the context provided below, keeping chat history context in mind for follow-ups. If the context doesn't contain the answer, state that you don't know based on the provided documents.

    Chat History:
    {history}

    Context:
    {context}

    Question: {question}
    """)

    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0.2, streaming=True)
    chain = prompt_template | llm | StrOutputParser()

    # Stream AI Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in chain.stream({"history": history_text, "context": context_text, "question": prompt}):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        # Only show sources container if sources actually exist and weren't just a greeting response
        if len(prompt.split()) > 3 or "hello" not in prompt.lower():
            with st.expander("📚 View Sources"):
                for idx, src in enumerate(sources_info, 1):
                    st.markdown(f"**Source {idx}:** {src['file']} (Page {src['page']})")
                    st.caption(src['content'])

    # Save to Session State
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": sources_info if len(prompt.split()) > 3 or "hello" not in prompt.lower() else []
    })