# ⚡ DocuQuery: Multi-Doc RAG Engine
Live app : https://docuquery-rag-fpglnj4fz59mxbh5kys5uz.streamlit.app/

DocuQuery is an end-to-end Retrieval-Augmented Generation (RAG) web application that enables users to perform fast, context-aware Q&A over complex PDF documents with page-level source attribution.

##  Key Features

* **Multi-PDF Ingestion:** Upload and index multiple PDF files simultaneously.
* **Fast Embeddings & Vector Search:** Uses `FastEmbed` (`BAAI/bge-small-en-v1.5`) paired with `FAISS` for high-speed local similarity search.
* **Ultra-Fast LLM Generation:** Powered by `GPT-OSS (120B / 20B)-versatile` via the Groq API.
* **Page-Level Citations:** Displays precise document references, page numbers, and text snippets to verify model answers and eliminate hallucinations.
* **Customizable Parameters:** Adjust Chunk Size, Overlap, and Top-K retrieval settings in real time.

## 🛠 Tech Stack

* **Frontend UI:** Streamlit
* **LLM Engine:** Groq API (`GPT-OSS (120B / 20B)-versatile`)
* **Embeddings:** FastEmbed
* **Vector Database:** FAISS
* **Orchestration:** LangChain / PyPDF

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/DocuQuery-RAG.git](https://github.com/YOUR_USERNAME/DocuQuery-RAG.git)
   cd DocuQuery-RAG
