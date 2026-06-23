# WikiMind: Hybrid Agentic RAG Pipeline

WikiMind is a state-of-the-art **Tri-Modal Retrieval-Augmented Generation (RAG)** pipeline designed to ingest, process, and accurately synthesize answers from the Wikipedia dataset. Built with production-grade 2025/2026 architectural patterns, it goes beyond naive vector search by implementing a self-healing LangGraph workflow.

## 🚀 Key Features

*   **Tri-Modal Retrieval Architecture**:
    *   *Dense Search*: Semantic matching using `fastembed` (BGE models) and Qdrant.
    *   *Sparse Search*: Exact keyword matching using BM25.
    *   *Vectorless / PageIndex*: Tree-based hierarchical document navigation to prevent context fragmentation.
*   **Agentic Self-Healing**: Utilizing **LangGraph** for a Corrective RAG (CRAG) loop. If retrieved documents are irrelevant, the agent automatically rewrites the query and searches again.
*   **Query Expansion**: Optional HyDE/Multi-Query expansion to maximize recall.
*   **Observability & Guardrails**: Integrated with Langfuse for distributed tracing and LLM safety filters.

---

## 🏗️ Project Progress

### ✅ Phase 1: Foundation & Data Pipeline
- Set up Python environment with `uv` and configured dependencies.
- Created scalable `wikipedia_stream.py` to ingest Hugging Face Wikipedia datasets via streaming (avoiding massive local downloads).
- Implemented intelligent semantic chunking using `MarkdownHeaderTextSplitter`.
- Initialized local **Qdrant** vector database and integrated `fastembed` for blazing-fast local embeddings.

### ⏳ Phase 2: Tri-Modal Retrieval Engine (Up Next)
- Implement Dense Embeddings (Vector Search).
- Implement Sparse Search (BM25).
- Build Vectorless/PageIndex tree structure processing.
- Implement Fusion (RRF) and Cross-Encoder Re-ranking.

### ⏳ Phase 3: Agentic Flow & Self-Healing
### ⏳ Phase 4: Observability & Guardrails
### ⏳ Phase 5: Cloud Deployment

---

## 🛠️ Tech Stack
- **Data & Embedding**: Hugging Face Datasets, LangChain, FastEmbed
- **Vector Database**: Qdrant (Local & Cloud)
- **Agent Framework**: LangGraph
- **Observability**: Langfuse
