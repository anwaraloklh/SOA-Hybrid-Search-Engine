# 🔎 Enterprise SOA Hybrid Search Engine

This is an advanced, production-grade hybrid search engine built on **Service-Oriented Architecture (SOA)**, integrating lexical matching (BM25) and dense semantic retrieval (Sentence-BERT).

## 🏛️ System Architecture
The project is strictly split into decoupled microservices communicating asynchronously via HTTP REST APIs:
- **Preprocessing & Refinement Service (Port 8001):** Handles text cleaning, stemming, and WordNet-based query expansion.
- **Retrieval & Ranking Service (Port 8002):** Orchestrates dynamic dataset swapping, BM25 indexing, SBERT cosine similarity, and both Parallel & Serial hybrid search.
- **API Gateway & Gradio UI (Port 8000):** Provides a visual, interactive search interface.

## 🚀 How to Run the Project
To launch the complete distributed system, open three terminal windows and run:

1. **Start Preprocessing Service:**
```bash
python preprocessing_service.py