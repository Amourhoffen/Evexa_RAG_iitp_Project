"""
====================================================================
  Hybrid Document Retrieval System
  BM25 (weight=0.4) + FAISS (weight=0.6)
  Embeddings: HuggingFace all-MiniLM-L6-v2
====================================================================
"""
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Imports
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever

print("=" * 60)
print("  Evexa Buddy - Hybrid Retrieval System")
print("  BM25 (0.4) + FAISS (0.6) | all-MiniLM-L6-v2")
print("=" * 60)

# 5 Sample Documents
documents = [
    Document(
        page_content=(
            "A connection timeout error occurs when a client cannot "
            "establish a connection to a server within a specified time. "
            "To fix connection timeout: increase the timeout value in settings, "
            "check if firewall is blocking the port, verify the server IP and port "
            "number are correct, and ensure the remote server is actually running."
        ),
        metadata={"source": "networking_guide", "topic": "timeout"}
    ),
    Document(
        page_content=(
            "BM25 is a bag-of-words retrieval function used by search engines "
            "to rank matching documents. It is based on term frequency and "
            "inverse document frequency. BM25 is a sparse retrieval method "
            "that works well for exact keyword matching without neural networks."
        ),
        metadata={"source": "ir_textbook", "topic": "BM25"}
    ),
    Document(
        page_content=(
            "FAISS (Facebook AI Similarity Search) enables fast nearest-neighbor "
            "search over dense embedding vectors. It is widely used in RAG pipelines "
            "to semantically retrieve documents using neural embeddings. FAISS "
            "supports both CPU and GPU-based indexing for large-scale retrieval."
        ),
        metadata={"source": "faiss_docs", "topic": "vector_search"}
    ),
    Document(
        page_content=(
            "Hybrid retrieval combines sparse keyword search (BM25) with dense "
            "semantic search (FAISS). Using an EnsembleRetriever improves recall "
            "by leveraging both exact word matching and contextual meaning, "
            "making retrieval more robust than either method used in isolation."
        ),
        metadata={"source": "rag_paper", "topic": "hybrid_retrieval"}
    ),
    Document(
        page_content=(
            "The all-MiniLM-L6-v2 model maps sentences to 384-dimensional dense "
            "vectors. It is a lightweight sentence-transformer optimized for "
            "semantic similarity tasks. It is widely used in RAG systems "
            "due to its speed and accuracy balance."
        ),
        metadata={"source": "huggingface_docs", "topic": "embeddings"}
    ),
]

print(f"\n[OK] {len(documents)} documents loaded\n")

# BM25 Retriever
print("[...] Building BM25 Retriever (sparse/keyword)...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3
print("[OK] BM25 ready\n")

# FAISS Retriever
print("[...] Loading HuggingFace Model: all-MiniLM-L6-v2")
print("      (First run downloads ~90MB - please wait...)\n")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("[OK] Embeddings model loaded\n")

print("[...] Building FAISS Vector Store...")
faiss_vectorstore = FAISS.from_documents(documents, embeddings)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})
print("[OK] FAISS ready\n")

# EnsembleRetriever
print("[...] Creating EnsembleRetriever [BM25=0.4, FAISS=0.6]...")
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]
)
print("[OK] EnsembleRetriever ready\n")

# Run Query
query = "how to fix connection timeout"

print("=" * 60)
print(f'  QUERY: "{query}"')
print("=" * 60)

results = ensemble_retriever.invoke(query)

print(f"\n  Top {len(results)} Retrieved Documents:\n")
for i, doc in enumerate(results, 1):
    print(f"  --- Result {i} ---")
    print(f"  Source  : {doc.metadata.get('source', 'N/A')}")
    print(f"  Topic   : {doc.metadata.get('topic', 'N/A')}")
    print(f"  Content : {doc.page_content[:200]}...")
    print()

print("=" * 60)
print("  SUCCESS - Hybrid Retrieval Complete!")
print()
print("  BM25  (0.4) -> keyword match: 'connection', 'timeout'")
print("  FAISS (0.6) -> semantic match by meaning")
print("  Ensemble    -> merged and re-ranked both results")
print("=" * 60)
