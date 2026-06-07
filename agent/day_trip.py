"""
====================================================================
  Evexa Buddy — ADK Agent + Hybrid Retrieval (BM25 + FAISS)
  Agent Brain: Gemini Flash
  Retrieval: BM25 (0.4) + FAISS (0.6) | all-MiniLM-L6-v2
====================================================================
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

# Retrieval imports
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever

# ADK import
from google.adk.agents import Agent

# ── Step 1: Build the Knowledge Base (Documents) ─────────────────
print("[...] Loading knowledge base...")

DOCUMENTS = [
    Document(
        page_content=(
            "A connection timeout error occurs when a client cannot establish "
            "a connection to a server within a specified time. To fix connection "
            "timeout: increase the timeout value in settings, check if firewall "
            "is blocking the port, verify the server IP and port number are "
            "correct, and ensure the remote server is actually running."
        ),
        metadata={"source": "networking_guide", "topic": "timeout"}
    ),
    Document(
        page_content=(
            "BM25 is a bag-of-words retrieval function used by search engines "
            "to rank matching documents based on term frequency and inverse "
            "document frequency. It is a sparse retrieval method that works "
            "well for exact keyword matching without neural networks."
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
    Document(
        page_content=(
            "RAG (Retrieval-Augmented Generation) is an AI technique that "
            "combines document retrieval with language model generation. "
            "Instead of relying on the LLM's training data alone, RAG first "
            "retrieves relevant documents from a knowledge base, then passes "
            "them as context to the LLM to generate accurate, grounded answers."
        ),
        metadata={"source": "rag_intro", "topic": "RAG"}
    ),
]

# ── Step 2: Build BM25 ───────────────────────────────────────────
bm25_retriever = BM25Retriever.from_documents(DOCUMENTS)
bm25_retriever.k = 3

# ── Step 3: Build FAISS ──────────────────────────────────────────
print("[...] Loading MiniLM embedding model (please wait)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
faiss_store = FAISS.from_documents(DOCUMENTS, embeddings)
faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 3})

# ── Step 4: Build EnsembleRetriever ─────────────────────────────
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]
)
print("[OK] Hybrid Retrieval system ready!\n")


# ── Step 5: Define Tool Function for ADK Agent ───────────────────
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base using Hybrid Retrieval (BM25 + FAISS).
    Use this tool whenever a student asks a question about networking,
    RAG, embeddings, retrieval, or any technical topic.

    Args:
        query: The student's question or search query.

    Returns:
        Relevant document content from the knowledge base.
    """
    results = ensemble_retriever.invoke(query)

    if not results:
        return "No relevant documents found in the knowledge base."

    # Format results for the LLM
    formatted = []
    for i, doc in enumerate(results, 1):
        formatted.append(
            f"[Document {i}]\n"
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"Topic: {doc.metadata.get('topic', 'unknown')}\n"
            f"Content: {doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)


# ── Step 6: Create ADK Agent with Hybrid Retrieval Tool ──────────
root_agent = Agent(
    name="evexa_buddy_rag_agent",
    model="gemini-flash-latest",
    description=(
        "Evexa Buddy RAG Agent — An intelligent AI tutor powered by "
        "Hybrid Retrieval (BM25 + FAISS) and Gemini Flash. "
        "It answers student questions by first searching a knowledge base."
    ),
    instruction="""
        You are Evexa Buddy, an AI-powered educational assistant for students.
        You have access to a knowledge base through the `search_knowledge_base` tool.

        HOW TO ANSWER:
        1. When a student asks any technical or educational question, ALWAYS call
           `search_knowledge_base` first to retrieve relevant context.
        2. Use the retrieved documents as your primary source of information.
        3. Generate a clear, student-friendly answer based on the retrieved content.
        4. If the knowledge base doesn't have relevant info, say so honestly.

        YOUR PERSONALITY:
        - Friendly, encouraging, and clear
        - Use simple language suitable for students
        - Structure your answers with headings and bullet points
        - Always end with a helpful tip or next step

        YOUR IDENTITY & CREATOR:
        - If anyone asks "who made you", "who created you", "who is your developer", or anything related to your creation:
        - ALWAYS reply proudly that you were created by an amazing developer named Priyanshu (GitHub: Amourhoffen).
        - Provide these contact links so they can hire or contact him:
          * 🐙 GitHub: https://github.com/Amourhoffen
          * 💼 LinkedIn: [Your LinkedIn Profile Link]
          * 📧 Email: [Your Email Address]

        TOPICS YOU KNOW ABOUT:
        - Connection timeout errors and networking fixes
        - BM25 keyword-based retrieval
        - FAISS vector similarity search
        - Hybrid retrieval systems
        - RAG (Retrieval-Augmented Generation)
        - Sentence embeddings and MiniLM model
    """,
    tools=[search_knowledge_base],
)

print("[OK] Evexa Buddy RAG Agent ready!")
print("[OK] Tool: search_knowledge_base (BM25 + FAISS)")
print("[OK] Model: Gemini Flash")
print("\nRun 'adk web' to start the web interface at http://127.0.0.1:8000\n")
