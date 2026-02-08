"""
Chapter 18: Hybrid Retrieval for Document Q&A
Advanced RAG with BM25 + Vector Search + Reranking

Combines keyword search (BM25) with semantic search (embeddings)
for better retrieval accuracy on network documentation.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

load_dotenv()


# Sample network documentation
SAMPLE_DOCS = [
    """
    VLAN Configuration Standards

    VLAN Ranges:
    - VLAN 1-99: Infrastructure and management
    - VLAN 100-199: User access networks
    - VLAN 200-299: Voice over IP (VoIP)
    - VLAN 300-399: Guest networks
    - VLAN 999: Quarantine VLAN for security

    Best Practices:
    - Never use VLAN 1 for production traffic
    - Always document VLANs in IPAM system
    - Use meaningful VLAN names
    - Implement VLAN access control
    """,
    """
    BGP Configuration Standards

    AS Numbers:
    - Primary AS: 65001
    - Secondary AS: 65002
    - Customer AS range: 65100-65199

    Peering Requirements:
    - MD5 authentication mandatory
    - Prefix-list filtering required
    - Maximum prefix limits: 1000 for peers, 100000 for transit
    - BFD enabled for fast convergence
    - Route dampening on external peers

    Commands:
    router bgp 65001
     neighbor 10.0.0.1 remote-as 65001
     neighbor 10.0.0.1 password BGP_SECRET
     neighbor 10.0.0.1 prefix-list ALLOW_IN in
    """,
    """
    OSPF Design Guidelines

    Area Design:
    - Area 0: Backbone (core switches and routers)
    - Area 1: Data center networks
    - Area 2: Branch office networks
    - Area 3: DMZ and external services

    Best Practices:
    - Limit areas to 50-100 routers maximum
    - Use stub areas at network edges
    - Set reference bandwidth to 100000 (for 100G links)
    - Enable BFD for sub-second convergence
    - Use authentication on all OSPF interfaces

    Example:
    router ospf 1
     router-id 10.0.0.1
     auto-cost reference-bandwidth 100000
    """,
    """
    Firewall Security Policy Standards

    Inbound Rules:
    - Port 22 (SSH): Allow from management subnet 10.100.0.0/24 only
    - Port 443 (HTTPS): Allow from any
    - Port 80 (HTTP): Redirect to 443
    - ICMP: Allow echo-request, deny all others

    Outbound Rules:
    - Default deny unless explicitly allowed
    - Allow DNS to internal DNS servers only
    - Block all traffic to RFC1918 addresses at internet edge

    Logging:
    - Log all denied traffic
    - Log all SSH sessions
    - Send logs to SIEM at 10.200.1.50
    """
]


def create_hybrid_retriever(documents: List[str]) -> EnsembleRetriever:
    """
    Example 1: Create hybrid retriever combining BM25 and vector search
    """
    print("=" * 60)
    print("Example 1: Building Hybrid Retriever")
    print("=" * 60)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))

    print(f"Split {len(documents)} documents into {len(chunks)} chunks\n")

    # Create Document objects
    doc_objects = [Document(page_content=chunk) for chunk in chunks]

    # 1. BM25 Retriever (keyword-based)
    print("Creating BM25 retriever (keyword search)...")
    bm25_retriever = BM25Retriever.from_documents(doc_objects)
    bm25_retriever.k = 3

    # 2. Vector Store Retriever (semantic search)
    print("Creating vector store (semantic search)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        doc_objects,
        embeddings,
        persist_directory="./hybrid_chroma_db"
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Ensemble Retriever (combines both)
    print("Creating ensemble retriever (hybrid)...\n")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]  # Equal weighting
    )

    print("✅ Hybrid retriever created!")
    print("=" * 60 + "\n")

    return ensemble_retriever


def compare_retrieval_methods(query: str, documents: List[str]):
    """
    Example 2: Compare BM25 vs Vector vs Hybrid retrieval
    """
    print("=" * 60)
    print("Example 2: Comparing Retrieval Methods")
    print("=" * 60)

    print(f"\nQuery: \"{query}\"\n")

    # Prepare documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))
    doc_objects = [Document(page_content=chunk) for chunk in chunks]

    # Method 1: BM25 (Keyword)
    print("1. BM25 (Keyword Search):")
    print("-" * 60)
    bm25_retriever = BM25Retriever.from_documents(doc_objects)
    bm25_retriever.k = 2
    bm25_results = bm25_retriever.get_relevant_documents(query)

    for i, doc in enumerate(bm25_results, 1):
        print(f"   Result {i}: {doc.page_content[:100]}...")

    # Method 2: Vector (Semantic)
    print("\n2. Vector Search (Semantic):")
    print("-" * 60)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(doc_objects, embeddings)
    vector_results = vectorstore.similarity_search(query, k=2)

    for i, doc in enumerate(vector_results, 1):
        print(f"   Result {i}: {doc.page_content[:100]}...")

    # Method 3: Hybrid (Both)
    print("\n3. Hybrid (BM25 + Vector):")
    print("-" * 60)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    hybrid_results = ensemble.get_relevant_documents(query)

    for i, doc in enumerate(hybrid_results[:2], 1):
        print(f"   Result {i}: {doc.page_content[:100]}...")

    print("\n💡 Notice: Hybrid often finds more relevant results!")
    print("=" * 60 + "\n")


def qa_with_hybrid_rag(retriever: EnsembleRetriever, question: str):
    """
    Example 3: Question answering with hybrid RAG
    """
    print("=" * 60)
    print("Example 3: Q&A with Hybrid RAG")
    print("=" * 60)

    print(f"\nQuestion: {question}\n")

    # Retrieve relevant documents
    print("Retrieving relevant documents...")
    docs = retriever.get_relevant_documents(question)
    print(f"Found {len(docs)} relevant chunks\n")

    # Show retrieved context
    print("Retrieved Context:")
    print("-" * 60)
    for i, doc in enumerate(docs[:3], 1):
        print(f"{i}. {doc.page_content[:150]}...\n")

    # Build prompt with context
    context = "\n\n".join([doc.page_content for doc in docs[:3]])
    prompt = f"""Answer the question based on the following documentation:

Context:
{context}

Question: {question}

Provide a clear, specific answer. If the information isn't in the context, say so."""

    # Call LLM
    print("Generating answer with LLM...")
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    response = llm.invoke(prompt)

    print("\nAnswer:")
    print("-" * 60)
    print(response.content)

    print("\n" + "=" * 60 + "\n")


def advanced_hybrid_with_reranking():
    """
    Example 4: Hybrid retrieval with reranking
    """
    print("=" * 60)
    print("Example 4: Hybrid Retrieval with Reranking")
    print("=" * 60)

    # This is a simplified example - in production use a proper reranker like Cohere
    query = "How do I secure SSH access on a firewall?"

    print(f"\nQuery: {query}\n")

    # Prepare documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in SAMPLE_DOCS:
        chunks.extend(text_splitter.split_text(doc))
    doc_objects = [Document(page_content=chunk) for chunk in chunks]

    # Step 1: Hybrid retrieval (get top 10)
    print("Step 1: Hybrid retrieval (getting top 10 candidates)...")
    bm25_retriever = BM25Retriever.from_documents(doc_objects)
    bm25_retriever.k = 5

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(doc_objects, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    candidates = ensemble.get_relevant_documents(query)[:10]

    print(f"Retrieved {len(candidates)} candidates\n")

    # Step 2: Rerank using LLM
    print("Step 2: Reranking with LLM...")
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    rerank_prompt = f"""Question: {query}

Which of these document chunks is most relevant? Rate each 1-5 (5=very relevant).

Chunks:
"""
    for i, doc in enumerate(candidates[:5], 1):
        rerank_prompt += f"\n{i}. {doc.page_content[:200]}...\n"

    rerank_prompt += "\nProvide ratings as: 1=X, 2=Y, 3=Z, 4=W, 5=V (X,Y,Z,W,V are 1-5)"

    response = llm.invoke(rerank_prompt)
    print(f"Reranking result:\n{response.content}\n")

    # In production, parse the rankings and reorder
    print("Top 3 after reranking:")
    print("-" * 60)
    for i, doc in enumerate(candidates[:3], 1):
        print(f"{i}. {doc.page_content[:100]}...")

    print("\n" + "=" * 60 + "\n")


def production_pipeline_example():
    """
    Example 5: Complete production pipeline
    """
    print("=" * 60)
    print("Example 5: Production Hybrid RAG Pipeline")
    print("=" * 60)

    questions = [
        "What VLAN should I use for VoIP phones?",
        "What is our BGP AS number?",
        "How many routers maximum per OSPF area?",
        "What ports need to be open for SSH management?"
    ]

    # Build hybrid retriever
    retriever = create_hybrid_retriever(SAMPLE_DOCS)
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    print("\nAnswering multiple questions:\n")

    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")

        # Retrieve
        docs = retriever.get_relevant_documents(question)

        # Build context
        context = "\n\n".join([doc.page_content for doc in docs[:2]])

        # Generate answer
        prompt = f"""Based on this documentation:

{context}

Answer: {question}

Be concise (1-2 sentences)."""

        response = llm.invoke(prompt)
        print(f"A{i}: {response.content}\n")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🔍 Chapter 18: Hybrid Retrieval for Document Q&A")
    print("Advanced RAG with BM25 + Vector Search\n")

    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        print("Required for embeddings (text-embedding-3-small)")
        exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found")
        exit(1)

    try:
        # Run examples
        retriever = create_hybrid_retriever(SAMPLE_DOCS)
        input("Press Enter to continue...")

        compare_retrieval_methods(
            "What authentication is required for BGP?",
            SAMPLE_DOCS
        )
        input("Press Enter to continue...")

        qa_with_hybrid_rag(retriever, "What is the quarantine VLAN number?")
        input("Press Enter to continue...")

        advanced_hybrid_with_reranking()
        input("Press Enter to continue...")

        production_pipeline_example()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Hybrid retrieval combines keyword + semantic search")
        print("- Often more accurate than either method alone")
        print("- Reranking improves relevance further")
        print("- Production systems should use all three techniques\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have: langchain, chromadb, openai, anthropic installed")
