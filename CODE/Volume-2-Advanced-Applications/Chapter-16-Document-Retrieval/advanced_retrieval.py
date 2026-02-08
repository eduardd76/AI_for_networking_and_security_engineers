#!/usr/bin/env python3
"""
Advanced Document Retrieval Techniques

Learn different strategies to retrieve the most relevant information
from your knowledge base.

From: AI for Networking Engineers - Volume 2, Chapter 16
Author: Eduard Dulharu

Usage:
    python advanced_retrieval.py
"""

import os
from dotenv import load_dotenv
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load environment variables
load_dotenv()


def create_sample_knowledge_base():
    """
    Create a sample network knowledge base for testing.

    Returns:
        Chroma vectorstore with network documentation
    """
    print("\n" + "="*60)
    print("Creating Sample Knowledge Base")
    print("="*60)

    # Sample network documentation
    docs = [
        """VLAN Troubleshooting Guide

        Common VLAN Issues:
        1. VLAN Mismatch - Trunk ports must allow the VLAN
        2. Native VLAN Mismatch - Can cause spanning tree issues
        3. VLAN Pruning - Check if VLAN is pruned on trunk
        4. VTP Issues - Version and domain mismatch

        Troubleshooting Steps:
        - show vlan brief (verify VLAN exists)
        - show interfaces trunk (check allowed VLANs)
        - show spanning-tree vlan X (check STP state)
        - show vtp status (verify VTP config)

        Common Fix: switchport trunk allowed vlan add X
        """,

        """BGP Peering Troubleshooting

        BGP Not Establishing:
        1. Check reachability - ping neighbor IP
        2. Verify AS numbers match configuration
        3. Check BGP is enabled (router bgp X)
        4. Verify neighbor statements
        5. Check ACLs blocking TCP 179

        Commands:
        - show ip bgp summary (check peer state)
        - show ip bgp neighbors X.X.X.X (detailed info)
        - debug ip bgp (see BGP messages)

        States: Idle -> Connect -> Active -> OpenSent -> OpenConfirm -> Established

        Common issues: Wrong AS number, firewall blocking, no route to neighbor
        """,

        """OSPF Adjacency Problems

        OSPF Not Forming Adjacency:
        1. Check network types match (broadcast, point-to-point)
        2. Verify hello/dead timers match
        3. Check area numbers match
        4. Verify authentication (if configured)
        5. Check MTU size (must match for DBD exchange)

        Commands:
        - show ip ospf neighbor (check state)
        - show ip ospf interface (verify config)
        - debug ip ospf adj (see adjacency process)

        States: Down -> Init -> 2-Way -> ExStart -> Exchange -> Loading -> Full

        Common fix: ip ospf network point-to-point
        """,

        """Spanning Tree Troubleshooting

        STP Issues:
        1. Root bridge location - Should be core switches
        2. Loops causing broadcast storms
        3. Convergence time too slow
        4. PortFast on trunk ports (dangerous!)

        Commands:
        - show spanning-tree (check root bridge)
        - show spanning-tree interface X (port state)
        - show spanning-tree summary (quick overview)

        Best Practices:
        - Set root bridge manually (spanning-tree vlan X priority)
        - Use RSTP or MST for faster convergence
        - Enable PortFast only on access ports
        - Enable BPDUGuard on access ports

        Modes: PVST+ (Cisco), RSTP (802.1w), MST (802.1s)
        """,

        """Interface Errors and Troubleshooting

        Common Interface Errors:
        1. CRC Errors - Physical layer issues (bad cable, EMI)
        2. Input Errors - Frames with errors
        3. Collisions - Half-duplex or speed mismatch
        4. Runts - Frames smaller than 64 bytes
        5. Giants - Frames larger than MTU

        Commands:
        - show interfaces X (detailed errors)
        - show interfaces status (quick overview)
        - show controllers X (physical layer)

        Speed/Duplex Issues:
        - Always set both sides or use auto on both
        - Never set one side auto and other side manual
        - GigabitEthernet should use auto-negotiation

        Common fixes: Replace cable, clean fiber, check duplex settings
        """
    ]

    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    splits = []
    for doc in docs:
        splits.extend(text_splitter.split_text(doc))

    # Create vector store
    vectorstore = Chroma.from_texts(
        splits,
        embeddings,
        persist_directory="./retrieval_demo_db"
    )

    print(f"\n✓ Created knowledge base with {len(splits)} chunks")
    return vectorstore


def example_1_basic_retrieval(vectorstore):
    """
    Example 1: Basic Similarity Search

    Simple retrieval based on vector similarity.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Similarity Search")
    print("="*60)

    query = "BGP is not establishing"

    # Basic similarity search
    docs = vectorstore.similarity_search(query, k=2)

    print(f"\nQuery: {query}")
    print(f"\nFound {len(docs)} relevant chunks:\n")

    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content[:200]}...")
        print()


def example_2_mmr_retrieval(vectorstore):
    """
    Example 2: Maximum Marginal Relevance (MMR)

    Retrieves diverse results, not just most similar.
    """
    print("\n" + "="*60)
    print("Example 2: MMR (Diverse Results)")
    print("="*60)

    query = "troubleshooting network connectivity"

    # MMR retrieval - balances relevance with diversity
    docs = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=10)

    print(f"\nQuery: {query}")
    print(f"\nMMR returned {len(docs)} diverse chunks:\n")

    for i, doc in enumerate(docs, 1):
        # Extract topic from first line
        topic = doc.page_content.split('\n')[0].strip()
        print(f"{i}. {topic}")
        print(f"   Preview: {doc.page_content[:150]}...")
        print()


def example_3_multi_query(vectorstore):
    """
    Example 3: Multi-Query Retrieval

    Generates multiple queries from user's question to improve retrieval.
    """
    print("\n" + "="*60)
    print("Example 3: Multi-Query Retrieval")
    print("="*60)

    # Create LLM
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    # Create multi-query retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        llm=llm
    )

    question = "Why won't my devices communicate across VLANs?"

    print(f"\nOriginal question: {question}")
    print("\nMulti-query retriever will:")
    print("1. Generate variations of the question")
    print("2. Search for each variation")
    print("3. Combine and deduplicate results")

    # Retrieve documents
    docs = retriever.get_relevant_documents(question)

    print(f"\n✓ Found {len(docs)} unique relevant chunks")
    print("\nMost relevant content:")
    print(docs[0].page_content[:300] + "...")


def example_4_contextual_compression(vectorstore):
    """
    Example 4: Contextual Compression

    Compresses retrieved documents to only the most relevant parts.
    """
    print("\n" + "="*60)
    print("Example 4: Contextual Compression")
    print("="*60)

    # Create LLM
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    # Create compressor
    compressor = LLMChainExtractor.from_llm(llm)

    # Create compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    question = "What commands check OSPF neighbors?"

    print(f"\nQuestion: {question}")
    print("\n1. Basic retrieval (might include irrelevant text):")

    # Basic retrieval
    basic_docs = vectorstore.similarity_search(question, k=1)
    print(f"   Length: {len(basic_docs[0].page_content)} characters")
    print(f"   Preview: {basic_docs[0].page_content[:200]}...")

    print("\n2. With compression (extracts only relevant parts):")

    # Compressed retrieval
    compressed_docs = compression_retriever.get_relevant_documents(question)
    print(f"   Length: {len(compressed_docs[0].page_content)} characters")
    print(f"   Relevant parts: {compressed_docs[0].page_content}")


def main():
    """Demo all retrieval techniques."""
    print("="*60)
    print("Advanced Document Retrieval Techniques")
    print("="*60)

    try:
        # Create knowledge base
        vectorstore = create_sample_knowledge_base()

        # Run examples
        example_1_basic_retrieval(vectorstore)
        example_2_mmr_retrieval(vectorstore)
        example_3_multi_query(vectorstore)
        example_4_contextual_compression(vectorstore)

        print("\n" + "="*60)
        print("✓ All retrieval techniques demonstrated!")
        print("="*60)
        print("\nKey Takeaways:")
        print("1. Basic similarity - Fast, works for clear queries")
        print("2. MMR - Better when you want diverse results")
        print("3. Multi-query - Better for ambiguous questions")
        print("4. Compression - Reduces noise, focuses on relevant parts")
        print("\nCleanup: rm -rf ./retrieval_demo_db")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set ANTHROPIC_API_KEY in .env")
        print("  2. Set OPENAI_API_KEY in .env")
        print("  3. Installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
