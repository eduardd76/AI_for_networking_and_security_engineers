#!/usr/bin/env python3
"""
Simple RAG System for Network Documentation

A clean, straightforward implementation of Retrieval Augmented Generation
for network documentation using LangChain and ChromaDB.

From: AI for Networking Engineers - Volume 2, Chapter 14
Author: Eduard Dulharu

Usage:
    python simple_rag.py
"""

import os
from dotenv import load_dotenv

# LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()


class NetworkRAG:
    """
    Simple RAG system for network documentation.

    This class handles:
    - Adding network documentation
    - Searching for relevant docs
    - Answering questions using RAG
    """

    def __init__(self, persist_dir="./chroma_db"):
        """Initialize the RAG system."""
        # Setup embeddings (converts text to vectors)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        # Setup vector database
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )

        # Setup LLM
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0
        )

        # Setup text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def add_documents(self, texts, metadatas=None):
        """
        Add documents to the knowledge base.

        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dicts
        """
        # Split texts into chunks
        chunks = []
        chunk_metadatas = []

        for i, text in enumerate(texts):
            text_chunks = self.text_splitter.split_text(text)
            chunks.extend(text_chunks)

            # Add metadata for each chunk
            if metadatas and i < len(metadatas):
                chunk_metadatas.extend([metadatas[i]] * len(text_chunks))
            else:
                chunk_metadatas.extend([{"source": f"doc_{i}"}] * len(text_chunks))

        # Add to vector store
        self.vectorstore.add_texts(chunks, metadatas=chunk_metadatas)
        print(f"✓ Added {len(chunks)} chunks from {len(texts)} documents")

    def search(self, query, k=3):
        """
        Search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant document chunks
        """
        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def ask(self, question):
        """
        Ask a question and get an answer using RAG.

        Args:
            question: Question to answer

        Returns:
            Answer string
        """
        # Create a custom prompt
        prompt_template = """Use the following network documentation to answer the question.
If you cannot find the answer in the documentation, say "I don't have enough information to answer that."

Documentation:
{context}

Question: {question}

Answer (be specific and cite the documentation):"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        # Get answer
        result = qa_chain({"query": question})

        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }


def main():
    """Demo the RAG system with network documentation."""

    print("="*60)
    print("Simple RAG System for Network Documentation")
    print("="*60)

    # Initialize RAG
    rag = NetworkRAG(persist_dir="./demo_chroma")

    # Sample network documentation
    docs = [
        """
        VLAN Configuration Guide

        VLANs (Virtual LANs) segment network traffic at Layer 2.

        Best Practices:
        - Use VLAN 1 only for management
        - Data VLANs: 10-999
        - Voice VLANs: 100-199
        - Guest WiFi: VLAN 99

        Configuration Example (Cisco):
        vlan 10
          name USERS
        vlan 20
          name SERVERS
        vlan 99
          name GUEST_WIFI

        Trunk Configuration:
        interface GigabitEthernet0/1
          switchport mode trunk
          switchport trunk allowed vlan 10,20,99
        """,

        """
        BGP Configuration Standards

        Border Gateway Protocol (BGP) is used for internet routing.

        Our Standards:
        - AS Number: 65001
        - eBGP with ISPs
        - iBGP within our network

        Required Configuration:
        1. Set router-id explicitly
        2. Use BGP authentication (MD5)
        3. Configure prefix-lists for filtering
        4. Set maximum-prefix limits

        Example:
        router bgp 65001
          router-id 10.0.0.1
          neighbor 203.0.113.1 remote-as 174
          neighbor 203.0.113.1 password 7 <encrypted>
          neighbor 203.0.113.1 maximum-prefix 1000
        """,

        """
        OSPF Configuration Guide

        OSPF (Open Shortest Path First) for internal routing.

        Design:
        - Area 0 (backbone): Core routers
        - Area 1: Branch offices
        - Area 2: Data centers

        Configuration Steps:
        1. Enable OSPF process
        2. Assign router-id
        3. Configure networks per area
        4. Set passive interfaces where needed

        Example:
        router ospf 1
          router-id 10.0.0.1
          network 10.0.0.0 0.0.0.255 area 0
          network 172.16.0.0 0.0.255.255 area 1
          passive-interface GigabitEthernet0/0
        """
    ]

    # Add metadata
    metadata = [
        {"type": "vlan", "category": "switching"},
        {"type": "bgp", "category": "routing"},
        {"type": "ospf", "category": "routing"}
    ]

    # Add documents
    print("\n1. Adding documentation to knowledge base...")
    rag.add_documents(docs, metadata)

    # Test search
    print("\n2. Testing search...")
    query = "VLAN configuration"
    results = rag.search(query, k=2)
    print(f"\nQuery: '{query}'")
    print(f"Found {len(results)} relevant chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\n  Chunk {i}:")
        print(f"  {doc.page_content[:100]}...")

    # Test RAG questions
    print("\n" + "="*60)
    print("3. Testing RAG Question Answering")
    print("="*60)

    questions = [
        "What VLAN should I use for guest WiFi?",
        "What is our BGP AS number?",
        "What OSPF areas do we use?",
        "How do I configure a trunk port?"
    ]

    for question in questions:
        print(f"\nQ: {question}")
        result = rag.ask(question)
        print(f"A: {result['answer']}")
        print(f"   (Sources: {len(result['sources'])} documents)")

    # Cleanup
    print("\n" + "="*60)
    print("Demo complete!")
    print("\nNote: Run 'rm -rf ./demo_chroma' to clean up test database")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set ANTHROPIC_API_KEY and OPENAI_API_KEY in .env")
        print("  2. Installed requirements: pip install -r requirements.txt")
