# Chapter 16: Building Document Retrieval with RAG

## Why This Chapter Matters

You have 1,000 pages of network documentation. A engineer asks: "What's our BGP peering policy for AWS?"

**Without AI**: Search docs for 20 minutes, find 5 relevant pages, read them all, synthesize answer.

**With RAG**: Ask the question, get accurate answer in 5 seconds with source citations.

**RAG = Retrieval Augmented Generation**:
1. **Retrieval**: Find relevant docs
2. **Augmented**: Add them to prompt
3. **Generation**: AI answers using those docs

This chapter builds a production documentation Q&A system that:
- Ingests PDFs, Word docs, wikis, configs
- Creates searchable embeddings
- Retrieves relevant context
- Answers questions with citations

**The networking equivalent**: RAG is like a routing table for knowledge. Instead of sending queries everywhere (flooding), you have an index (routing table) that points to exactly where the answer lives.

---

## Section 1: Understanding RAG

### The Problem RAG Solves

**LLMs without RAG**:
```python
response = llm.invoke("What's our BGP policy for AWS?")
# Output: "I don't have access to your specific BGP policies..."
```

The LLM doesn't know your internal documentation.

**LLMs with RAG**:
```python
# 1. Find relevant docs
docs = retriever.search("BGP policy AWS")
# Returns: [doc1: "BGP Peering Guide p.42", doc2: "AWS Integration doc p.12"]

# 2. Add docs to prompt
prompt = f"""Using these documents:
{docs}

Answer: What's our BGP policy for AWS?"""

# 3. Get informed answer
response = llm.invoke(prompt)
# Output: "According to the BGP Peering Guide (p.42), AWS peering requires..."
```

Now the LLM has your documentation as context.

### How RAG Works

**Step 1: Ingestion (one-time)**
```
Network Docs (PDF, Word, Wiki)
    ↓
Split into chunks (500-1000 tokens each)
    ↓
Generate embeddings (convert text to vectors)
    ↓
Store in vector database (ChromaDB, Pinecone, etc.)
```

**Step 2: Retrieval (every query)**
```
User question: "BGP policy for AWS?"
    ↓
Generate embedding for question
    ↓
Search vector DB for similar embeddings
    ↓
Return top N most relevant chunks
```

**Step 3: Generation (every query)**
```
Relevant chunks + user question
    ↓
Send to LLM with prompt
    ↓
LLM generates answer using retrieved docs
    ↓
Return answer + source citations
```

### Key Components

1. **Embeddings**: Convert text to vectors (arrays of numbers) that capture semantic meaning
2. **Vector Database**: Store and search embeddings efficiently
3. **Retriever**: Find relevant documents based on similarity
4. **LLM**: Generate answers using retrieved context

---

## Section 2: Building Your First RAG System

### Installation

```bash
pip install langchain langchain-anthropic
pip install chromadb sentence-transformers
pip install pypdf python-docx  # For document loading
```

### Simple RAG Example

```python
# simple_rag.py
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class SimpleNetworkRAG:
    """Basic RAG system for network documentation."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.0
        )

        # Embeddings model (runs locally)
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Vector store
        self.vectorstore = None

    def ingest_documents(self, documents: list[str]):
        """Ingest documents into vector store."""
        # Create vector store from documents
        self.vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

        print(f"Ingested {len(documents)} documents")

    def query(self, question: str, k: int = 3) -> dict:
        """Query the RAG system."""
        if not self.vectorstore:
            return {"error": "No documents ingested yet"}

        # Retrieve relevant documents
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        relevant_docs = retriever.get_relevant_documents(question)

        # Format context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Prompt template
        prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:""")

        # Build RAG chain
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Get answer
        answer = chain.invoke(question)

        return {
            "answer": answer,
            "sources": [doc.page_content[:100] + "..." for doc in relevant_docs],
            "num_sources": len(relevant_docs)
        }


# Example usage
if __name__ == "__main__":
    # Network documentation
    documents = [
        """BGP Peering Policy - AWS Integration
        All BGP peerings with AWS must use MD5 authentication.
        AS number for AWS connections: 64512
        Peering IP ranges: 169.254.0.0/16
        Required: BGP session monitoring with BFD enabled.
        Contact: network-ops@company.com for AWS peer requests.""",

        """OSPF Area Design
        Area 0 (backbone): Core routers only
        Area 1: Branch offices
        Area 2: Datacenter
        All areas must connect to Area 0.
        Stub areas allowed for branch offices.
        MTU must be 1500 across all OSPF areas.""",

        """VLAN Standards
        VLAN 1: Management (DO NOT USE for user traffic)
        VLAN 10-100: User VLANs
        VLAN 200-300: Server VLANs
        VLAN 500-600: Guest/IoT
        All VLANs require approval from NetOps team.""",

        """ACL Standard - Management Access
        Permit SSH (port 22) from 10.0.0.0/8 only
        Permit HTTPS (port 443) from 10.0.0.0/8 only
        Deny all other management traffic
        Log all denied attempts
        Review ACL logs weekly for suspicious activity."""
    ]

    # Create RAG system
    rag = SimpleNetworkRAG(api_key="your-api-key")

    # Ingest documents
    rag.ingest_documents(documents)

    # Query
    questions = [
        "What's our BGP policy for AWS?",
        "What VLAN should I use for servers?",
        "What authentication do we use for AWS BGP?",
        "What's our EIGRP policy?"  # Not in docs
    ]

    for question in questions:
        print(f"\nQ: {question}")
        result = rag.query(question)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['num_sources']} documents")
```

**Output**:
```
Q: What's our BGP policy for AWS?
A: For AWS BGP peering, you must use MD5 authentication. The AS number for AWS connections is 64512, and peering IP ranges use 169.254.0.0/16. BGP session monitoring with BFD must be enabled. For AWS peer requests, contact network-ops@company.com.
Sources: 3 documents

Q: What VLAN should I use for servers?
A: For servers, use VLANs in the 200-300 range according to our VLAN standards.
Sources: 3 documents

Q: What authentication do we use for AWS BGP?
A: MD5 authentication is required for all BGP peerings with AWS.
Sources: 3 documents

Q: What's our EIGRP policy?
A: I don't have that information.
Sources: 3 documents
```

---

## Section 3: Document Loading and Chunking

### Loading Multiple Document Types

```python
# document_loader.py
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List
from langchain.docstore.document import Document

class NetworkDocumentLoader:
    """Load and process network documentation."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF file."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Add metadata
        for doc in documents:
            doc.metadata['source_type'] = 'pdf'
            doc.metadata['filename'] = Path(file_path).name

        return documents

    def load_docx(self, file_path: str) -> List[Document]:
        """Load a Word document."""
        loader = Docx2txtLoader(file_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata['source_type'] = 'docx'
            doc.metadata['filename'] = Path(file_path).name

        return documents

    def load_text(self, file_path: str) -> List[Document]:
        """Load a text file."""
        loader = TextLoader(file_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata['source_type'] = 'text'
            doc.metadata['filename'] = Path(file_path).name

        return documents

    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory."""
        all_docs = []

        directory = Path(directory_path)

        # Load PDFs
        for pdf_file in directory.glob("**/*.pdf"):
            all_docs.extend(self.load_pdf(str(pdf_file)))

        # Load Word docs
        for docx_file in directory.glob("**/*.docx"):
            all_docs.extend(self.load_docx(str(docx_file)))

        # Load text files
        for txt_file in directory.glob("**/*.txt"):
            all_docs.extend(self.load_text(str(txt_file)))

        # Load markdown files
        for md_file in directory.glob("**/*.md"):
            all_docs.extend(self.load_text(str(md_file)))

        return all_docs

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunked = self.text_splitter.split_documents(documents)

        # Add chunk metadata
        for i, chunk in enumerate(chunked):
            chunk.metadata['chunk_id'] = i

        return chunked

    def load_and_chunk(self, directory_path: str) -> List[Document]:
        """Load directory and chunk all documents."""
        print(f"Loading documents from {directory_path}...")
        documents = self.load_directory(directory_path)
        print(f"Loaded {len(documents)} documents")

        print("Chunking documents...")
        chunks = self.chunk_documents(documents)
        print(f"Created {len(chunks)} chunks")

        return chunks


# Example usage
if __name__ == "__main__":
    loader = NetworkDocumentLoader(chunk_size=500, chunk_overlap=50)

    # Load and chunk all docs from a directory
    chunks = loader.load_and_chunk("./network_docs")

    # Print sample chunk
    if chunks:
        sample = chunks[0]
        print(f"\nSample chunk:")
        print(f"Content: {sample.page_content[:200]}...")
        print(f"Metadata: {sample.metadata}")
```

### Smart Chunking for Network Docs

```python
# network_doc_chunker.py
from langchain.text_splitter import TextSplitter
from typing import List
import re

class NetworkDocumentChunker(TextSplitter):
    """Custom chunker that respects network documentation structure."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks respecting structure."""
        chunks = []

        # Split by major sections (markdown headers)
        sections = re.split(r'\n#+\s+', text)

        current_chunk = ""

        for section in sections:
            # If section is small enough, add to current chunk
            if len(current_chunk) + len(section) < self.chunk_size:
                current_chunk += section
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk with section
                if len(section) > self.chunk_size:
                    # Section too large, split by paragraphs
                    paragraphs = section.split('\n\n')
                    for para in paragraphs:
                        if len(current_chunk) + len(para) < self.chunk_size:
                            current_chunk += para + "\n\n"
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = para + "\n\n"
                else:
                    current_chunk = section

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
```

---

## Section 4: Vector Stores and Embeddings

### Understanding Embeddings

**Text** → **Embedding** (vector of numbers)

```python
"BGP routing protocol" → [0.23, -0.45, 0.67, ..., 0.12]  # 384 numbers
"Border Gateway Protocol" → [0.21, -0.43, 0.69, ..., 0.15]  # Similar!
"VLAN configuration" → [-0.52, 0.31, -0.22, ..., 0.88]  # Different!
```

Similar meaning = similar vectors. This lets us find relevant docs.

### Choosing an Embedding Model

```python
# embedding_comparison.py
from langchain_community.embeddings import (
    SentenceTransformerEmbeddings,
    HuggingFaceEmbeddings
)
from langchain_openai import OpenAIEmbeddings
import time

class EmbeddingBenchmark:
    """Compare embedding models for network docs."""

    def __init__(self):
        self.models = {
            "MiniLM": SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            ),  # Fast, small, good enough
            "MPNet": SentenceTransformerEmbeddings(
                model_name="all-mpnet-base-v2"
            ),  # Better quality, slower
            # "OpenAI": OpenAIEmbeddings()  # Costs money, best quality
        }

    def benchmark(self, texts: List[str]):
        """Benchmark different embedding models."""
        print(f"Benchmarking with {len(texts)} texts\n")

        for name, model in self.models.items():
            start = time.time()

            # Generate embeddings
            embeddings = model.embed_documents(texts)

            elapsed = time.time() - start

            print(f"{name}:")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Dimension: {len(embeddings[0])}")
            print(f"  Speed: {len(texts)/elapsed:.1f} docs/sec\n")


# Recommendation for network docs
"""
Use all-MiniLM-L6-v2 for most cases:
- Fast (100+ docs/sec)
- Small (80MB model)
- Good quality for technical docs
- Runs locally (no API costs)

Use all-mpnet-base-v2 if you need better accuracy:
- Slower (30-50 docs/sec)
- Larger (400MB model)
- Best open-source quality

Use OpenAI embeddings if budget allows:
- Best quality
- Costs $0.0001 per 1K tokens
- Requires API call (latency)
"""
```

### Vector Store Options

```python
# vector_store_comparison.py
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Option 1: ChromaDB (recommended for most cases)
"""
Pros:
- Easy to use
- Persistent storage
- Good performance up to 1M documents
- Built-in metadata filtering

Cons:
- Not for massive scale (>10M docs)
"""
vectorstore_chroma = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Option 2: FAISS (for large scale)
"""
Pros:
- Extremely fast
- Handles billions of vectors
- Facebook-backed

Cons:
- More complex setup
- In-memory (need to save/load manually)
"""
vectorstore_faiss = FAISS.from_documents(
    documents=[],
    embedding=embeddings
)
vectorstore_faiss.save_local("./faiss_index")

# Option 3: Pinecone, Weaviate, Qdrant (cloud services)
"""
Pros:
- Fully managed
- Scalable
- Good for production

Cons:
- Costs money
- Requires internet connection
"""

# Recommendation: Start with ChromaDB, move to FAISS if you hit scale issues
```

---

## Section 5: Production RAG System

### Complete RAG with Metadata Filtering

```python
# production_rag.py
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from document_loader import NetworkDocumentLoader
from typing import List, Dict, Optional
import json

class ProductionNetworkRAG:
    """Production-grade RAG system for network documentation."""

    def __init__(self, api_key: str, persist_directory: str = "./chroma_db"):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.0
        )

        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.persist_directory = persist_directory
        self.vectorstore = None
        self.loader = NetworkDocumentLoader()

    def ingest_directory(self, directory_path: str):
        """Ingest all documents from a directory."""
        # Load and chunk documents
        chunks = self.loader.load_and_chunk(directory_path)

        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(chunks)

        # Persist to disk
        self.vectorstore.persist()

        print(f"Ingested {len(chunks)} chunks")

    def load_existing(self):
        """Load existing vector store from disk."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print("Loaded existing vector store")

    def query(
        self,
        question: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """Query with metadata filtering."""
        if not self.vectorstore:
            return {"error": "No vector store loaded"}

        # Build retriever with optional filtering
        search_kwargs = {"k": k}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata

        retriever = self.vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )

        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(question)

        if not relevant_docs:
            return {
                "answer": "No relevant documentation found.",
                "sources": [],
                "confidence": "low"
            }

        # Format context with metadata
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(
                f"[Source {i+1}: {source}, page {page}]\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        # Prompt with citation requirements
        prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following documentation.
Always cite sources using [Source N] notation.
If information is not in the docs, say "I don't have that information."

Documentation:
{context}

Question: {question}

Answer (with citations):""")

        # RAG chain
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

        # Extract sources
        sources = []
        for doc in relevant_docs:
            sources.append({
                "filename": doc.metadata.get('filename', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "chunk_id": doc.metadata.get('chunk_id', 'N/A'),
                "content_preview": doc.page_content[:150] + "..."
            })

        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
            "confidence": "high" if len(relevant_docs) >= 3 else "medium"
        }

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:
        """Direct similarity search without LLM generation."""
        if not self.vectorstore:
            return []

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score)
            }
            for doc, score in results
        ]


# Example usage
if __name__ == "__main__":
    rag = ProductionNetworkRAG(api_key="your-api-key")

    # Ingest documents (one-time)
    # rag.ingest_directory("./network_docs")

    # Or load existing
    rag.load_existing()

    # Query with citations
    result = rag.query(
        "What are the requirements for AWS BGP peering?",
        k=3
    )

    print(f"Answer: {result['answer']}\n")
    print(f"Confidence: {result['confidence']}")
    print(f"\nSources ({result['num_sources']}):")
    for source in result['sources']:
        print(f"- {source['filename']} (page {source['page']})")

    # Filter by document type
    result = rag.query(
        "VLAN standards",
        k=3,
        filter_metadata={"source_type": "pdf"}
    )
```

---

## What Can Go Wrong

**1. Chunking too small/large**
- Too small (< 200 tokens): Loses context
- Too large (> 2000 tokens): Dilutes relevance
- Sweet spot: 500-1000 tokens with 50-100 overlap

**2. Wrong embedding model**
- Generic embeddings don't understand networking terms
- Consider fine-tuning embeddings on your docs

**3. Not enough documents retrieved**
- Retrieve more chunks (k=5-10) for complex questions
- Too few = incomplete answers

**4. Stale documentation**
- RAG doesn't know docs are outdated
- Add timestamp metadata, prefer recent docs

**5. Metadata ignored**
- Not using document type, author, date for filtering
- Leads to irrelevant results

---

## Key Takeaways

1. **RAG grounds AI in your documentation** - No more hallucinated answers
2. **Embeddings capture semantic meaning** - Find relevant docs by similarity
3. **Chunking strategy matters** - Balance context vs relevance
4. **Metadata enables filtering** - Filter by doc type, date, author
5. **Citations build trust** - Always show sources

Next chapter: Advanced RAG patterns and semantic search.
