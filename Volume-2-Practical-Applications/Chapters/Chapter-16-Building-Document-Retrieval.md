# Chapter 16: Building Document Retrieval with RAG

## Why This Chapter Matters

You have 1,000 pages of network documentation. An engineer asks: "What's our BGP peering policy for AWS?"

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
Ingested 4 documents

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

### Progressive Build: RAG System V1 → V4

The example above works, but it's missing persistence, real document loading, and production features. Let's build it incrementally.

#### V1: In-Memory RAG (No Persistence)

Start with the simplest possible RAG: hardcoded documents in memory.

```python
# rag_v1.py - Minimal RAG
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

class RAGV1:
    """V1: In-memory RAG with hardcoded docs."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.0
        )

        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # In-memory only (no persist_directory)
        self.vectorstore = None

    def ingest(self, docs: list[str]):
        """Ingest documents into memory."""
        self.vectorstore = Chroma.from_texts(
            texts=docs,
            embedding=self.embeddings
            # No persist_directory = in-memory only
        )
        print(f"Ingested {len(docs)} documents (in memory)")

    def query(self, question: str) -> str:
        """Query the RAG system."""
        if not self.vectorstore:
            return "No documents loaded"

        # Retrieve
        docs = self.vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        # Generate
        prompt = f"""Answer based on this context:
{context}

Question: {question}
Answer:"""

        response = self.llm.invoke(prompt)
        return response.content


# Test it
if __name__ == "__main__":
    rag = RAGV1(api_key="your-api-key")

    docs = [
        "BGP with AWS requires MD5 auth and AS 64512",
        "VLANs 200-300 are for servers",
        "OSPF Area 0 is backbone only"
    ]

    rag.ingest(docs)
    print(rag.query("What VLAN for servers?"))
    # Output: "VLANs 200-300 are for servers"

    # Restart the script...
    # rag.query("What VLAN for servers?")
    # Error: No documents loaded!
```

**Works**: Retrieves and answers questions.

**Problem**: Data lost on restart. No persistence.

---

#### V2: Add Persistence

Add ChromaDB persistence so data survives restarts.

```python
# rag_v2.py - Add persistence
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_anthropic import ChatAnthropic

class RAGV2:
    """V2: Persistent storage with ChromaDB."""

    def __init__(self, api_key: str, persist_dir: str = "./chroma_db"):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.0
        )

        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.persist_dir = persist_dir
        self.vectorstore = None

    def ingest(self, docs: list[str]):
        """Ingest and persist documents."""
        self.vectorstore = Chroma.from_texts(
            texts=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir  # Save to disk
        )

        # Explicitly persist
        self.vectorstore.persist()

        print(f"Ingested {len(docs)} documents (saved to {self.persist_dir})")

    def load_existing(self):
        """Load previously ingested documents."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

        count = self.vectorstore._collection.count()
        print(f"Loaded {count} documents from disk")

    def query(self, question: str) -> str:
        """Query the RAG system."""
        if not self.vectorstore:
            return "No documents loaded. Call load_existing() or ingest() first."

        docs = self.vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""Answer based on context:
{context}

Question: {question}"""

        response = self.llm.invoke(prompt)
        return response.content


# First run: Ingest
if __name__ == "__main__":
    rag = RAGV2(api_key="your-api-key")

    # Option 1: Ingest new docs
    docs = [
        "BGP with AWS requires MD5 auth and AS 64512",
        "VLANs 200-300 are for servers",
        "OSPF Area 0 is backbone only"
    ]
    rag.ingest(docs)

    # Query works
    print(rag.query("What VLAN for servers?"))

    # --- Restart script ---

    # Option 2: Load existing docs
    rag2 = RAGV2(api_key="your-api-key")
    rag2.load_existing()

    # Query still works!
    print(rag2.query("What VLAN for servers?"))
```

**Output**:
```
# First run
Ingested 3 documents (saved to ./chroma_db)
VLANs 200-300 are for servers

# After restart
Loaded 3 documents from disk
VLANs 200-300 are for servers
```

**Better**: Survives restarts!

**Problem**: Still using hardcoded strings. Need to load real PDF/DOCX files.

---

#### V3: Add Document Loading

Load actual files instead of hardcoded strings.

```python
# rag_v3.py - Load real files
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_anthropic import ChatAnthropic
from pathlib import Path

class RAGV3:
    """V3: Load and chunk real documents."""

    def __init__(self, api_key: str, persist_dir: str = "./chroma_db"):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.0
        )

        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.persist_dir = persist_dir
        self.vectorstore = None

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def ingest_directory(self, docs_dir: str):
        """Load all PDFs/text files from directory."""
        docs_path = Path(docs_dir)

        all_docs = []

        # Load PDFs
        for pdf_file in docs_path.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            all_docs.extend(docs)

        # Load text files
        for txt_file in docs_path.glob("*.txt"):
            with open(txt_file) as f:
                content = f.read()
                all_docs.append(type('obj', (object,), {
                    'page_content': content,
                    'metadata': {'source': str(txt_file)}
                })())

        print(f"Loaded {len(all_docs)} documents from {docs_dir}")

        # Chunk documents
        chunks = self.splitter.split_documents(all_docs)
        print(f"Created {len(chunks)} chunks")

        # Store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        self.vectorstore.persist()

        print(f"Ingested {len(chunks)} chunks to {self.persist_dir}")

    def load_existing(self):
        """Load existing vector store."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

        count = self.vectorstore._collection.count()
        print(f"Loaded vector store with {count} chunks")

    def query(self, question: str) -> dict:
        """Query with source information."""
        if not self.vectorstore:
            return {"error": "No documents loaded"}

        # Retrieve
        docs = self.vectorstore.similarity_search(question, k=3)

        context = "\n\n".join([d.page_content for d in docs])

        # Generate
        prompt = f"""Answer based on context:
{context}

Question: {question}"""

        response = self.llm.invoke(prompt)

        # Return answer + sources
        return {
            "answer": response.content,
            "sources": [d.metadata.get('source', 'Unknown') for d in docs]
        }


# Usage
if __name__ == "__main__":
    rag = RAGV3(api_key="your-api-key")

    # Ingest from directory
    rag.ingest_directory("./network_docs")

    # Query
    result = rag.query("What's our BGP policy?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
```

**Output**:
```
Loaded 5 documents from ./network_docs
Created 47 chunks
Ingested 47 chunks to ./chroma_db

Answer: BGP with AWS requires MD5 authentication and AS number 64512
Sources: ['./network_docs/bgp_policy.pdf', './network_docs/bgp_policy.pdf', './network_docs/aws_integration.txt']
```

**Better**: Loads real files and shows sources!

**Problem**: No metadata filtering, no citations in answer, no error handling.

---

#### V4: Production-Ready

Add metadata, citations, filtering, and error handling.

```python
# rag_v4.py - Production ready
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGV4:
    """V4: Production RAG with metadata and citations."""

    def __init__(self, api_key: str, persist_dir: str = "./chroma_db"):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.0
        )

        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.persist_dir = persist_dir
        self.vectorstore = None

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def ingest_directory(self, docs_dir: str):
        """Load documents with rich metadata."""
        docs_path = Path(docs_dir)
        all_chunks = []

        # Load PDFs
        for pdf_file in docs_path.glob("**/*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()

                # Add metadata
                for doc in docs:
                    doc.metadata['source_type'] = 'pdf'
                    doc.metadata['filename'] = pdf_file.name

                chunks = self.splitter.split_documents(docs)
                all_chunks.extend(chunks)

                logger.info(f"Loaded {len(chunks)} chunks from {pdf_file.name}")

            except Exception as e:
                logger.error(f"Failed to load {pdf_file}: {e}")

        # Load text files
        for txt_file in docs_path.glob("**/*.txt"):
            try:
                with open(txt_file, encoding='utf-8') as f:
                    content = f.read()

                # Create document with metadata
                doc = type('obj', (object,), {
                    'page_content': content,
                    'metadata': {
                        'source': str(txt_file),
                        'source_type': 'text',
                        'filename': txt_file.name
                    }
                })()

                chunks = self.splitter.split_documents([doc])
                all_chunks.extend(chunks)

                logger.info(f"Loaded {len(chunks)} chunks from {txt_file.name}")

            except Exception as e:
                logger.error(f"Failed to load {txt_file}: {e}")

        if not all_chunks:
            logger.error("No documents loaded!")
            return

        # Store with metadata
        self.vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        self.vectorstore.persist()

        logger.info(f"Ingested {len(all_chunks)} total chunks")

    def load_existing(self):
        """Load existing vector store."""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )

            count = self.vectorstore._collection.count()
            logger.info(f"Loaded {count} chunks from disk")

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")

    def query(
        self,
        question: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """Query with metadata filtering and citations."""
        if not self.vectorstore:
            return {"error": "No vector store loaded"}

        try:
            # Build search kwargs
            search_kwargs = {"k": k}
            if filter_metadata:
                search_kwargs["filter"] = filter_metadata

            # Retrieve
            retriever = self.vectorstore.as_retriever(
                search_kwargs=search_kwargs
            )

            docs = retriever.get_relevant_documents(question)

            if not docs:
                return {
                    "answer": "No relevant documentation found.",
                    "sources": [],
                    "confidence": "low"
                }

            # Format context with source markers
            context_parts = []
            for i, doc in enumerate(docs):
                filename = doc.metadata.get('filename', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                context_parts.append(
                    f"[Source {i+1}: {filename}, page {page}]\n{doc.page_content}"
                )

            context = "\n\n".join(context_parts)

            # Prompt with citation requirement
            prompt = ChatPromptTemplate.from_template("""
Answer the question based on the documentation below.
ALWAYS cite sources using [Source N] notation.
If information is not in the docs, say "I don't have that information."

Documentation:
{context}

Question: {question}

Answer (with citations):""")

            # Generate
            formatted_prompt = prompt.format(context=context, question=question)
            response = self.llm.invoke(formatted_prompt)

            # Extract sources
            sources = []
            for doc in docs:
                sources.append({
                    "filename": doc.metadata.get('filename', 'Unknown'),
                    "page": doc.metadata.get('page', 'N/A'),
                    "content_preview": doc.page_content[:100] + "..."
                })

            return {
                "answer": response.content,
                "sources": sources,
                "num_sources": len(sources),
                "confidence": "high" if len(docs) >= 3 else "medium"
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"error": str(e)}


# Usage
if __name__ == "__main__":
    rag = RAGV4(api_key="your-api-key")

    # First time: Ingest
    # rag.ingest_directory("./network_docs")

    # Load existing
    rag.load_existing()

    # Query with metadata filter
    result = rag.query(
        "What's our BGP policy for AWS?",
        k=3,
        filter_metadata={"source_type": "pdf"}  # Only PDFs
    )

    print(f"Answer: {result['answer']}\n")
    print(f"Confidence: {result['confidence']}")
    print(f"\nSources:")
    for source in result['sources']:
        print(f"  - {source['filename']} (page {source['page']})")
```

**Output**:
```
INFO:__main__:Loaded 47 chunks from disk

Answer: For AWS BGP peering, you must use MD5 authentication [Source 1]. The AS number for AWS connections is 64512 [Source 1], and peering IP ranges use 169.254.0.0/16 [Source 2]. BGP session monitoring with BFD must be enabled [Source 1].

Confidence: high

Sources:
  - bgp_policy.pdf (page 3)
  - bgp_policy.pdf (page 5)
  - aws_integration.pdf (page 2)
```

**Production features**:
- ✓ Metadata filtering (PDFs only, text only, etc.)
- ✓ Citations in answers
- ✓ Error handling and logging
- ✓ Confidence scoring
- ✓ Source previews

**Evolution summary**:
- **V1**: In-memory (30 lines)
- **V2**: Persistent storage (50 lines)
- **V3**: Load real files (80 lines)
- **V4**: Production with metadata and citations (150 lines)

Start with V1 to learn. Build V4 for production.

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

**Output**:
```
Loading documents from ./network_docs...
Loaded 10 documents

Chunking documents...
Created 147 chunks

Sample chunk:
Content: BGP Peering Policy - AWS Integration

All BGP peerings with AWS must use MD5 authentication. This ensures that routing updates are authenticated and prevents unauthorized route injection.

Configuration requirements:
- AS number: 64512
- Peering IP: 169.254.x.x/30
- MD5 password: Minimum 16 characters
- BFD enabled for fast failover...

Metadata: {'source_type': 'pdf', 'filename': 'bgp_policy.pdf', 'page': 1, 'chunk_id': 0}
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

## Lab 0: Environment Setup

Before starting the labs, set up your RAG development environment.

### Prerequisites

- Python 3.9+
- Anthropic API key
- 5-10 network documentation files (PDFs, Word docs, or text files)

### Installation

```bash
# Create project directory
mkdir network-rag
cd network-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install langchain langchain-anthropic langchain-community
pip install chromadb sentence-transformers
pip install pypdf python-docx

# Verify installation
python -c "from langchain_community.vectorstores import Chroma; print('Success!')"
```

### Create Sample Documentation

If you don't have network docs, create sample files:

```bash
# Create docs directory
mkdir network_docs

# Create sample BGP policy
cat > network_docs/bgp_policy.txt << 'EOF'
BGP Peering Policy - AWS Integration

All BGP peerings with AWS must use MD5 authentication.
AS number for AWS connections: 64512
Peering IP ranges: 169.254.0.0/16
Required: BGP session monitoring with BFD enabled.
Contact: network-ops@company.com for AWS peer requests.
EOF

# Create sample VLAN standards
cat > network_docs/vlan_standards.txt << 'EOF'
VLAN Standards

VLAN 1: Management (DO NOT USE for user traffic)
VLAN 10-100: User VLANs
VLAN 200-300: Server VLANs
VLAN 500-600: Guest/IoT
All VLANs require approval from NetOps team.
EOF

# Create sample OSPF policy
cat > network_docs/ospf_design.txt << 'EOF'
OSPF Area Design

Area 0 (backbone): Core routers only
Area 1: Branch offices
Area 2: Datacenter
All areas must connect to Area 0.
Stub areas allowed for branch offices.
MTU must be 1500 across all OSPF areas.
EOF
```

### Test API Connection

```python
# test_api.py
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

response = llm.invoke("Say 'RAG API working!'")
print(response.content)
```

Run it:
```bash
python test_api.py
```

**Expected output**:
```
RAG API working!
```

**Troubleshooting**:
- "ModuleNotFoundError: No module named 'chromadb'" → `pip install chromadb`
- "AuthenticationError" → Check API key in .env file
- "sqlite3.OperationalError" → ChromaDB needs write permissions, run as user (not root)

---

## Lab 1: Build Your First RAG System

**Objective**: Create a working RAG system that answers questions about your network documentation.

**Time**: 30 minutes

**What You'll Learn**:
- Ingest documents into a vector database
- Perform semantic search
- Generate answers with LLM
- Understand retrieval vs generation

### Success Criteria

- [ ] Ingest 3+ network policy documents
- [ ] Successfully answer 5 questions with citations
- [ ] Understand why irrelevant questions return "I don't have that information"
- [ ] Total cost < $0.05

### Step 1: Create Basic RAG (10 minutes)

Create `lab1_basic_rag.py`:

```python
# lab1_basic_rag.py
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

class NetworkRAG:
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0
        )

        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.vectorstore = None

    def ingest_from_files(self, docs_dir: str):
        """Load text files and ingest."""
        from pathlib import Path

        texts = []
        for txt_file in Path(docs_dir).glob("*.txt"):
            with open(txt_file, encoding='utf-8') as f:
                texts.append(f.read())

        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            persist_directory="./lab1_db"
        )

        self.vectorstore.persist()
        print(f"Ingested {len(texts)} documents")

    def query(self, question: str) -> dict:
        """Query and return answer with sources."""
        if not self.vectorstore:
            return {"error": "No documents ingested"}

        # Retrieve
        docs = self.vectorstore.similarity_search(question, k=3)

        # Generate
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""Answer based ONLY on this context.
If not in context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "num_sources": len(docs)
        }


if __name__ == "__main__":
    rag = NetworkRAG()

    # Ingest
    rag.ingest_from_files("./network_docs")

    # Test questions
    questions = [
        "What's our BGP policy for AWS?",
        "What VLAN range should I use for servers?",
        "What's the OSPF backbone area?",
        "What authentication is needed for AWS BGP?",
        "What's our EIGRP policy?"  # Not in docs
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = rag.query(q)
        print(f"A: {result['answer']}")
        print(f"Sources used: {result['num_sources']}")
```

Run it:
```bash
python lab1_basic_rag.py
```

**Expected output**:
```
Ingested 3 documents

Q: What's our BGP policy for AWS?
A: For AWS BGP peering, you must use MD5 authentication. The AS number is 64512, and peering IP ranges use 169.254.0.0/16. BGP session monitoring with BFD must be enabled.
Sources used: 3

Q: What VLAN range should I use for servers?
A: Server VLANs use the 200-300 range according to our VLAN standards.
Sources used: 3

Q: What's the OSPF backbone area?
A: OSPF Area 0 is the backbone area and is for core routers only.
Sources used: 3

Q: What authentication is needed for AWS BGP?
A: MD5 authentication is required for all BGP peerings with AWS.
Sources used: 3

Q: What's our EIGRP policy?
A: I don't have that information.
Sources used: 3
```

**Verification**:
- ✓ Did it ingest 3 documents?
- ✓ Do answers match your documentation?
- ✓ Does EIGRP question correctly return "I don't have that information"?

### Step 2: Add Persistence Test (5 minutes)

Restart the script and load existing data:

```python
# lab1_persistence_test.py
from lab1_basic_rag import NetworkRAG

if __name__ == "__main__":
    rag = NetworkRAG()

    # Load existing data
    rag.vectorstore = Chroma(
        persist_directory="./lab1_db",
        embedding_function=rag.embeddings
    )

    count = rag.vectorstore._collection.count()
    print(f"Loaded {count} documents from disk\n")

    # Query still works!
    result = rag.query("What VLAN for servers?")
    print(f"Answer: {result['answer']}")
```

**Expected output**:
```
Loaded 3 documents from disk

Answer: Server VLANs use the 200-300 range.
```

**Verification**:
- ✓ Does it load without re-ingesting?
- ✓ Are queries still accurate?

### Step 3: Understand Retrieval (10 minutes)

Add similarity search to see what gets retrieved:

```python
# lab1_debug.py
from lab1_basic_rag import NetworkRAG

if __name__ == "__main__":
    rag = NetworkRAG()
    rag.ingest_from_files("./network_docs")

    question = "What VLAN for servers?"

    # See what gets retrieved
    docs = rag.vectorstore.similarity_search_with_score(question, k=3)

    print(f"Question: {question}\n")
    print("Retrieved documents:")
    for i, (doc, score) in enumerate(docs, 1):
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   Content: {doc.page_content[:100]}...")
```

**Expected output**:
```
Question: What VLAN for servers?

Retrieved documents:

1. Score: 0.423
   Content: VLAN Standards

VLAN 1: Management (DO NOT USE for user traffic)
VLAN 10-100: User VLANs
VLAN 200-300: Server VLANs...

2. Score: 0.651
   Content: BGP Peering Policy - AWS Integration

All BGP peerings with AWS must use MD5 authentication...

3. Score: 0.702
   Content: OSPF Area Design

Area 0 (backbone): Core routers only...
```

**Key insight**: Lower score = more similar. Document 1 (VLAN standards) is most relevant!

### Extensions

1. **Add more documents**: Create 5 more policy files and re-ingest
2. **Increase k**: Try `k=5` or `k=10`, observe how more context affects answers
3. **Test edge cases**: Ask questions with typos, abbreviations, or partial matches
4. **Measure costs**: Track API calls and estimate monthly costs for 1000 queries

### Troubleshooting

**"No documents ingested"**
- Check `./network_docs` exists and has `.txt` files
- Verify files have content: `cat network_docs/*.txt`

**"ChromaDB error: unable to open database"**
- Delete `./lab1_db` directory and re-run
- Check write permissions in current directory

**"Answer is always 'I don't have that information'"**
- Check if documents were actually ingested: `print(len(texts))`
- Verify embeddings work: Test `embeddings.embed_query("test")`

### Verification Questions

<details>
<summary>Q1: Why does the EIGRP question return "I don't have that information"?</summary>

**Answer**: Because EIGRP is not mentioned in any of the ingested documents. The RAG system only answers based on the documents it has. Even though the LLM knows about EIGRP from its training, we've instructed it to answer ONLY from the provided context.

This prevents hallucination - the system won't make up answers.
</details>

<details>
<summary>Q2: What does `k=3` mean in similarity_search?</summary>

**Answer**: `k` is the number of most similar document chunks to retrieve. With `k=3`, the system finds the 3 most relevant chunks and uses them as context for the LLM.

- **k too low** (1-2): Might miss relevant info, incomplete answers
- **k too high** (20+): Includes irrelevant docs, dilutes context, costs more tokens
- **k=3-5**: Sweet spot for most questions
</details>

<details>
<summary>Q3: Why use temperature=0.0 for RAG?</summary>

**Answer**: Temperature controls randomness. For RAG:

- **temperature=0.0**: Deterministic answers, same question → same answer every time
- **temperature > 0**: Varied answers, might paraphrase differently

Since RAG answers are based on factual documentation, we want consistency. The same question should always produce the same factual answer.

Use `temperature=0.0` for RAG, classification, structured outputs.
Use `temperature=0.3-0.7` for conversational chatbots, creative writing.
</details>

---

## Lab 2: Load Real Documentation

**Objective**: Load PDF and Word documents, implement smart chunking, and handle real-world documentation.

**Time**: 45 minutes

**What You'll Learn**:
- Load multiple file formats
- Chunk large documents effectively
- Add metadata for filtering
- Handle encoding issues

### Success Criteria

- [ ] Load at least 2 PDFs and 2 text files
- [ ] Successfully chunk documents (target: 500 tokens/chunk with 50 overlap)
- [ ] Metadata includes filename and source type
- [ ] Answer questions across multiple file types
- [ ] Total cost < $0.10

### Step 1: Create Document Loader (15 minutes)

```python
# lab2_document_loader.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

class MultiFormatLoader:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_directory(self, docs_dir: str):
        """Load all supported formats."""
        all_docs = []
        docs_path = Path(docs_dir)

        # Load PDFs
        for pdf in docs_path.glob("**/*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf))
                docs = loader.load()

                # Add metadata
                for doc in docs:
                    doc.metadata['source_type'] = 'pdf'
                    doc.metadata['filename'] = pdf.name

                all_docs.extend(docs)
                print(f"Loaded PDF: {pdf.name} ({len(docs)} pages)")

            except Exception as e:
                print(f"Failed to load {pdf.name}: {e}")

        # Load text files
        for txt in docs_path.glob("**/*.txt"):
            try:
                with open(txt, encoding='utf-8') as f:
                    content = f.read()

                doc = type('obj', (object,), {
                    'page_content': content,
                    'metadata': {
                        'source_type': 'text',
                        'filename': txt.name
                    }
                })()

                all_docs.append(doc)
                print(f"Loaded TXT: {txt.name}")

            except Exception as e:
                print(f"Failed to load {txt.name}: {e}")

        print(f"\nTotal documents loaded: {len(all_docs)}")

        # Chunk
        chunks = self.splitter.split_documents(all_docs)
        print(f"Created {len(chunks)} chunks\n")

        return chunks


if __name__ == "__main__":
    loader = MultiFormatLoader()
    chunks = loader.load_directory("./network_docs")

    # Show sample
    if chunks:
        sample = chunks[0]
        print(f"Sample chunk:")
        print(f"Filename: {sample.metadata['filename']}")
        print(f"Type: {sample.metadata['source_type']}")
        print(f"Content ({len(sample.page_content)} chars):")
        print(sample.page_content[:200] + "...")
```

Run it:
```bash
python lab2_document_loader.py
```

**Expected output**:
```
Loaded PDF: bgp_policy.pdf (5 pages)
Loaded PDF: ospf_design.pdf (3 pages)
Loaded TXT: vlan_standards.txt
Loaded TXT: acl_policy.txt

Total documents loaded: 10
Created 47 chunks

Sample chunk:
Filename: bgp_policy.pdf
Type: pdf
Content (487 chars):
BGP Peering Policy - AWS Integration

All BGP peerings with AWS must use MD5 authentication. This ensures that routing updates are authenticated and prevents unauthorized route injection.

Configuration requirements:
- AS number: 64512
- Peering IP: 169.254.x.x/30...
```

**Verification**:
- ✓ Loaded both PDFs and text files?
- ✓ Chunk sizes around 500 characters?
- ✓ Metadata includes filename and source_type?

### Step 2: Ingest with Metadata (15 minutes)

```python
# lab2_rag_with_metadata.py
from lab2_document_loader import MultiFormatLoader
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class MetadataRAG:
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0
        )

        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.vectorstore = None

    def ingest_directory(self, docs_dir: str):
        """Load and ingest with metadata."""
        loader = MultiFormatLoader()
        chunks = loader.load_directory(docs_dir)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./lab2_db"
        )

        self.vectorstore.persist()

    def query_with_filter(self, question: str, source_type: str = None):
        """Query with optional metadata filter."""
        search_kwargs = {"k": 5}

        if source_type:
            search_kwargs["filter"] = {"source_type": source_type}

        retriever = self.vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )

        docs = retriever.get_relevant_documents(question)

        # Show what was retrieved
        print(f"\nRetrieved {len(docs)} chunks:")
        for doc in docs:
            print(f"  - {doc.metadata['filename']} ({doc.metadata['source_type']})")

        # Generate answer
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""Answer based on context:
{context}

Question: {question}

Answer:"""

        response = self.llm.invoke(prompt)
        return response.content


if __name__ == "__main__":
    rag = MetadataRAG()

    # Ingest
    rag.ingest_directory("./network_docs")

    # Query all documents
    print("\n=== Query: All Documents ===")
    answer = rag.query_with_filter("What's our BGP policy?")
    print(f"Answer: {answer}")

    # Query only PDFs
    print("\n\n=== Query: PDFs Only ===")
    answer = rag.query_with_filter("What's our BGP policy?", source_type="pdf")
    print(f"Answer: {answer}")

    # Query only text files
    print("\n\n=== Query: Text Files Only ===")
    answer = rag.query_with_filter("VLAN standards", source_type="text")
    print(f"Answer: {answer}")
```

**Expected output**:
```
=== Query: All Documents ===
Retrieved 5 chunks:
  - bgp_policy.pdf (pdf)
  - bgp_policy.pdf (pdf)
  - aws_integration.pdf (pdf)
  - network_policies.txt (text)
  - vlan_standards.txt (text)

Answer: For AWS BGP peering, you must use MD5 authentication...

=== Query: PDFs Only ===
Retrieved 5 chunks:
  - bgp_policy.pdf (pdf)
  - bgp_policy.pdf (pdf)
  - aws_integration.pdf (pdf)
  - ospf_design.pdf (pdf)
  - firewall_rules.pdf (pdf)

Answer: According to the BGP policy PDF, AWS peering requires MD5 auth...

=== Query: Text Files Only ===
Retrieved 5 chunks:
  - vlan_standards.txt (text)
  - network_policies.txt (text)
  - acl_policy.txt (text)

Answer: VLAN standards specify that VLANs 200-300 are for servers...
```

### Step 3: Test Chunking Sizes (15 minutes)

Compare different chunk sizes:

```python
# lab2_chunking_comparison.py
from lab2_document_loader import MultiFormatLoader

# Test different sizes
for chunk_size in [200, 500, 1000, 2000]:
    print(f"\n=== Chunk Size: {chunk_size} ===")

    loader = MultiFormatLoader(chunk_size=chunk_size, chunk_overlap=50)
    chunks = loader.load_directory("./network_docs")

    # Stats
    avg_length = sum(len(c.page_content) for c in chunks) / len(chunks)
    print(f"Chunks created: {len(chunks)}")
    print(f"Avg chunk length: {avg_length:.0f} chars")

    # Show sample
    print(f"Sample: {chunks[0].page_content[:100]}...")
```

**Expected output**:
```
=== Chunk Size: 200 ===
Chunks created: 235
Avg chunk length: 195 chars
Sample: BGP Peering Policy

All BGP peerings with AWS must use MD5 authentication...

=== Chunk Size: 500 ===
Chunks created: 94
Avg chunk length: 487 chars
Sample: BGP Peering Policy - AWS Integration

All BGP peerings with AWS must use MD5...

=== Chunk Size: 1000 ===
Chunks created: 47
Avg chunk length: 978 chars
Sample: BGP Peering Policy - AWS Integration

All BGP peerings with AWS must use MD5 authentication. This ensures that routing...

=== Chunk Size: 2000 ===
Chunks created: 24
Avg chunk length: 1943 chars
Sample: [Full BGP policy document in one chunk]
```

**Observation**:
- **Small chunks (200)**: Many chunks, lose context
- **Medium chunks (500-1000)**: Balanced, recommended
- **Large chunks (2000)**: Fewer chunks, diluted relevance

### Extensions

1. **Add Word docs**: Install `python-docx` and load `.docx` files
2. **Add markdown**: Load `.md` files for wikis
3. **Custom metadata**: Add date, author, version to metadata
4. **Hierarchical chunking**: Preserve document structure (sections, headings)

### Troubleshooting

**"UnicodeDecodeError"**
```python
# Try different encodings
with open(txt, encoding='utf-8') as f:  # Default
with open(txt, encoding='latin-1') as f:  # Fallback
with open(txt, encoding='cp1252') as f:  # Windows
```

**"PDF extraction failed"**
```bash
# Install tesseract for OCR
sudo apt-get install tesseract-ocr  # Linux
brew install tesseract  # Mac

# Try alternative loader
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader(str(pdf))
```

**"Too many chunks"**
- Increase `chunk_size` from 500 to 1000
- Reduce `chunk_overlap` from 50 to 0
- Filter out boilerplate sections

---

## Lab 3: Production RAG with Citations

**Objective**: Build a production-ready RAG system with proper citations, confidence scoring, and error handling.

**Time**: 60 minutes

**What You'll Learn**:
- Add citations to LLM answers
- Implement confidence scoring
- Handle edge cases (no results, errors)
- Production logging and monitoring

### Success Criteria

- [ ] Answers include [Source N] citations
- [ ] Confidence score based on number of relevant docs
- [ ] Handles queries with no relevant docs gracefully
- [ ] Logs all queries and errors
- [ ] Complete 10-query session under $0.20

### Step 1: Build Production RAG (25 minutes)

Create full production system from earlier V4 example:

```python
# lab3_production_rag.py
# (Use the RAGV4 code from progressive builds section)
```

### Step 2: Test Citation System (15 minutes)

```python
# lab3_test_citations.py
from lab3_production_rag import RAGV4
import os

if __name__ == "__main__":
    rag = RAGV4(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Load existing
    rag.load_existing()

    # Test citation
    result = rag.query("What's our BGP policy for AWS?", k=3)

    print(f"Answer:\n{result['answer']}\n")
    print(f"Confidence: {result['confidence']}\n")
    print(f"Sources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. {source['filename']} (page {source['page']})")
        print(f"   Preview: {source['content_preview']}\n")
```

**Expected output**:
```
Answer:
For AWS BGP peering, you must use MD5 authentication [Source 1]. The AS number for AWS connections is 64512 [Source 1], and peering IP ranges use 169.254.0.0/16 [Source 2]. BGP session monitoring with BFD must be enabled [Source 1].

Confidence: high

Sources:
1. bgp_policy.pdf (page 3)
   Preview: BGP Peering Policy - AWS Integration

All BGP peerings with AWS must use MD5 authentication...

2. bgp_policy.pdf (page 5)
   Preview: Peering IP Configuration

AWS Direct Connect uses 169.254.0.0/16 for BGP peering...

3. aws_integration.pdf (page 2)
   Preview: AWS Network Integration Guide

When establishing BGP sessions with AWS...
```

**Verification**:
- ✓ Answer includes [Source 1], [Source 2], etc?
- ✓ Confidence is "high" (3+ sources) or "medium" (1-2 sources)?
- ✓ Source previews show relevant content?

### Step 3: Handle Edge Cases (20 minutes)

```python
# lab3_edge_cases.py
from lab3_production_rag import RAGV4
import os

if __name__ == "__main__":
    rag = RAGV4(api_key=os.getenv("ANTHROPIC_API_KEY"))
    rag.load_existing()

    # Edge case 1: Completely irrelevant question
    print("=== Test 1: Irrelevant Question ===")
    result = rag.query("What's the weather in Paris?")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}\n")

    # Edge case 2: Very specific question
    print("=== Test 2: Very Specific ===")
    result = rag.query("What's the exact MD5 password for AWS BGP session 169.254.1.1?")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}\n")

    # Edge case 3: Filter with no matches
    print("=== Test 3: Filter No Matches ===")
    result = rag.query(
        "BGP policy",
        filter_metadata={"source_type": "docx"}  # No docx files
    )
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])}\n")
```

**Expected output**:
```
=== Test 1: Irrelevant Question ===
Answer: I don't have that information.
Confidence: low

=== Test 2: Very Specific ===
Answer: The documentation specifies that MD5 passwords must be minimum 16 characters [Source 1], but does not provide the exact password for session 169.254.1.1. Contact network-ops@company.com for specific session credentials [Source 2].
Confidence: medium

=== Test 3: Filter No Matches ===
Answer: No relevant documentation found.
Sources: 0
```

### Extensions

1. **Add web UI**: Build Flask/Streamlit interface
2. **Multi-turn conversations**: Remember previous questions
3. **Query optimization**: Rewrite user questions for better retrieval
4. **Hybrid search**: Combine semantic + keyword search
5. **Answer validation**: Check if answer actually cites sources

### Troubleshooting

**"Citations not appearing in answer"**
- Check prompt includes: "ALWAYS cite sources using [Source N]"
- Verify context includes source markers: `[Source 1: filename]`
- Try lowering temperature to 0.0 for more consistent output

**"Confidence always 'low'"**
- Check k value: Increase to retrieve more docs
- Verify documents were actually ingested
- Lower similarity threshold to retrieve more matches

**"Slow queries (>5 seconds)"**
- Use local embeddings (all-MiniLM-L6-v2) not API-based
- Reduce k from 10 to 3-5
- Consider FAISS instead of Chroma for large collections

### Verification Questions

<details>
<summary>Q1: Why does the system say "I don't have that information" instead of using LLM's general knowledge?</summary>

**Answer**: Because we explicitly instruct the LLM to answer ONLY based on the provided context. This prevents hallucination.

Without this constraint:
```python
# Bad: LLM uses general knowledge
Answer: AWS BGP typically uses ASN 65000... (WRONG - made up!)
```

With constraint:
```python
# Good: Only uses our docs
Answer: According to our BGP policy, AWS uses ASN 64512 [Source 1]
```

This ensures answers are grounded in YOUR documentation, not general internet knowledge.
</details>

<details>
<summary>Q2: How does confidence scoring work?</summary>

**Answer**: Based on number of relevant documents retrieved:

```python
if len(relevant_docs) >= 3:
    confidence = "high"  # Strong evidence from multiple sources
elif len(relevant_docs) >= 1:
    confidence = "medium"  # Some evidence, but limited
else:
    confidence = "low"  # No relevant docs found
```

You can enhance this with:
- Similarity scores: High scores (< 0.3) = high confidence
- Source diversity: Citations from multiple files = higher confidence
- LLM self-assessment: Ask LLM to rate its own confidence
</details>

<details>
<summary>Q3: When should you use metadata filtering?</summary>

**Answer**:

**Use filtering when**:
- User specifies document type: "According to the PDF..."
- Need recent docs only: `filter={"date": {"$gte": "2024-01-01"}}`
- Querying specific systems: `filter={"device_type": "firewall"}`
- Multi-tenant system: `filter={"customer_id": "abc123"}`

**Skip filtering when**:
- Querying across all documentation
- Not sure which document type has the answer
- Want comprehensive results

Example:
```python
# Filter: Only search firewall configs
result = rag.query(
    "ACL rules",
    filter_metadata={"device_type": "firewall", "date": {"$gte": "2024-01-01"}}
)
```
</details>

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

## Check Your Understanding: RAG Concepts

Test your knowledge before moving to the next chapter.

<details>
<summary>Q1: What are the three steps in RAG?</summary>

**Answer**:

1. **Retrieval**: Find relevant documents using embeddings/vector search
2. **Augmented**: Add retrieved docs to the LLM prompt as context
3. **Generation**: LLM generates answer using the provided context

Example:
```python
# 1. Retrieval
docs = vectorstore.similarity_search("BGP policy", k=3)

# 2. Augmented
context = "\n\n".join([d.page_content for d in docs])
prompt = f"Context: {context}\n\nQuestion: BGP policy?"

# 3. Generation
answer = llm.invoke(prompt)
```

Without RAG, the LLM would answer from its training data (may be outdated/wrong).
With RAG, the LLM answers from YOUR current documentation.
</details>

<details>
<summary>Q2: Why does chunk size matter?</summary>

**Answer**:

Chunk size affects both **context** and **relevance**:

**Too small (< 200 tokens)**:
- Loses context: "Use MD5" without explaining what for
- Needs more chunks to get complete answer
- More API calls, higher cost

**Too large (> 2000 tokens)**:
- Dilutes relevance: Entire document in one chunk
- Hard to find specific information
- Exceeds context limits faster

**Sweet spot (500-1000 tokens)**:
- Balances context and specificity
- One chunk = one concept/policy
- Works for most technical documentation

**Practical test**:
```python
# Test on your docs
for size in [200, 500, 1000, 2000]:
    chunks = text_splitter.split_documents(docs)
    print(f"Size {size}: {len(chunks)} chunks")
    # See which gives best retrieval
```

Rule: If answers are incomplete, increase chunk size. If answers include irrelevant info, decrease it.
</details>

<details>
<summary>Q3: What's the difference between similarity_search() and as_retriever()?</summary>

**Answer**:

**similarity_search()**: Direct vector database method
```python
# Returns documents sorted by similarity
docs = vectorstore.similarity_search("BGP", k=5)
# Simple, direct access to vector DB
```

**as_retriever()**: LangChain abstraction for chains
```python
# Returns a retriever object for use in chains
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.get_relevant_documents("BGP")

# Can be used in chains
chain = retriever | format_docs | llm
```

**When to use each**:
- **similarity_search()**: Debugging, exploring, simple scripts
- **as_retriever()**: Building chains, production systems, metadata filtering

**Key benefit of as_retriever()**:
```python
# Metadata filtering
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source_type": "pdf"}  # Only PDFs
    }
)
```
</details>

<details>
<summary>Q4: How do embeddings capture meaning?</summary>

**Answer**:

Embeddings convert text to vectors (arrays of numbers) where similar meanings = similar vectors.

**Example**:
```python
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Embed similar texts
vec1 = embeddings.embed_query("BGP routing protocol")
vec2 = embeddings.embed_query("Border Gateway Protocol")
vec3 = embeddings.embed_query("VLAN configuration")

# Calculate similarity (cosine similarity)
similarity(vec1, vec2)  # 0.92 (very similar!)
similarity(vec1, vec3)  # 0.34 (different topics)
```

**How it works**:
1. Model is trained on millions of text pairs
2. Learns to put similar meanings close together in vector space
3. "BGP" and "Border Gateway Protocol" end up near each other
4. "BGP" and "VLAN" end up far apart

**Vector space visualization**:
```
          "Border Gateway Protocol"
                   |
                  0.92 similarity
                   |
          "BGP routing protocol" ----0.34---- "VLAN configuration"
```

This is why RAG can find "BGP policy" even when you search for "Border Gateway Protocol" - the embeddings understand they mean the same thing.
</details>

<details>
<summary>Q5: Why use local embeddings (all-MiniLM-L6-v2) instead of OpenAI/Anthropic embeddings?</summary>

**Answer**:

**Local embeddings (all-MiniLM-L6-v2)**:
- Free (no API costs)
- Fast (runs on your machine)
- Private (data never leaves your system)
- Offline (works without internet)
- Good quality for most use cases

**Costs**:
```python
# OpenAI embeddings: $0.0001 per 1K tokens
# For 100,000 docs × 500 tokens = 50M tokens
# Cost: 50,000 × $0.0001 = $5,000

# Local embeddings: $0
```

**When to use API embeddings**:
- Need absolute best quality
- Budget allows
- Already using that provider's LLM
- Don't want to manage local models

**Performance comparison**:
```
Model                Speed       Quality    Cost
all-MiniLM-L6-v2     100/sec     Good       Free
all-mpnet-base-v2    50/sec      Better     Free
OpenAI text-emb      API limited Best       $$$
```

**Recommendation**: Start with all-MiniLM-L6-v2. Only upgrade if quality is insufficient.
</details>

---

## Lab Time Budget and ROI Analysis

### Time Investment

| Lab | Setup | Coding | Testing | Total Time | Skill Level |
|-----|-------|--------|---------|------------|-------------|
| **Lab 0: Environment Setup** | 15 min | - | 5 min | **20 min** | Beginner |
| **Lab 1: Basic RAG** | - | 15 min | 15 min | **30 min** | Beginner |
| **Lab 2: Document Loading** | - | 25 min | 20 min | **45 min** | Intermediate |
| **Lab 3: Production RAG** | - | 35 min | 25 min | **60 min** | Advanced |
| **Total** | **15 min** | **75 min** | **65 min** | **2 hours 35 min** | - |

**Recommended schedule**:
- Day 1: Lab 0 + Lab 1 (50 min)
- Day 2: Lab 2 (45 min)
- Day 3: Lab 3 (60 min)

### API Cost Breakdown

**Per-lab costs** (Claude Sonnet 4: $3/$15 per 1M input/output tokens):

| Lab | Queries | Est. Tokens (input/output) | Cost | Notes |
|-----|---------|----------------------------|------|-------|
| **Lab 0: Setup** | 1 | 100/20 | $0.001 | API test only |
| **Lab 1: Basic RAG** | 5 | 3K/1.5K | $0.03 | 5 questions × 3 sources each |
| **Lab 2: Document Loading** | 8 | 5K/2K | $0.05 | Metadata filtering tests |
| **Lab 3: Production RAG** | 10 | 8K/3K | $0.07 | Citations + edge cases |
| **Total** | **24** | **16K/6.5K** | **$0.15** | Full chapter completion |

**Cost breakdown**:
- Embeddings: $0 (local model)
- Vector DB: $0 (ChromaDB local)
- Claude API: ~$0.15 (only for generation step)

**Cost optimization**:
1. **Cache embeddings**: Generate once, reuse forever
   - First ingest: $0
   - Subsequent queries: $0 for retrieval, only pay for LLM generation

2. **Use smaller context**: Reduce k from 10 to 3-5
   - Saves ~50% on input tokens
   - Still maintains answer quality

3. **Batch questions**: Ask multiple questions in one API call
   ```python
   # Instead of 5 calls ($0.05)
   for q in questions:
       answer = llm.invoke(prompt)

   # Do 1 batched call ($0.02)
   batch_prompt = "\n".join([f"Q{i}: {q}" for i, q in enumerate(questions)])
   answers = llm.invoke(batch_prompt)
   ```

**Realistic total cost with optimization**: $0.05-$0.10 per learner

### Infrastructure Costs

**One-time setup**:
- Python environment: Free
- VS Code: Free
- ChromaDB: Free (open source)
- Embedding model (all-MiniLM-L6-v2): Free (80MB download)
- **Total**: $0

**Ongoing costs** (after free API credits):
- Anthropic API: $0.05-$0.15 per chapter
- Storage: ~100MB for 1,000 docs (negligible)
- **Monthly**: $5-10 for active development

### Production Deployment Costs

Once you deploy your RAG system in production:

| System | Daily Usage | Daily Cost | Monthly Cost | Savings vs Manual |
|--------|-------------|------------|--------------|-------------------|
| **Documentation Q&A** | 100 queries | $0.35 | $10 | 25 hours/month ($1,000) |
| **Policy Lookup** | 50 queries | $0.18 | $5 | 15 hours/month ($600) |
| **Config Reference** | 200 queries | $0.70 | $21 | 40 hours/month ($1,600) |
| **Total** | **350/day** | **$1.23** | **$37** | **$3,200/month saved** |

**Payback period**: < 1 day (save 5+ hours on day 1)

### Business Value

**What you can build after these labs**:

1. **Network Documentation Search**
   - Instant answers from 1,000+ pages of docs
   - Saves: 20 hours/week (was: 20min search × 60 queries/week)
   - Value: $8,000/month

2. **Automated Policy Compliance Checker**
   - Query: "Does this config comply with our BGP policy?"
   - Saves: 10 hours/week on manual reviews
   - Value: $4,000/month

3. **Onboarding Assistant**
   - New engineers get instant answers to "How do we do X?"
   - Saves: 40 hours per new hire onboarding
   - Value: $4,000 per hire

4. **Change Review Assistant**
   - Query: "What's our process for adding a new VLAN?"
   - Saves: 5 hours/week looking up procedures
   - Value: $2,000/month

**Total potential value**: $18,000/month in time savings

**Investment**:
- Learning time: 2.5 hours
- API costs: $0.15
- Ongoing production: $37/month

**ROI Calculation**:
- Time to value: 1 day (deploy first system)
- Payback period: < 1 hour (first day's time savings >> $0.15)
- Annual ROI: 580,000% (save $216K/year, spend $444/year)

### Cost Justification Template

Use this when requesting budget:

```
Subject: RAG Documentation System - Budget Request

Request: $50/month for Anthropic API credits
Purpose: Build AI-powered documentation search system

Expected Outcomes:
1. Instant answers from 1,000+ pages of network documentation
   - Current: 20 min average search time
   - With RAG: 5 second answer time
   - Saves: 20 hours/week across team

2. Automated policy compliance checking
   - Current: 2 hours/week manual reviews
   - With RAG: Instant automated checks
   - Saves: 100+ hours/year

3. Faster onboarding for new engineers
   - Current: 2 weeks to learn all policies/procedures
   - With RAG: Instant self-service Q&A
   - Saves: 40 hours per new hire

Cost-Benefit:
- Monthly cost: $50 (API + buffer)
- Monthly savings: 80 hours × $40/hr = $3,200
- ROI: 6,400%
- Payback: 1 day

This system pays for itself in the first day of use and generates
$3,200 in monthly value.

Recommended: Approve $50/month budget for 3-month trial.
```

### Scaling Costs

As you scale to more users/queries:

| Monthly Queries | API Cost | Storage | Total Cost | Cost per Query |
|----------------|----------|---------|------------|----------------|
| 1,000 | $3.50 | $0 | $3.50 | $0.0035 |
| 10,000 | $35 | $1 | $36 | $0.0036 |
| 100,000 | $350 | $5 | $355 | $0.0036 |
| 1,000,000 | $3,500 | $50 | $3,550 | $0.0036 |

**Key insight**: Costs scale linearly with usage, and retrieval (embeddings + vector search) is FREE. You only pay for LLM generation.

**Cost reduction at scale**:
- Use Claude Haiku for simple queries: $0.25/$1.25 (90% cheaper)
- Cache frequently asked questions: Free after first query
- Hybrid approach: Haiku for 80% of queries, Sonnet for complex ones
- Potential savings: 70-80% reduction in costs

---

## Key Takeaways

1. **RAG grounds AI in your documentation** - No more hallucinated answers
2. **Embeddings capture semantic meaning** - Find relevant docs by similarity
3. **Chunking strategy matters** - Balance context vs relevance
4. **Metadata enables filtering** - Filter by doc type, date, author
5. **Citations build trust** - Always show sources

Next chapter: Advanced RAG patterns and semantic search.
