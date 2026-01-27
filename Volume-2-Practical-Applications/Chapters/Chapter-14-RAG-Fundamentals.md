# Chapter 14: RAG Fundamentals - Making Documentation Searchable with AI

## Introduction

You just finished Chapter 13. Your network now has **auto-generated, always-current documentation** for all 500 devices. Great!

But there's a problem: **Nobody uses it.**

Why? Because finding the answer you need still takes too long:
- Engineer needs to know: "Which routers run BGP in production?"
- They search the documentation directory
- 500 markdown files appear
- They manually open and read each one
- 30 minutes later: answer found (or maybe not)

Or worse:
- Engineer asks: "What's the OSPF cost on the backup core router?"
- No clear answer in any document
- They give up and ask someone else
- Knowledge stays in someone's head instead of the docs

**The solution**: Build a system that understands natural language questions and retrieves relevant documentation automatically.

This is **Retrieval-Augmented Generation (RAG)** — the most practical application of AI for enterprise knowledge management.

---

## What is RAG?

### The Basic Idea

**RAG = Retrieval + Generation**

```
User Question
    ↓
Search Documentation (Retrieval)
    ↓
Find Relevant Sections
    ↓
Send to Claude with Question (Generation)
    ↓
Claude Answers Based on Docs
    ↓
User Gets Grounded Answer
```

**Without RAG**:
```
User: "What's the BGP AS for AWS?"
Claude: "AWS typically uses AS 16509. But I'm not sure about your specific config."
↑ Generic answer, not grounded in your actual network
```

**With RAG**:
```
User: "What's the BGP AS for AWS?"
RAG System: [Searches docs, finds router-core-01.md]
Claude: [Reads that doc] "According to your router-core-01 documentation,
         the BGP AS for AWS is 65001, configured on interface Gi0/0."
↑ Specific answer grounded in your actual documentation
```

### Why RAG Matters

**Before RAG**:
- Documentation exists but is hard to search
- Information buried in 500 separate files
- Text search finds documents but not answers
- Engineers still ask questions verbally

**After RAG**:
- Natural language questions get instant answers
- Answers grounded in actual documentation
- Reduces tribal knowledge dependency
- Faster onboarding for new engineers
- Audit trail of what people know

---

## The Problem with Text Search

### Why Google-Style Search Doesn't Work for Docs

**Text search is literal**:
```
Search: "ospf cost"
Results:
  - router-core-01.md: "ip ospf cost 100"
  - router-core-02.md: "ip ospf cost 100"
  - switch-dist-01.md: [no match]
  - firewall-01.md: [no match]
  - [495 more files with no matches]

Found 2 results. User has to read them all.
```

**RAG is semantic**:
```
Question: "Which routers have OSPF configured?"
RAG searches for: [routers + OSPF + similar configs]
Results:
  - router-core-01.md: Has OSPF config
  - router-core-02.md: Has OSPF config
  - switch-dist-01.md: No OSPF (correctly excluded)
  - firewall-01.md: No OSPF (correctly excluded)

Found 2 results. No false positives.
```

### The Semantic Search Advantage

**Semantic search understands meaning**, not just keywords:

| Question | Text Search | Semantic Search |
|----------|------------|-----------------|
| "How do we route traffic to AWS?" | Searches for "AWS" in text | Understands "routing to AWS" = BGP config, static routes, VPN, etc. |
| "Which devices have redundancy?" | Searches for "redundancy" | Understands HSRP, VRRP, dual links, failover as synonyms |
| "Show me firewall rules" | Finds "ACL" entries | Understands firewall, ACL, security policy are related |

---

## Section 1: Vector Embeddings

### How Embeddings Work

**An embedding is a numerical representation of text**:

```
"ip ospf cost 100"
    ↓
[0.234, -0.891, 0.123, 0.456, ...]  ← 1,536 numbers
    ↓
This vector represents the meaning of that text
```

**The magic**: Texts with similar meanings have similar vectors.

```
"ip ospf cost 100"    → [0.234, -0.891, 0.123, ...]
"ospf interface cost" → [0.235, -0.890, 0.124, ...]  ← Very similar!

"BGP neighbor 1.2.3.4" → [-0.456, 0.789, 0.234, ...]  ← Different!
```

### Creating Embeddings with Claude

```python
from anthropic import Anthropic

client = Anthropic()

# Get embedding for a piece of documentation
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You are an embedding generator...",
    messages=[{
        "role": "user",
        "content": "Get embedding for: 'ip ospf cost 100'"
    }]
)
```

### Why This Matters

**Semantic similarity means smarter search**:

```python
# All of these would be found by semantic search
questions_about_ospf = [
    "What OSPF costs are configured?",
    "How are OSPF interfaces configured?",
    "Show me OSPF settings",
    "What's the OSPF cost on Gi0/1?"
]

# All would find docs with "ip ospf cost X"
# because their embeddings are semantically similar
```

---

## Section 2: Vector Database Setup

### What is a Vector Database?

A vector database stores embeddings and finds similar ones quickly:

```
Document: "ip ospf cost 100"
Embedding: [0.234, -0.891, 0.123, ...]
          ↓
Vector Database
(optimized for semantic search)
          ↓
User Question: "What are OSPF costs?"
Embedding: [0.235, -0.890, 0.124, ...]
          ↓
Search finds documents with similar embeddings
```

### Vector Database Options

| Database | Cost | Speed | Complexity | Best For |
|----------|------|-------|-----------|----------|
| **Pinecone** | $$ (Serverless) | Very Fast | Low | Cloud-native, no infra |
| **Weaviate** | $ (Open Source) | Fast | Medium | Self-hosted, control |
| **Milvus** | $ (Open Source) | Fast | Medium | High-volume data |
| **Chroma** | FREE (Embedded) | Medium | Very Low | Prototyping, dev |
| **FAISS** | FREE (Library) | Very Fast | High | Research, advanced |

### Chroma: The Easiest Start

**Chroma is embedded - no server needed**:

```python
import chromadb

# Create client (file-based, no setup needed)
client = chromadb.Client()

# Create collection
collection = client.create_collection(name="network_docs")

# Add documents with embeddings (automatic)
collection.add(
    ids=["doc1", "doc2"],
    documents=["OSPF config...", "BGP config..."],
    metadatas=[
        {"device": "router-core-01"},
        {"device": "router-core-02"}
    ]
)

# Search (automatic embedding + semantic search)
results = collection.query(
    query_texts=["What OSPF costs are configured?"],
    n_results=3
)
```

**That's it! No vector database to manage.**

---

## Section 3: Building Your First RAG System

### The Complete RAG Pipeline

```python
# rag_system.py
from anthropic import Anthropic
import chromadb
from pathlib import Path

class DocumentationRAG:
    """Retrieve and answer questions about network documentation."""

    def __init__(self, docs_directory: str):
        self.client = Anthropic()
        self.chroma = chromadb.Client()
        self.collection = self.chroma.create_collection(
            name="network_documentation"
        )
        self.docs_directory = Path(docs_directory)

    def index_documentation(self):
        """Load all .md files and index them."""
        
        print("Indexing documentation...")
        doc_count = 0

        for doc_file in self.docs_directory.glob("*.md"):
            with open(doc_file, 'r') as f:
                content = f.read()

            # Split long documents into chunks
            chunks = self._chunk_document(content, chunk_size=1000)

            for i, chunk in enumerate(chunks):
                doc_id = f"{doc_file.stem}_chunk_{i}"
                
                self.collection.add(
                    ids=[doc_id],
                    documents=[chunk],
                    metadatas={
                        "source": doc_file.name,
                        "chunk": i
                    }
                )
                doc_count += 1

        print(f"✓ Indexed {doc_count} document chunks")

    def _chunk_document(self, text: str, chunk_size: int = 1000) -> list:
        """Split long documents into chunks for better retrieval."""
        
        chunks = []
        current_chunk = ""

        for paragraph in text.split("\n\n"):
            if len(current_chunk) + len(paragraph) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def retrieve_relevant_docs(self, question: str, n_results: int = 3) -> list:
        """Find relevant documentation for a question."""
        
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )

        # Format results for Claude
        retrieved_docs = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            retrieved_docs.append({
                'content': doc,
                'source': metadata['source']
            })

        return retrieved_docs

    def answer_question(self, question: str) -> str:
        """Answer a question using retrieved documentation."""
        
        # Retrieve relevant documents
        docs = self.retrieve_relevant_docs(question)

        # Build context from retrieved docs
        context = "\n\n".join([
            f"From {doc['source']}:\n{doc['content']}"
            for doc in docs
        ])

        # Ask Claude to answer using the context
        prompt = f"""Based on the following network documentation, answer this question:

Question: {question}

Documentation:
{context}

Answer based ONLY on the documentation above. If the answer isn't in the docs, say so."""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        return response.content[0].text

    def interactive_session(self):
        """Start interactive question-answering session."""
        
        print("Network Documentation Q&A System")
        print("=" * 50)
        print("Ask questions about your network documentation.")
        print("Type 'exit' to quit.\n")

        while True:
            question = input("You: ").strip()

            if question.lower() == 'exit':
                break

            if not question:
                continue

            answer = self.answer_question(question)
            print(f"\nAssistant: {answer}\n")


# Usage
if __name__ == "__main__":
    import os

    rag = DocumentationRAG(
        docs_directory="./network_docs"
    )

    # Index all documentation files
    rag.index_documentation()

    # Start interactive session
    rag.interactive_session()
```

### Example Questions Your RAG Can Answer

```
Q: "Which routers run OSPF?"
A: "Based on your documentation, router-core-01, router-core-02, 
   and router-branch-01 all run OSPF process 1."

Q: "What's the BGP configuration for AWS?"
A: "Your router-core-01 has BGP AS 65001 configured with AWS
   neighbor 203.0.113.2 (AS 16509) on interface Gi0/0."

Q: "Show me all devices with HSRP"
A: "HSRP is configured on: router-core-01 (priority 110), 
   router-core-02 (priority 100), and router-branch-01 (priority 90)."

Q: "What security policies are in place?"
A: "Management access is restricted to 10.0.0.0/16 via SSH only.
   VTY lines have access-class MANAGEMENT_ACCESS applied."
```

---

## Section 4: Advanced RAG Patterns

### Pattern 1: Multi-Step Retrieval

For complex questions, retrieve multiple times:

```python
def answer_complex_question(self, question: str) -> str:
    """Answer questions that need multiple retrieval passes."""
    
    # First pass: Find relevant devices
    print("Step 1: Finding relevant devices...")
    device_docs = self.retrieve_relevant_docs(
        f"Which devices are relevant to: {question}",
        n_results=5
    )
    
    # Second pass: Find specific configuration
    print("Step 2: Finding specific configuration...")
    config_docs = self.retrieve_relevant_docs(
        f"Configuration details for: {question}",
        n_results=3
    )
    
    # Combine both sets
    all_docs = device_docs + config_docs
    
    # Answer with all context
    context = "\n\n".join([d['content'] for d in all_docs])
    
    return self._answer_with_context(question, context)
```

### Pattern 2: Confidence Scoring

Know when the system is confident vs. uncertain:

```python
def answer_with_confidence(self, question: str):
    """Return answer with confidence score."""
    
    docs = self.retrieve_relevant_docs(question, n_results=5)
    
    # Check if results are relevant
    similarity_scores = [
        self._calculate_similarity(question, doc['content'])
        for doc in docs
    ]
    
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    
    if avg_similarity < 0.5:
        confidence = "LOW - Documentation may not cover this"
    elif avg_similarity < 0.7:
        confidence = "MEDIUM - Partially relevant documentation found"
    else:
        confidence = "HIGH - Strong documentation match"
    
    answer = self.answer_question(question)
    
    return {
        "answer": answer,
        "confidence": confidence,
        "similarity_score": avg_similarity
    }
```

### Pattern 3: Citation Tracking

Show users which docs were used:

```python
def answer_with_citations(self, question: str):
    """Return answer with sources cited."""
    
    docs = self.retrieve_relevant_docs(question, n_results=3)
    
    context_with_refs = "\n\n".join([
        f"[{i+1}] From {doc['source']}:\n{doc['content']}"
        for i, doc in enumerate(docs)
    ])
    
    # Answer with citations
    response = self.client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Answer this question citing sources:

Question: {question}

Documentation:
{context_with_refs}

Format your answer with [1], [2], [3] citations where appropriate."""
        }]
    )
    
    return {
        "answer": response.content[0].text,
        "sources": [doc['source'] for doc in docs]
    }
```

---

## Section 5: Production RAG System

### Complete Production Implementation

```python
# production_rag.py
from anthropic import Anthropic
import chromadb
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionDocumentationRAG:
    """Production-grade RAG system for network documentation."""

    def __init__(
        self,
        docs_directory: str,
        vector_db_path: str = "./vector_db",
        log_file: str = "./rag_queries.log"
    ):
        self.client = Anthropic()
        self.docs_directory = Path(docs_directory)
        self.vector_db_path = Path(vector_db_path)
        self.log_file = Path(log_file)
        
        # Initialize vector database
        self.chroma = chromadb.PersistentClient(path=str(self.vector_db_path))
        self.collection = self.chroma.get_or_create_collection(
            name="network_documentation"
        )
        
        # Track stats
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "api_calls": 0
        }

    def index_documentation(self, force_reindex: bool = False):
        """Index all documentation, with caching."""
        
        if not force_reindex and self.collection.count() > 0:
            logger.info(f"Using cached index with {self.collection.count()} chunks")
            return
        
        logger.info("Indexing documentation...")
        doc_count = 0

        for doc_file in self.docs_directory.glob("*.md"):
            with open(doc_file, 'r') as f:
                content = f.read()

            chunks = self._chunk_document(content)

            for i, chunk in enumerate(chunks):
                doc_id = f"{doc_file.stem}_chunk_{i}"
                
                self.collection.add(
                    ids=[doc_id],
                    documents=[chunk],
                    metadatas=[{
                        "source": doc_file.name,
                        "chunk": i,
                        "timestamp": datetime.now().isoformat()
                    }]
                )
                doc_count += 1

        logger.info(f"✓ Indexed {doc_count} document chunks")

    def _chunk_document(self, text: str, chunk_size: int = 1500) -> List[str]:
        """Intelligently chunk documents."""
        
        chunks = []
        current_chunk = ""

        # Split by sections first (##)
        sections = text.split("\n## ")
        
        for section in sections:
            # Further split by paragraphs if section is too large
            paragraphs = section.split("\n\n")
            
            for para in paragraphs:
                if len(current_chunk) + len(para) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def answer_question(
        self,
        question: str,
        max_results: int = 5,
        include_citations: bool = True
    ) -> Dict:
        """Answer a question with full production features."""
        
        try:
            # Log query
            self._log_query(question)
            self.stats["total_queries"] += 1

            # Retrieve relevant docs
            results = self.collection.query(
                query_texts=[question],
                n_results=max_results
            )

            if not results['documents'][0]:
                return {
                    "answer": "No relevant documentation found.",
                    "confidence": "LOW",
                    "sources": []
                }

            # Format retrieved documents
            retrieved_docs = [
                {
                    'content': doc,
                    'source': metadata['source'],
                    'distance': distance
                }
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]

            # Build context
            context = self._build_context(retrieved_docs, include_citations)

            # Generate answer
            prompt = f"""Based on the following network documentation, answer this question:

Question: {question}

Documentation:
{context}

Provide a clear, concise answer based ONLY on the documentation.
If information is incomplete, state what's missing."""

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            self.stats["api_calls"] += 1
            self.stats["successful_queries"] += 1

            return {
                "answer": response.content[0].text,
                "confidence": self._calculate_confidence(retrieved_docs),
                "sources": [doc['source'] for doc in retrieved_docs],
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            self.stats["failed_queries"] += 1
            return {
                "answer": f"Error processing query: {str(e)}",
                "confidence": "ERROR",
                "sources": []
            }

    def _build_context(self, docs: List[Dict], include_citations: bool) -> str:
        """Build context string from retrieved documents."""
        
        if include_citations:
            return "\n\n".join([
                f"[{i+1}] From {doc['source']}:\n{doc['content']}"
                for i, doc in enumerate(docs)
            ])
        else:
            return "\n\n".join([doc['content'] for doc in docs])

    def _calculate_confidence(self, docs: List[Dict]) -> str:
        """Calculate confidence based on retrieval quality."""
        
        if not docs:
            return "NONE"
        
        avg_distance = sum(d['distance'] for d in docs) / len(docs)
        
        if avg_distance < 0.3:
            return "HIGH"
        elif avg_distance < 0.6:
            return "MEDIUM"
        else:
            return "LOW"

    def _log_query(self, question: str):
        """Log all queries for audit trail."""
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "question": question
            }) + "\n")

    def get_stats(self) -> Dict:
        """Get system statistics."""
        
        return {
            **self.stats,
            "indexed_chunks": self.collection.count(),
            "success_rate": (
                self.stats["successful_queries"] / max(1, self.stats["total_queries"])
            )
        }


# FastAPI Integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Network Documentation RAG API")

# Initialize RAG system
rag = ProductionDocumentationRAG(docs_directory="./network_docs")
rag.index_documentation()

class QuestionRequest(BaseModel):
    question: str
    max_results: int = 5

@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about network documentation."""
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    result = rag.answer_question(
        question=request.question,
        max_results=request.max_results
    )
    
    return result

@app.get("/api/stats")
async def get_stats():
    """Get RAG system statistics."""
    
    return rag.get_stats()

@app.post("/api/reindex")
async def reindex():
    """Manually reindex documentation."""
    
    rag.index_documentation(force_reindex=True)
    return {"status": "Reindexing started"}
```

---

## Section 6: Integration with Chapter 13

### Auto-Update RAG Index on Doc Generation

```python
# Updated documentation_pipeline.py
from production_rag import ProductionDocumentationRAG

class DocumentationPipeline:
    def __init__(self, ...):
        # ... existing code ...
        self.rag = ProductionDocumentationRAG(
            docs_directory=self.output_dir,
            vector_db_path="./vector_db"
        )

    def generate_all_documentation(self):
        # ... generate docs as before ...
        
        # NEW: Update RAG index
        print("Updating RAG index...")
        self.rag.index_documentation(force_reindex=True)
        print("✓ RAG index updated")
```

### Workflow

```
1. Device config changes
   ↓
2. Chapter 13: Auto-generate new documentation
   ↓
3. Chapter 14: Auto-update RAG index
   ↓
4. RAG system immediately searchable with latest docs
```

---

## Best Practices

### 1. Chunking Strategy

- **Too small** (100 words): Loses context
- **Too large** (5000 words): Retrieves irrelevant data
- **Optimal** (1000-1500 words): Balanced context

### 2. Query Expansion

Ask related questions to find more relevant docs:

```python
def expand_query(self, question: str) -> List[str]:
    """Generate related queries to improve retrieval."""
    
    response = self.client.messages.create(
        model="claude-3-5-haiku-20241022",  # Faster, cheaper
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Generate 2-3 related questions for: {question}
            
Format: ["question1", "question2", "question3"]"""
        }]
    )
    
    import json
    return json.loads(response.content[0].text)
```

### 3. Re-ranking Results

Improve retrieval by re-ranking with Claude:

```python
def rerank_results(self, question: str, candidates: List[Dict]) -> List[Dict]:
    """Re-rank retrieved documents for relevance."""
    
    candidate_texts = "\n".join([
        f"{i}. {c['content'][:500]}..." 
        for i, c in enumerate(candidates)
    ])
    
    response = self.client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Rank these by relevance to: {question}
            
{candidate_texts}

Return: [most relevant index, 2nd, 3rd, ...]"""
        }]
    )
    
    ranking = json.loads(response.content[0].text)
    return [candidates[i] for i in ranking]
```

### 4. Feedback Loop

Track what users found helpful:

```python
@app.post("/api/feedback")
async def submit_feedback(
    question: str,
    answer: str,
    helpful: bool,
    user_id: str
):
    """Track which answers were helpful."""
    
    with open("feedback.jsonl", "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "helpful": helpful,
            "user_id": user_id
        }) + "\n")
    
    return {"status": "Feedback recorded"}
```

---

## Deployment Checklist

- [ ] Vector database set up and persistent
- [ ] All documentation indexed
- [ ] API endpoints tested with sample questions
- [ ] Query logging enabled
- [ ] Error handling in place
- [ ] Rate limiting configured
- [ ] Monitoring and alerts set up
- [ ] User feedback collection enabled
- [ ] Regular reindexing scheduled (daily)
- [ ] Documentation for end users
- [ ] Security (API keys, access control)
- [ ] Performance benchmarks established

---

## Chapter Summary

### What You've Learned

1. **RAG basics**: Retrieval + Generation = Grounded answers
2. **Vector embeddings**: Text → numbers → semantic search
3. **Vector databases**: Store and search embeddings
4. **RAG implementation**: Complete working system
5. **Advanced patterns**: Confidence, citations, multi-step
6. **Production deployment**: Enterprise-grade system
7. **Integration with Chapter 13**: Auto-updating documentation pipeline

### Key Metrics

| Metric | Before RAG | After RAG |
|--------|-----------|-----------|
| Time to find answer | 10+ minutes | < 1 minute |
| Answer accuracy | Variable | Grounded in docs |
| Documentation discovery | Manual search | Automatic retrieval |
| Tribal knowledge | High | Minimized |
| Onboarding time | 5 days | 2 days |

### Cost Analysis

**Monthly costs (500 device network)**:
- Vector database: $50-500 (depending on volume)
- API calls for answers: ~$10-50 (based on query volume)
- Total: $60-550/month
- **ROI**: Pays for itself in a week of engineer time saved

---

## Next Chapter

**Chapter 15: Building AI Agents** - Create autonomous systems that don't just answer questions, but take action.

---

## Resources

### Documentation
- [Chroma Vector Database](https://docs.trychroma.com/)
- [Pinecone RAG Guide](https://docs.pinecone.io/guides/retrieval-augmented-generation)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [Anthropic API](https://docs.anthropic.com/)

### Related Chapters
- Chapter 13: Network Documentation Basics
- Chapter 15: Building AI Agents
- Chapter 20: Vector Databases at Scale

### Further Reading
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- "Dense Passage Retrieval for Open-Domain Question Answering"

---

**Chapter 14 Complete** ✓

*Your documentation is only useful if people can find it. RAG makes it discoverable.*
