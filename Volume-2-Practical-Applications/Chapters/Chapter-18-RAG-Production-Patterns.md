# Chapter 18: RAG Production Patterns

## Why This Chapter Matters

You've built a RAG system. It works great with 100 documents. Now you have 10,000 documents, 50 concurrent users, and documents updating daily.

**Production challenges**:
- **Performance**: Queries take 5+ seconds
- **Freshness**: Docs updated, RAG shows old answers
- **Scale**: Vector DB growing unbounded
- **Cost**: Re-embedding entire corpus daily is expensive

This chapter covers production patterns that handle real-world scale:
- Incremental document updates
- Caching strategies
- Async processing
- Monitoring and observability
- Multi-tenant RAG systems

---

## Section 1: Incremental Document Updates

### The Naive Approach (Don't Do This)

```python
# BAD: Re-embed everything when one doc changes
def update_documentation():
    # Delete all vectors
    vectorstore.delete_collection()

    # Re-embed ALL documents (even unchanged ones)
    all_docs = load_all_documents()  # 10,000 docs
    vectorstore.add_documents(all_docs)

    # This takes 30+ minutes for large corpora
    # During this time, RAG is down or serving stale data
```

### Production Approach: Incremental Updates

```python
# incremental_updater.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from typing import List, Dict
import hashlib
import json
import sqlite3
from pathlib import Path
from datetime import datetime

class IncrementalDocumentUpdater:
    """Track document changes and update only what changed."""

    def __init__(self, vectorstore_path: str = "./chroma_db"):
        self.vectorstore_path = vectorstore_path
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=self.embeddings
        )

        # Track document hashes
        self.tracker_db = sqlite3.connect("doc_tracker.db")
        self._init_tracker()

    def _init_tracker(self):
        """Initialize document tracking database."""
        cursor = self.tracker_db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_tracking (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                chunk_ids TEXT NOT NULL
            )
        """)
        self.tracker_db.commit()

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_stored_hash(self, file_path: str) -> str:
        """Get previously stored hash for file."""
        cursor = self.tracker_db.cursor()
        cursor.execute(
            "SELECT content_hash FROM document_tracking WHERE file_path = ?",
            (file_path,)
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def _store_hash(
        self,
        file_path: str,
        content_hash: str,
        chunk_ids: List[str]
    ):
        """Store hash and chunk IDs for file."""
        cursor = self.tracker_db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO document_tracking
            (file_path, content_hash, last_updated, chunk_ids)
            VALUES (?, ?, ?, ?)
        """, (
            file_path,
            content_hash,
            datetime.now().isoformat(),
            json.dumps(chunk_ids)
        ))
        self.tracker_db.commit()

    def _delete_old_chunks(self, file_path: str):
        """Delete old chunks for a file."""
        cursor = self.tracker_db.cursor()
        cursor.execute(
            "SELECT chunk_ids FROM document_tracking WHERE file_path = ?",
            (file_path,)
        )
        result = cursor.fetchone()

        if result:
            old_chunk_ids = json.loads(result[0])

            # Delete from vector store
            if old_chunk_ids:
                self.vectorstore.delete(ids=old_chunk_ids)

            print(f"Deleted {len(old_chunk_ids)} old chunks for {file_path}")

    def update_document(
        self,
        file_path: str,
        content: str,
        chunks: List[str],
        metadata: Dict = None
    ) -> Dict[str, any]:
        """
        Update a single document.

        Args:
            file_path: Path to document
            content: Full document content
            chunks: List of text chunks
            metadata: Optional metadata for chunks

        Returns:
            Dict with update status
        """
        # Compute content hash
        content_hash = self._compute_hash(content)
        stored_hash = self._get_stored_hash(file_path)

        # Check if changed
        if content_hash == stored_hash:
            return {
                "status": "unchanged",
                "file_path": file_path,
                "chunks_updated": 0
            }

        # Delete old chunks
        self._delete_old_chunks(file_path)

        # Add new chunks
        chunk_metadata = metadata or {}
        chunk_metadata.update({
            "file_path": file_path,
            "content_hash": content_hash
        })

        # Add to vector store and get IDs
        chunk_ids = self.vectorstore.add_texts(
            texts=chunks,
            metadatas=[chunk_metadata] * len(chunks)
        )

        # Update tracker
        self._store_hash(file_path, content_hash, chunk_ids)

        return {
            "status": "updated",
            "file_path": file_path,
            "chunks_updated": len(chunks),
            "chunk_ids": chunk_ids
        }

    def update_directory(self, directory_path: str) -> Dict[str, any]:
        """Update all documents in directory."""
        from document_loader import NetworkDocumentLoader

        loader = NetworkDocumentLoader()

        directory = Path(directory_path)
        stats = {
            "total_files": 0,
            "updated": 0,
            "unchanged": 0,
            "errors": 0
        }

        for file_path in directory.rglob("*.md"):
            stats["total_files"] += 1

            try:
                # Load document
                docs = loader.load_text(str(file_path))
                if not docs:
                    continue

                content = docs[0].page_content

                # Chunk
                chunks = loader.chunk_documents(docs)
                chunk_texts = [chunk.page_content for chunk in chunks]

                # Update
                result = self.update_document(
                    str(file_path),
                    content,
                    chunk_texts,
                    metadata={"source_type": "markdown"}
                )

                if result["status"] == "updated":
                    stats["updated"] += 1
                    print(f"✓ Updated: {file_path.name} ({result['chunks_updated']} chunks)")
                else:
                    stats["unchanged"] += 1

            except Exception as e:
                stats["errors"] += 1
                print(f"✗ Error: {file_path.name}: {e}")

        return stats


# Example usage
if __name__ == "__main__":
    updater = IncrementalDocumentUpdater()

    # Update entire directory
    stats = updater.update_directory("./network_docs")

    print(f"\nUpdate complete:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Updated: {stats['updated']}")
    print(f"  Unchanged: {stats['unchanged']}")
    print(f"  Errors: {stats['errors']}")

    # Only changed files were re-embedded!
    # Saves 90%+ of processing time
```

---

## Section 2: Caching Strategies

### Response Caching

```python
# rag_cache.py
import hashlib
import json
from typing import Dict, Optional
from datetime import datetime, timedelta

class RAGCache:
    """Cache RAG responses to reduce costs."""

    def __init__(self, ttl_hours: int = 24):
        self.cache = {}  # In production: use Redis
        self.ttl = timedelta(hours=ttl_hours)

    def _generate_key(self, query: str, metadata: Dict = None) -> str:
        """Generate cache key from query."""
        key_data = {
            "query": query.lower().strip(),
            "metadata": metadata or {}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, metadata: Dict = None) -> Optional[Dict]:
        """Get cached response."""
        key = self._generate_key(query, metadata)

        if key in self.cache:
            cached = self.cache[key]

            # Check expiration
            cached_time = datetime.fromisoformat(cached["timestamp"])
            if datetime.now() - cached_time < self.ttl:
                cached["cache_hit"] = True
                return cached

            # Expired, remove
            del self.cache[key]

        return None

    def set(self, query: str, response: Dict, metadata: Dict = None):
        """Cache a response."""
        key = self._generate_key(query, metadata)

        self.cache[key] = {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "cache_hit": False
        }

    def invalidate(self, pattern: str = None):
        """Invalidate cache entries."""
        if pattern is None:
            # Clear all
            self.cache.clear()
        else:
            # Remove matching keys
            keys_to_remove = [
                k for k, v in self.cache.items()
                if pattern.lower() in v["query"].lower()
            ]
            for key in keys_to_remove:
                del self.cache[key]


# Integration with RAG system
class CachedRAG:
    """RAG system with response caching."""

    def __init__(self, rag_system, cache_ttl_hours: int = 24):
        self.rag = rag_system
        self.cache = RAGCache(ttl_hours=cache_ttl_hours)

    def query(self, question: str, **kwargs) -> Dict:
        """Query with caching."""
        # Check cache
        cached = self.cache.get(question, metadata=kwargs)
        if cached:
            print(f"✓ Cache hit for: {question[:50]}...")
            return cached["response"]

        # Cache miss, query RAG
        print(f"✗ Cache miss, querying RAG...")
        response = self.rag.query(question, **kwargs)

        # Cache response
        self.cache.set(question, response, metadata=kwargs)

        return response

    def invalidate_cache_for_document(self, document_path: str):
        """Invalidate cache when document updates."""
        # Invalidate all queries mentioning keywords from this doc
        # In production: maintain query-to-document mapping
        self.cache.invalidate()


# Example
"""
First query: "What's our BGP policy?"
→ Cache miss
→ Query RAG (2 seconds, costs $0.02)
→ Cache response

Same query 5 minutes later:
→ Cache hit
→ Return cached response (0.001 seconds, costs $0)

Document updates:
→ Invalidate cache
→ Next query is cache miss, gets fresh data
"""
```

---

## Section 3: Async Processing for Performance

### Sync vs Async RAG

```python
# async_rag.py
import asyncio
from typing import List, Dict
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma

class AsyncRAG:
    """Async RAG for concurrent queries."""

    def __init__(self, api_key: str, vectorstore: Chroma):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key
        )
        self.vectorstore = vectorstore

    async def query_single(self, question: str, k: int = 5) -> Dict:
        """Query single question asynchronously."""
        # Retrieve documents (sync, but fast)
        docs = self.vectorstore.similarity_search(question, k=k)

        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate answer (async)
        prompt = f"""Answer based on this context:
{context}

Question: {question}
Answer:"""

        # Anthropic SDK doesn't have native async, so we use asyncio.to_thread
        response = await asyncio.to_thread(
            self.llm.invoke,
            prompt
        )

        return {
            "question": question,
            "answer": response.content,
            "sources": [doc.metadata for doc in docs]
        }

    async def query_batch(self, questions: List[str]) -> List[Dict]:
        """Query multiple questions concurrently."""
        tasks = [self.query_single(q) for q in questions]
        results = await asyncio.gather(*tasks)
        return results


# Example usage
async def main():
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    rag = AsyncRAG(api_key="your-api-key", vectorstore=vectorstore)

    # Query 5 questions concurrently
    questions = [
        "What's our BGP policy?",
        "How do I configure VLANs?",
        "What are OSPF area requirements?",
        "What's the ACL standard?",
        "How to enable SSH?"
    ]

    import time
    start = time.time()

    results = await rag.query_batch(questions)

    elapsed = time.time() - start

    print(f"Answered {len(questions)} questions in {elapsed:.1f}s")
    print(f"Average: {elapsed/len(questions):.1f}s per question")

    # Sync version would take ~10s (2s × 5)
    # Async version takes ~2.5s (parallelized)

    for result in results:
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Section 4: Monitoring and Observability

### Production Monitoring

```python
# rag_monitor.py
import time
import sqlite3
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict

@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    timestamp: str
    question: str
    query_hash: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    num_docs_retrieved: int
    answer_length: int
    cache_hit: bool
    tokens_used: int
    cost: float
    user_id: str = None
    success: bool = True
    error: str = None

class RAGMonitor:
    """Monitor RAG system performance and usage."""

    def __init__(self, db_path: str = "rag_metrics.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize metrics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_metrics (
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                retrieval_time_ms REAL,
                generation_time_ms REAL,
                total_time_ms REAL,
                num_docs_retrieved INTEGER,
                answer_length INTEGER,
                cache_hit BOOLEAN,
                tokens_used INTEGER,
                cost REAL,
                user_id TEXT,
                success BOOLEAN,
                error TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON query_metrics(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_hash
            ON query_metrics(query_hash)
        """)

        conn.commit()
        conn.close()

    def log_query(self, metrics: QueryMetrics):
        """Log query metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO query_metrics VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            metrics.timestamp,
            metrics.question,
            metrics.query_hash,
            metrics.retrieval_time_ms,
            metrics.generation_time_ms,
            metrics.total_time_ms,
            metrics.num_docs_retrieved,
            metrics.answer_length,
            metrics.cache_hit,
            metrics.tokens_used,
            metrics.cost,
            metrics.user_id,
            metrics.success,
            metrics.error
        ))

        conn.commit()
        conn.close()

    def get_stats(self, hours: int = 24) -> Dict:
        """Get system statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Time window
        start_time = datetime.now().replace(microsecond=0).isoformat()

        # Total queries
        cursor.execute("""
            SELECT COUNT(*) FROM query_metrics
            WHERE timestamp >= datetime('now', ?)
        """, (f'-{hours} hours',))
        total_queries = cursor.fetchone()[0]

        # Success rate
        cursor.execute("""
            SELECT
                SUM(CASE WHEN success THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
            FROM query_metrics
            WHERE timestamp >= datetime('now', ?)
        """, (f'-{hours} hours',))
        success_rate = cursor.fetchone()[0] or 0

        # Cache hit rate
        cursor.execute("""
            SELECT
                SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
            FROM query_metrics
            WHERE timestamp >= datetime('now', ?)
        """, (f'-{hours} hours',))
        cache_hit_rate = cursor.fetchone()[0] or 0

        # Average latency
        cursor.execute("""
            SELECT AVG(total_time_ms)
            FROM query_metrics
            WHERE timestamp >= datetime('now', ?) AND success = 1
        """, (f'-{hours} hours',))
        avg_latency = cursor.fetchone()[0] or 0

        # P95 latency
        cursor.execute("""
            SELECT total_time_ms
            FROM query_metrics
            WHERE timestamp >= datetime('now', ?) AND success = 1
            ORDER BY total_time_ms
            LIMIT 1 OFFSET (
                SELECT COUNT(*) * 95 / 100
                FROM query_metrics
                WHERE timestamp >= datetime('now', ?) AND success = 1
            )
        """, (f'-{hours} hours', f'-{hours} hours'))
        p95_result = cursor.fetchone()
        p95_latency = p95_result[0] if p95_result else 0

        # Total cost
        cursor.execute("""
            SELECT SUM(cost)
            FROM query_metrics
            WHERE timestamp >= datetime('now', ?)
        """, (f'-{hours} hours',))
        total_cost = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "time_window_hours": hours,
            "total_queries": total_queries,
            "success_rate": f"{success_rate:.1f}%",
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "avg_latency_ms": round(avg_latency, 1),
            "p95_latency_ms": round(p95_latency, 1),
            "total_cost": f"${total_cost:.4f}"
        }

    def get_slow_queries(self, threshold_ms: float = 5000, limit: int = 10) -> List[Dict]:
        """Get slowest queries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT question, total_time_ms, timestamp
            FROM query_metrics
            WHERE total_time_ms > ? AND success = 1
            ORDER BY total_time_ms DESC
            LIMIT ?
        """, (threshold_ms, limit))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "question": row[0],
                "latency_ms": row[1],
                "timestamp": row[2]
            }
            for row in rows
        ]


# Integration with RAG
class MonitoredRAG:
    """RAG system with monitoring."""

    def __init__(self, rag_system):
        self.rag = rag_system
        self.monitor = RAGMonitor()

    def query(self, question: str, user_id: str = None) -> Dict:
        """Query with monitoring."""
        import hashlib

        query_hash = hashlib.md5(question.encode()).hexdigest()
        start_total = time.time()

        try:
            # Retrieval phase
            start_retrieval = time.time()
            docs = self.rag.vectorstore.similarity_search(question, k=5)
            retrieval_time = (time.time() - start_retrieval) * 1000

            # Generation phase
            start_generation = time.time()
            result = self.rag.query(question)
            generation_time = (time.time() - start_generation) * 1000

            total_time = (time.time() - start_total) * 1000

            # Log metrics
            metrics = QueryMetrics(
                timestamp=datetime.now().isoformat(),
                question=question[:200],  # Truncate long questions
                query_hash=query_hash,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=total_time,
                num_docs_retrieved=len(docs),
                answer_length=len(result['answer']),
                cache_hit=result.get('cache_hit', False),
                tokens_used=result.get('tokens_used', 0),
                cost=result.get('cost', 0.0),
                user_id=user_id,
                success=True
            )

            self.monitor.log_query(metrics)

            return result

        except Exception as e:
            # Log error
            metrics = QueryMetrics(
                timestamp=datetime.now().isoformat(),
                question=question[:200],
                query_hash=query_hash,
                retrieval_time_ms=0,
                generation_time_ms=0,
                total_time_ms=(time.time() - start_total) * 1000,
                num_docs_retrieved=0,
                answer_length=0,
                cache_hit=False,
                tokens_used=0,
                cost=0.0,
                user_id=user_id,
                success=False,
                error=str(e)
            )

            self.monitor.log_query(metrics)
            raise


# Example usage
if __name__ == "__main__":
    monitor = RAGMonitor()

    # Get stats
    stats = monitor.get_stats(hours=24)
    print("RAG System Stats (last 24h):")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get slow queries
    slow = monitor.get_slow_queries(threshold_ms=3000)
    print(f"\nSlow queries (>3s):")
    for query in slow:
        print(f"  {query['latency_ms']:.0f}ms: {query['question'][:60]}...")
```

---

## Section 5: Multi-Tenant RAG

### Supporting Multiple Organizations

```python
# multi_tenant_rag.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from typing import Dict, List

class MultiTenantRAG:
    """RAG system supporting multiple tenants."""

    def __init__(self, base_path: str = "./vectorstores"):
        self.base_path = base_path
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.tenant_stores = {}

    def _get_tenant_store(self, tenant_id: str) -> Chroma:
        """Get or create vector store for tenant."""
        if tenant_id not in self.tenant_stores:
            store_path = f"{self.base_path}/{tenant_id}"

            self.tenant_stores[tenant_id] = Chroma(
                persist_directory=store_path,
                embedding_function=self.embeddings
            )

        return self.tenant_stores[tenant_id]

    def add_documents(
        self,
        tenant_id: str,
        documents: List[str],
        metadatas: List[Dict] = None
    ):
        """Add documents for a specific tenant."""
        store = self._get_tenant_store(tenant_id)

        # Add tenant_id to metadata
        if metadatas is None:
            metadatas = [{}] * len(documents)

        for metadata in metadatas:
            metadata['tenant_id'] = tenant_id

        store.add_texts(texts=documents, metadatas=metadatas)
        store.persist()

    def query(
        self,
        tenant_id: str,
        question: str,
        k: int = 5
    ) -> Dict:
        """Query tenant-specific documentation."""
        store = self._get_tenant_store(tenant_id)

        # Search with tenant filter
        docs = store.similarity_search(
            question,
            k=k,
            filter={"tenant_id": tenant_id}
        )

        # Generate answer using tenant docs only
        context = "\n\n".join([doc.page_content for doc in docs])

        return {
            "tenant_id": tenant_id,
            "question": question,
            "documents": docs,
            "context": context
        }


# Example usage
"""
# Tenant A (Company A)
rag.add_documents(
    tenant_id="company_a",
    documents=["Company A BGP policy...", "Company A VLAN standards..."]
)

# Tenant B (Company B)
rag.add_documents(
    tenant_id="company_b",
    documents=["Company B BGP policy...", "Company B VLAN standards..."]
)

# Query for Company A - only sees their docs
result_a = rag.query(tenant_id="company_a", question="What's our BGP policy?")

# Query for Company B - only sees their docs
result_b = rag.query(tenant_id="company_b", question="What's our BGP policy?")
```

---

## What Can Go Wrong

**1. Vector DB corruption**
- ChromaDB files get corrupted
- Lost all embeddings
- Solution: Regular backups, use managed service

**2. Memory leaks with large queries**
- Loading 10,000 docs into memory
- System crashes
- Solution: Pagination, streaming results

**3. Inconsistent cache state**
- Document updates but cache not invalidated
- Users get stale answers
- Solution: Cache invalidation on document change

**4. Token limit exceeded**
- Too many retrieved docs exceed context
- API error
- Solution: Limit total context tokens, compress

**5. Cross-tenant data leakage**
- Metadata filtering fails
- Tenant A sees Tenant B's docs
- Solution: Separate vector stores per tenant

---

## Key Takeaways

1. **Incremental updates** save 90%+ of processing time
2. **Caching** reduces cost and latency dramatically
3. **Async processing** enables concurrent queries
4. **Monitoring** reveals performance bottlenecks
5. **Multi-tenancy** requires careful data isolation

### Core Concepts Expanded

- **RAG (Retrieval-Augmented Generation)**: A blend of retrieval and generation, RAG combines the ability to fetch relevant data from a vast repository with the generation of coherent text. This ensures the AI provides contextually accurate and informative responses.

- **Vector Store**: Think of a vector store as a smart filing system where data is stored to enable quick and meaningful searches. It converts complex information into vectors (numerical representations) that cluster similar information together.

- **Embedding**: Embeddings are like the unique fingerprints for words and sentences. They convert these into numbers that machines comprehend, capturing their essence and context. This allows AI to recognize similarities and differences, understand nuances, and deliver precise results.

#### Importance of Fine-Tuning Embedding Models

Embedding models translate human language into a form that AI systems can understand. Fine-tuning these models is like refining an artist's palette:

- **Customization**: Tailor embeddings to focus on specific needs, such as industry jargon or special terminologies.
- **Precision**: Enhance accuracy by concentrating on relevant areas and discarding unnecessary data. It helps AI differentiate subtle context changes.
- **Adaptation**: As language evolves, embedding models can adjust to incorporate new words, meanings, or contexts, ensuring AI stays current and effective.

Fine-tuning is crucial for crafting an AI that truly understands the unique and changing language of any field, making interactions smoother and more accurate.

Your RAG system is now production-ready.

Next chapter: Building intelligent troubleshooting agents.
