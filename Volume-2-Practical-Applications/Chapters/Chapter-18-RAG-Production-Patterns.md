# Chapter 18: RAG Production Patterns

## From Lab to Production

Building a RAG system that works in a demo is easy. Building one that works reliably for a team of 50 engineers across 3,000 devices — that's the real challenge. This chapter covers the patterns you need to move from a proof-of-concept to a production system.

**Networking analogy**: The difference between a lab RAG and a production RAG is like the difference between a GNS3 topology and a production network. The concepts are the same, but production needs redundancy, monitoring, change management, and the ability to handle unexpected inputs without crashing.

---

## Incremental Document Updates

Your network documentation changes constantly — new configs pushed, runbooks updated, post-mortems written. You can't re-embed your entire document corpus every time something changes.

### Change Detection: Know What Changed

**Networking analogy**: This is like how OSPF only floods LSA updates when something changes, not the entire LSDB. You track document hashes and only re-process what's actually different.

```python
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class DocumentChangeTracker:
    """Track document changes to avoid unnecessary re-embedding.

    Like OSPF's LSA sequence numbers — only process updates,
    not the entire database every time.
    """

    def __init__(self, state_file: str = "doc_state.json"):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load previous document state from disk."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self):
        """Persist document state to disk."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def compute_hash(self, content: str) -> str:
        """Compute content hash for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()

    def get_changed_documents(
        self, documents: Dict[str, str]
    ) -> Dict[str, str]:
        """Return only documents that have changed since last check.

        Args:
            documents: Dict mapping doc_id to content

        Returns:
            Dict of only changed/new documents
        """
        changed = {}

        for doc_id, content in documents.items():
            current_hash = self.compute_hash(content)
            stored_hash = self.state.get(doc_id, {}).get('hash')

            if current_hash != stored_hash:
                changed[doc_id] = content
                self.state[doc_id] = {
                    'hash': current_hash,
                    'last_updated': datetime.now().isoformat(),
                    'size': len(content)
                }

        self._save_state()
        return changed

    def get_deleted_documents(
        self, current_doc_ids: set
    ) -> List[str]:
        """Find documents that existed before but are now gone.

        Important for keeping the vector store clean — stale
        embeddings lead to irrelevant search results.
        """
        stored_ids = set(self.state.keys())
        deleted = stored_ids - current_doc_ids

        for doc_id in deleted:
            del self.state[doc_id]

        if deleted:
            self._save_state()

        return list(deleted)


# Usage
tracker = DocumentChangeTracker()

# Scan your documentation directory
all_docs = {}
for md_file in Path("./network-docs").glob("**/*.md"):
    all_docs[str(md_file)] = md_file.read_text()

# Only re-embed what changed
changed = tracker.get_changed_documents(all_docs)
deleted = tracker.get_deleted_documents(set(all_docs.keys()))

print(f"Total documents: {len(all_docs)}")
print(f"Changed/new: {len(changed)} (will re-embed)")
print(f"Deleted: {len(deleted)} (will remove from vector store)")
```

### Smart Chunking Updates

When a document changes, you don't necessarily need to re-embed every chunk. If only one section of a runbook changed, you can re-embed just that section's chunks.

```python
def get_changed_chunks(
    old_chunks: List[str],
    new_chunks: List[str]
) -> Dict[str, List]:
    """Compare old and new chunks to minimize re-embedding.

    Returns which chunks to add, remove, and keep.
    Like computing a routing table diff rather than doing a full
    SPF recalculation.
    """
    old_set = {hashlib.sha256(c.encode()).hexdigest(): c for c in old_chunks}
    new_set = {hashlib.sha256(c.encode()).hexdigest(): c for c in new_chunks}

    old_hashes = set(old_set.keys())
    new_hashes = set(new_set.keys())

    to_add = [new_set[h] for h in (new_hashes - old_hashes)]
    to_remove = [old_set[h] for h in (old_hashes - new_hashes)]
    unchanged = len(old_hashes & new_hashes)

    return {
        "add": to_add,
        "remove": to_remove,
        "unchanged_count": unchanged
    }
```

---

## Caching Strategies

Every API call costs money and takes time. Smart caching dramatically reduces both.

### Embedding Cache

**Networking analogy**: This is your ARP cache for the AI system. Just like you don't ARP for every single packet to the same destination, you don't re-embed text you've already embedded.

```python
import hashlib
import json
import os
from typing import List, Optional

class EmbeddingCache:
    """Cache embeddings to avoid redundant API calls.

    Like an ARP cache — store recently computed results so you
    don't have to make the same request twice.
    """

    def __init__(self, cache_dir: str = ".embedding_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, text: str, model: str) -> str:
        """Generate a cache key from text + model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Retrieve cached embedding if available."""
        key = self._cache_key(text, model)
        cache_path = os.path.join(self.cache_dir, f"{key}.json")

        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def set(self, text: str, model: str, embedding: List[float]):
        """Store embedding in cache."""
        key = self._cache_key(text, model)
        cache_path = os.path.join(self.cache_dir, f"{key}.json")

        with open(cache_path, 'w') as f:
            json.dump(embedding, f)

    def get_or_compute(
        self, text: str, model: str, embed_fn
    ) -> List[float]:
        """Get from cache or compute and cache the result."""
        cached = self.get(text, model)
        if cached is not None:
            return cached

        embedding = embed_fn(text)
        self.set(text, model, embedding)
        return embedding


# Usage with OpenAI embeddings
cache = EmbeddingCache()

def embed_text(text: str) -> List[float]:
    """Embed text, using cache when possible."""
    from openai import OpenAI
    openai_client = OpenAI()  # Uses OPENAI_API_KEY env var

    model = "text-embedding-3-small"
    return cache.get_or_compute(
        text=text,
        model=model,
        embed_fn=lambda t: openai_client.embeddings.create(
            input=t, model=model
        ).data[0].embedding
    )
```

### Query Result Cache

For queries that get asked repeatedly (e.g., "What's our MPLS architecture?"), cache the entire search result.

```python
from datetime import datetime, timedelta

class QueryCache:
    """Cache search results for frequently asked queries.

    Network teams often ask the same questions — "What's the
    standard VLAN layout?" or "How do I add a BGP peer?"
    Caching these saves API costs and speeds up responses.
    """

    def __init__(self, ttl_minutes: int = 60):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)

    def _is_expired(self, timestamp: datetime) -> bool:
        return datetime.now() - timestamp > self.ttl

    def get(self, query: str) -> Optional[dict]:
        """Get cached result if available and not expired."""
        key = query.lower().strip()
        if key in self.cache:
            entry = self.cache[key]
            if not self._is_expired(entry['timestamp']):
                entry['hits'] += 1
                return entry['result']
            else:
                del self.cache[key]
        return None

    def set(self, query: str, result: dict):
        """Cache a query result."""
        key = query.lower().strip()
        self.cache[key] = {
            'result': result,
            'timestamp': datetime.now(),
            'hits': 0
        }

    def get_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        total = len(self.cache)
        total_hits = sum(e['hits'] for e in self.cache.values())
        return {
            "cached_queries": total,
            "total_cache_hits": total_hits,
            "avg_hits_per_query": total_hits / total if total > 0 else 0
        }
```

---

## Error Handling and Resilience

Production systems must handle failures gracefully. API rate limits, timeouts, malformed responses — these all happen in the real world.

**Networking analogy**: This is your network's fault tolerance strategy. Just like you design networks with HSRP, BFD, and graceful restart, your RAG system needs retry logic, fallbacks, and circuit breakers.

### Retry with Exponential Backoff

```python
import time
import random
from anthropic import Anthropic, APIError, RateLimitError

def call_with_retry(
    fn,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0
):
    """Call a function with exponential backoff on failure.

    Like CSMA/CD in Ethernet — when there's a collision (error),
    wait a random time before retrying. Each retry waits longer
    (exponential backoff) to avoid overwhelming the system.
    """

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RateLimitError:
            if attempt == max_retries:
                raise
            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            print(f"  Rate limited. Retrying in {delay + jitter:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(delay + jitter)
        except APIError as e:
            if attempt == max_retries:
                raise
            if e.status_code and e.status_code >= 500:
                # Server error — retry
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                raise  # Client error — don't retry


# Usage
client = Anthropic()

def make_api_call():
    return client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Summarize this config..."}]
    )

result = call_with_retry(make_api_call)
```

### Circuit Breaker Pattern

When an API is consistently failing, stop hammering it. This prevents cascading failures and wasted API credits.

```python
class CircuitBreaker:
    """Stop calling a failing service after too many errors.

    Like BGP route dampening — if a route (service) keeps flapping,
    suppress it for a while to prevent instability. After a cooldown
    period, try again (half-open state).

    States:
        CLOSED: Normal operation, calls go through
        OPEN: Service is failing, calls are blocked
        HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    def call(self, fn, fallback=None):
        """Execute function through the circuit breaker."""

        if self.state == "OPEN":
            # Check if recovery timeout has passed
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF_OPEN"
                print("Circuit breaker: HALF_OPEN — testing recovery")
            else:
                if fallback:
                    return fallback()
                raise RuntimeError("Circuit breaker is OPEN — service unavailable")

        try:
            result = fn()
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
                print("Circuit breaker: CLOSED — service recovered")
            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                print(f"Circuit breaker: OPEN after {self.failures} failures")

            if fallback:
                return fallback()
            raise
```

---

## Monitoring and Observability

You can't improve what you can't measure. Track these metrics to understand how your RAG system is performing.

**Networking analogy**: This is your SNMP/telemetry for the AI system. Just like you monitor interface utilization, error rates, and latency on your network, you need to monitor search quality, response times, and costs.

### Key Metrics to Track

```python
from datetime import datetime
from typing import Dict, List
import json

class RAGMetrics:
    """Track RAG system performance metrics.

    Like SNMP counters for your AI system — tracks latency,
    quality, cost, and usage patterns to identify problems
    before users complain.
    """

    def __init__(self):
        self.queries = []

    def record_query(
        self,
        query: str,
        num_results: int,
        latency_ms: float,
        tokens_used: int,
        model: str,
        user_feedback: str = None
    ):
        """Record metrics for a single query."""
        self.queries.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_results": num_results,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "model": model,
            "user_feedback": user_feedback  # "helpful" / "not_helpful" / None
        })

    def get_summary(self) -> Dict:
        """Generate performance summary.

        Returns metrics similar to 'show interface' counters:
        - Total queries (packets processed)
        - Average latency (average forwarding delay)
        - Error rate (packet drops)
        - Token usage (bandwidth utilization)
        """
        if not self.queries:
            return {"status": "no data"}

        latencies = [q['latency_ms'] for q in self.queries]
        tokens = [q['tokens_used'] for q in self.queries]
        feedback = [q for q in self.queries if q['user_feedback']]
        helpful = [q for q in feedback if q['user_feedback'] == 'helpful']

        return {
            "total_queries": len(self.queries),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "total_tokens": sum(tokens),
            "estimated_cost_usd": sum(tokens) * 0.000003,  # Approximate
            "feedback_count": len(feedback),
            "satisfaction_rate": len(helpful) / len(feedback) if feedback else None,
            "queries_per_hour": self._queries_per_hour()
        }

    def _queries_per_hour(self) -> float:
        """Calculate query rate."""
        if len(self.queries) < 2:
            return 0
        first = datetime.fromisoformat(self.queries[0]['timestamp'])
        last = datetime.fromisoformat(self.queries[-1]['timestamp'])
        hours = (last - first).total_seconds() / 3600
        return len(self.queries) / hours if hours > 0 else 0

    def export_metrics(self, filepath: str):
        """Export metrics for analysis (e.g., to Grafana via JSON)."""
        with open(filepath, 'w') as f:
            json.dump({
                "summary": self.get_summary(),
                "queries": self.queries
            }, f, indent=2)
```

### What Good Looks Like

| Metric | Target | Red Flag |
|--------|--------|----------|
| Average latency | < 3 seconds | > 10 seconds |
| P95 latency | < 8 seconds | > 20 seconds |
| User satisfaction | > 80% helpful | < 50% helpful |
| Cache hit rate | > 30% | < 10% (cache not working) |
| Empty results rate | < 5% | > 20% (missing documents) |

---

## Multi-Tenant Access Control

In enterprise networks, not everyone should see everything. The security team's incident reports shouldn't appear when a junior engineer searches for VLAN configurations.

**Networking analogy**: This is RBAC (Role-Based Access Control) for your documentation — like how VRFs separate routing tables, you separate document access based on roles.

```python
from typing import List, Dict, Set

class TenantAwareRAG:
    """RAG system with role-based document access.

    Like VRF-lite for documentation — each role has its own
    'routing table' of accessible documents. A search in one
    VRF never returns results from another.
    """

    # Define which document tags each role can access
    ROLE_PERMISSIONS = {
        "network_engineer": {
            "configs", "runbooks", "standards", "topology",
            "change_requests", "post_mortems"
        },
        "security_analyst": {
            "configs", "security_policies", "incident_reports",
            "vulnerability_scans", "firewall_rules", "compliance"
        },
        "helpdesk": {
            "runbooks", "faq", "troubleshooting_guides"
        },
        "management": {
            "standards", "compliance", "architecture_docs",
            "capacity_reports"
        }
    }

    def __init__(self, all_documents: List[Dict]):
        self.all_documents = all_documents

    def search(
        self,
        query: str,
        user_role: str,
        search_fn
    ) -> List[Dict]:
        """Search with role-based filtering.

        First filters documents by role permissions, then searches
        only within the permitted set.
        """
        allowed_tags = self.ROLE_PERMISSIONS.get(user_role, set())

        # Filter documents to only those this role can access
        permitted_docs = [
            doc for doc in self.all_documents
            if doc.get('tags', set()) & allowed_tags
        ]

        # Search within permitted documents only
        results = search_fn(query, permitted_docs)

        # Add access metadata for audit logging
        for r in results:
            r['accessed_by_role'] = user_role

        return results
```

---

## Async Processing for Scale

When your team grows or your document corpus gets large, synchronous processing becomes a bottleneck. Async processing lets you handle multiple requests concurrently.

```python
import asyncio
from anthropic import AsyncAnthropic

class AsyncRAGPipeline:
    """Async RAG pipeline for handling concurrent requests.

    Like MPLS fast-reroute — multiple paths are pre-computed
    and ready to go. Requests don't wait in a single queue.
    """

    def __init__(self):
        self.client = AsyncAnthropic()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts concurrently."""
        # Process in batches of 10 to respect rate limits
        results = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i+10]
            tasks = [self._embed_single(text) for text in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results

    async def _embed_single(self, text: str) -> List[float]:
        """Embed a single text (placeholder — use your embedding provider)."""
        # In production, use OpenAI or another embedding API
        # This is a placeholder showing the async pattern
        await asyncio.sleep(0.1)  # Simulate API call
        return [0.0] * 1536  # Placeholder

    async def search_and_answer(
        self, query: str, context_docs: List[str]
    ) -> str:
        """Search and generate answer asynchronously."""
        context = "\n\n---\n\n".join(context_docs[:5])

        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"""Answer this network engineering question using
the provided context. If the context doesn't contain the answer, say so.

Question: {query}

Context:
{context}"""
            }]
        )

        return response.content[0].text

    async def handle_concurrent_queries(
        self, queries: List[str], context_docs: List[str]
    ) -> List[str]:
        """Handle multiple queries concurrently."""
        tasks = [
            self.search_and_answer(q, context_docs)
            for q in queries
        ]
        return await asyncio.gather(*tasks)
```

---

## Document Update Pipeline

Putting it all together — a production pipeline that watches for document changes, updates embeddings incrementally, and keeps the system current.

```python
class ProductionRAGPipeline:
    """Complete production RAG pipeline with incremental updates.

    This is your documentation system's 'routing protocol' —
    it detects topology changes (document updates), computes
    new routes (embeddings), and updates the forwarding table
    (vector store) automatically.
    """

    def __init__(self, docs_dir: str, vector_store):
        self.docs_dir = Path(docs_dir)
        self.tracker = DocumentChangeTracker()
        self.embedding_cache = EmbeddingCache()
        self.metrics = RAGMetrics()
        self.vector_store = vector_store

    def run_update_cycle(self):
        """Run one update cycle — detect changes and update embeddings.

        Call this on a schedule (e.g., every hour) or trigger it
        from a Git webhook when documentation is updated.
        """
        print(f"\n{'='*50}")
        print(f"RAG Update Cycle: {datetime.now()}")
        print(f"{'='*50}")

        # Step 1: Scan for documents
        all_docs = {}
        for md_file in self.docs_dir.glob("**/*.md"):
            all_docs[str(md_file)] = md_file.read_text()

        print(f"Total documents scanned: {len(all_docs)}")

        # Step 2: Find what changed
        changed = self.tracker.get_changed_documents(all_docs)
        deleted = self.tracker.get_deleted_documents(set(all_docs.keys()))

        print(f"Changed/new: {len(changed)}")
        print(f"Deleted: {len(deleted)}")

        # Step 3: Remove deleted documents from vector store
        for doc_id in deleted:
            self.vector_store.delete(doc_id)
            print(f"  Removed: {doc_id}")

        # Step 4: Re-embed changed documents
        for doc_id, content in changed.items():
            chunks = self._chunk_document(content)
            for i, chunk in enumerate(chunks):
                embedding = self.embedding_cache.get_or_compute(
                    text=chunk,
                    model="text-embedding-3-small",
                    embed_fn=lambda t: self._embed(t)
                )
                self.vector_store.upsert(
                    id=f"{doc_id}_chunk_{i}",
                    embedding=embedding,
                    metadata={"source": doc_id, "chunk_index": i}
                )
            print(f"  Updated: {doc_id} ({len(chunks)} chunks)")

        print(f"\nUpdate cycle complete.")

    def _chunk_document(self, content: str, chunk_size: int = 500) -> List[str]:
        """Split document into chunks for embedding."""
        words = content.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _embed(self, text: str) -> List[float]:
        """Generate embedding (placeholder — use your provider)."""
        from openai import OpenAI
        openai_client = OpenAI()
        response = openai_client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding
```

---

## Production Deployment Checklist

Before going live, verify these items:

### Infrastructure
- [ ] Vector store deployed (Pinecone, Qdrant, ChromaDB, or pgvector)
- [ ] API keys stored securely (environment variables or secrets manager)
- [ ] Embedding cache configured and tested
- [ ] Query result cache configured with appropriate TTL

### Reliability
- [ ] Retry logic with exponential backoff on all API calls
- [ ] Circuit breaker for external service failures
- [ ] Fallback responses when RAG is unavailable
- [ ] Error logging and alerting

### Security
- [ ] Role-based access control configured
- [ ] Sensitive documents tagged appropriately
- [ ] API keys rotated on schedule
- [ ] Audit logging for document access

### Monitoring
- [ ] Latency tracking (avg, P95, P99)
- [ ] Cost tracking (tokens per query, total spend)
- [ ] User satisfaction metrics
- [ ] Cache hit rate monitoring
- [ ] Empty result rate monitoring

### Operations
- [ ] Document update pipeline scheduled (hourly/daily)
- [ ] Runbook for common issues (stale cache, embedding failures)
- [ ] Capacity planning for growing document corpus

---

## Cost Optimization Summary

| Strategy | Savings | Implementation Effort |
|----------|---------|----------------------|
| Embedding cache | 40-60% of embedding costs | Low |
| Query result cache | 20-40% of generation costs | Low |
| Incremental updates | 80-95% of re-indexing costs | Medium |
| Haiku for classification | 10x cheaper than Sonnet | Low |
| Batch processing | 10-30% latency reduction | Medium |

For a mid-sized network team (50 engineers, 5,000 documents):
- Without optimization: ~$200-400/month
- With all optimizations: ~$30-80/month

---

## Key Takeaways

1. **Incremental updates** (OSPF LSA flooding): Only re-process documents that actually changed — don't rebuild the entire index.

2. **Caching** (ARP cache): Store embeddings and frequent query results to avoid redundant API calls.

3. **Resilience** (CSMA/CD + BGP dampening): Retry with backoff, use circuit breakers, and have fallback responses.

4. **Monitoring** (SNMP/telemetry): Track latency, cost, satisfaction, and cache hit rates to keep the system healthy.

5. **Access control** (VRF-lite): Separate document access by role to maintain security boundaries.

---

## What's Next

In **Chapter 19**, we'll build on these patterns to create intelligent agents that don't just search documentation — they actively troubleshoot problems, plan changes, and automate workflows.
