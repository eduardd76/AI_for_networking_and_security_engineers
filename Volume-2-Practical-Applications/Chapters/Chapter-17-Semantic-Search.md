# Chapter 17: Semantic Search for Network Documentation

## Why This Chapter Matters

You built a RAG system in Chapter 16. It works, but:

**The problem**: Search for "cloud connection setup" and it misses documents about "BGP peering with AWS" or "Direct Connect configuration." The keywords don't match, even though the meaning is the same.

**Without semantic search**:
```python
# Keyword search for "cloud connection"
results = keyword_search("cloud connection")
# Returns: 2 docs (only those with exact phrase "cloud connection")
```

**With semantic search**:
```python
# Semantic search understands meaning
results = semantic_search("cloud connection")
# Returns: 15 docs including:
# - "AWS Direct Connect setup"
# - "BGP peering with Azure"
# - "VPN tunnel to GCP"
# - "ExpressRoute configuration"
```

Semantic search understands that "cloud connection" relates to BGP, Direct Connect, VPN, and ExpressRoute.

**The networking equivalent**: Semantic search is like route summarization. Instead of exact prefix matching (10.1.1.0/24), you match intent (10.0.0.0/8 catches all 10.x.x.x traffic).

**What you'll build**:
- Hybrid search (keyword + semantic)
- Query expansion (one question becomes multiple)
- Re-ranking (sort by true relevance, not just similarity)
- Contextual compression (extract only relevant snippets)

This chapter takes your RAG system from "finds exact matches" to "understands what you mean."

---

## Section 1: Understanding Semantic Search

### The Limitation of Basic RAG

Chapter 16's RAG system uses **vector similarity**: documents close in embedding space are "similar."

**Problem**: Similarity ≠ Relevance

```python
# Query: "BGP configuration for AWS"
query_embedding = embeddings.embed_query("BGP configuration for AWS")

# Document 1: "BGP Configuration Guide" (similarity: 0.15)
# Document 2: "AWS Network Architecture" (similarity: 0.18)
# Document 3: "OSPF Configuration Guide" (similarity: 0.22)

# Returns OSPF doc because it's most "similar" to query!
```

Why? "Configuration Guide" appears in both query and OSPF doc, making them similar even though OSPF is wrong.

### What Semantic Search Adds

1. **Query expansion**: Turn one query into multiple variations
2. **Hybrid search**: Combine keyword (exact match) + semantic (meaning)
3. **Re-ranking**: Use LLM to score true relevance
4. **Compression**: Extract only relevant paragraphs, not full docs

**Result**: Find "BGP peering with AWS" even when you search for "cloud connection setup."

---

## Section 2: Building Semantic Search V1 → V4

### V1: Basic Vector Search (Baseline)

This is what you have from Chapter 16.

```python
# semantic_search_v1.py - Baseline from Chapter 16
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_anthropic import ChatAnthropic
import os

class SemanticSearchV1:
    """V1: Basic vector similarity search."""

    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

    def search(self, query: str, k: int = 5):
        """Simple similarity search."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return [
            {
                "content": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata
            }
            for doc, score in results
        ]


# Test it
if __name__ == "__main__":
    search = SemanticSearchV1()

    query = "How do I configure cloud connectivity?"
    results = search.search(query, k=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   {result['content'][:100]}...")
```

**Output**:
```
1. Score: 0.421
   Cloud Services Integration Guide
   This document covers integration with major cloud providers...

2. Score: 0.538
   Network Connectivity Options
   Various methods exist for connecting on-premise to cloud...

3. Score: 0.602
   VPN Configuration Standards
   VPN tunnels provide encrypted connectivity for remote access...
```

**Works**: Returns similar documents.

**Problem**: Misses "BGP peering with AWS" doc because it doesn't use the word "connectivity." Also, VPN doc scores better than BGP doc even though BGP is more relevant for cloud connections.

---

### V2: Add Hybrid Search (Keyword + Semantic)

Combine BM25 (keyword) with vector search using Reciprocal Rank Fusion (RRF).

```python
# semantic_search_v2.py - Hybrid search
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from typing import List
import os

class SemanticSearchV2:
    """V2: Hybrid search (BM25 + Vector)."""

    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

        # Load all docs for BM25
        self.all_docs = self._load_all_docs()

        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.all_docs)
        self.bm25_retriever.k = 5

    def _load_all_docs(self) -> List[Document]:
        """Load all documents from vector store."""
        # Get all docs
        results = self.vectorstore.get()

        docs = []
        for i in range(len(results['ids'])):
            doc = Document(
                page_content=results['documents'][i],
                metadata=results['metadatas'][i] if results['metadatas'] else {}
            )
            docs.append(doc)

        return docs

    def search(self, query: str, k: int = 5):
        """Hybrid search using RRF."""
        # Vector search
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # Ensemble with RRF
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]  # Equal weight to keyword and semantic
        )

        results = ensemble_retriever.get_relevant_documents(query)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "hybrid"
            }
            for doc in results[:k]
        ]


# Test it
if __name__ == "__main__":
    search = SemanticSearchV2()

    query = "AS 65001 BGP configuration"

    print("=== Hybrid Search Results ===")
    results = search.search(query, k=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['metadata'].get('filename', 'Unknown')}")
        print(f"   {result['content'][:100]}...")
```

**Output**:
```
=== Hybrid Search Results ===

1. bgp_config.pdf
   BGP Configuration for AS 65001

   To configure BGP for autonomous system 65001, use the following commands...

2. aws_direct_connect.pdf
   AWS Direct Connect BGP Setup

   Direct Connect requires BGP configuration. The AWS side uses AS 7224...

3. bgp_policy.pdf
   BGP Peering Policy

   All BGP sessions must use MD5 authentication and AS path filtering...
```

**Better**: BM25 catches "AS 65001" (exact match) that semantic search might miss. Hybrid combines both strengths.

**Problem**: Results aren't truly ranked by relevance. "BGP Peering Policy" might be less relevant than "AWS Direct Connect BGP Setup" for the specific query about AWS.

---

### V3: Add Re-ranking

Use cross-encoder to re-rank results by true relevance.

```python
# semantic_search_v3.py - Add re-ranking
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from typing import List
import numpy as np

class SemanticSearchV3:
    """V3: Hybrid search + re-ranking."""

    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

        # Load all docs for BM25
        self.all_docs = self._load_all_docs()

        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.all_docs)
        self.bm25_retriever.k = 20  # Retrieve more for re-ranking

        # Cross-encoder for re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _load_all_docs(self) -> List[Document]:
        """Load all documents from vector store."""
        results = self.vectorstore.get()

        docs = []
        for i in range(len(results['ids'])):
            doc = Document(
                page_content=results['documents'][i],
                metadata=results['metadatas'][i] if results['metadatas'] else {}
            )
            docs.append(doc)

        return docs

    def search(self, query: str, k: int = 5):
        """Hybrid search with re-ranking."""
        # Step 1: Hybrid retrieval (get top 20)
        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 20}
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )

        initial_results = ensemble_retriever.get_relevant_documents(query)

        # Step 2: Re-rank with cross-encoder
        if not initial_results:
            return []

        # Create query-doc pairs
        pairs = [[query, doc.page_content] for doc in initial_results]

        # Get relevance scores
        scores = self.reranker.predict(pairs)

        # Sort by relevance score
        scored_docs = list(zip(initial_results, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": float(score)
            }
            for doc, score in scored_docs[:k]
        ]


# Test it
if __name__ == "__main__":
    search = SemanticSearchV3()

    query = "How do I set up BGP peering with AWS Direct Connect?"

    print("=== Re-ranked Results ===")
    results = search.search(query, k=5)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Relevance: {result['relevance_score']:.3f}")
        print(f"   File: {result['metadata'].get('filename', 'Unknown')}")
        print(f"   Preview: {result['content'][:100]}...")
```

**Output**:
```
=== Re-ranked Results ===

1. Relevance: 0.872
   File: aws_direct_connect_bgp.pdf
   Preview: AWS Direct Connect BGP Configuration

   To establish BGP peering with AWS Direct Connect:
   1. Create a virtual interface...

2. Relevance: 0.745
   File: bgp_config_guide.pdf
   Preview: BGP Configuration Best Practices

   Before configuring BGP, ensure you have the peer AS number, IP addresses...

3. Relevance: 0.621
   File: cloud_connectivity.pdf
   Preview: Cloud Connectivity Options

   AWS Direct Connect provides dedicated network connections between your on-premise...

4. Relevance: 0.543
   File: bgp_policy.pdf
   Preview: BGP Peering Policy

   All external BGP sessions require approval from NetOps team...

5. Relevance: 0.412
   File: routing_protocols_overview.pdf
   Preview: Routing Protocols Overview

   BGP (Border Gateway Protocol) is used for inter-AS routing...
```

**Better**: Results are truly ranked by relevance! AWS Direct Connect BGP doc is #1 (most relevant), generic overview is #5 (least relevant).

**Problem**: Returns full documents. For a 50-page PDF, you get all 50 pages when you only need 2 paragraphs.

---

### V4: Add Contextual Compression

Extract only relevant snippets using LLM.

```python
# semantic_search_v4.py - Production with compression
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_anthropic import ChatAnthropic
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from typing import List, Dict
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSearchV4:
    """V4: Production semantic search with all features."""

    def __init__(self, api_key: str):
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

        # LLM for compression
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.0
        )

        # Load all docs for BM25
        self.all_docs = self._load_all_docs()

        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.all_docs)
        self.bm25_retriever.k = 20

        # Cross-encoder for re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Stats
        self.total_searches = 0
        self.total_docs_retrieved = 0
        self.total_docs_after_compression = 0

    def _load_all_docs(self) -> List[Document]:
        """Load all documents from vector store."""
        results = self.vectorstore.get()

        docs = []
        for i in range(len(results['ids'])):
            doc = Document(
                page_content=results['documents'][i],
                metadata=results['metadatas'][i] if results['metadatas'] else {}
            )
            docs.append(doc)

        logger.info(f"Loaded {len(docs)} documents for BM25")
        return docs

    def search(
        self,
        query: str,
        k: int = 5,
        use_compression: bool = True
    ) -> List[Dict]:
        """
        Production semantic search.

        Args:
            query: Search query
            k: Number of results to return
            use_compression: Whether to use LLM compression

        Returns:
            List of search results with metadata
        """
        logger.info(f"Searching for: {query}")

        try:
            # Step 1: Hybrid retrieval (top 20)
            vector_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 20}
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )

            initial_results = ensemble_retriever.get_relevant_documents(query)
            self.total_docs_retrieved += len(initial_results)

            logger.info(f"Retrieved {len(initial_results)} initial results")

            if not initial_results:
                return []

            # Step 2: Re-rank with cross-encoder
            pairs = [[query, doc.page_content] for doc in initial_results]
            scores = self.reranker.predict(pairs)

            scored_docs = list(zip(initial_results, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Take top k
            top_docs = [doc for doc, score in scored_docs[:k]]
            top_scores = [score for doc, score in scored_docs[:k]]

            logger.info(f"Re-ranked to top {len(top_docs)} results")

            # Step 3: Contextual compression (optional)
            if use_compression:
                compressor = LLMChainExtractor.from_llm(self.llm)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=vector_retriever
                )

                # Compress each doc
                compressed_results = []
                for doc, score in zip(top_docs, top_scores):
                    try:
                        # Extract relevant content
                        compressed = compressor.compress_documents(
                            [doc],
                            query
                        )

                        if compressed:
                            compressed_results.append({
                                "content": compressed[0].page_content,
                                "original_length": len(doc.page_content),
                                "compressed_length": len(compressed[0].page_content),
                                "compression_ratio": len(compressed[0].page_content) / len(doc.page_content),
                                "metadata": doc.metadata,
                                "relevance_score": float(score)
                            })
                    except Exception as e:
                        logger.warning(f"Compression failed: {e}")
                        # Fall back to original doc
                        compressed_results.append({
                            "content": doc.page_content,
                            "original_length": len(doc.page_content),
                            "compressed_length": len(doc.page_content),
                            "compression_ratio": 1.0,
                            "metadata": doc.metadata,
                            "relevance_score": float(score)
                        })

                self.total_docs_after_compression += len(compressed_results)
                logger.info(f"Compressed to {len(compressed_results)} relevant snippets")

                results = compressed_results
            else:
                # Return without compression
                results = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": float(score)
                    }
                    for doc, score in zip(top_docs, top_scores)
                ]

            self.total_searches += 1
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get search statistics."""
        return {
            "total_searches": self.total_searches,
            "avg_docs_retrieved": self.total_docs_retrieved / max(self.total_searches, 1),
            "avg_docs_after_compression": self.total_docs_after_compression / max(self.total_searches, 1),
            "avg_compression_ratio": self.total_docs_after_compression / max(self.total_docs_retrieved, 1) if self.total_docs_retrieved > 0 else 0
        }


# Test it
if __name__ == "__main__":
    search = SemanticSearchV4(api_key=os.getenv("ANTHROPIC_API_KEY"))

    query = "How do I configure BGP MD5 authentication for AWS peering?"

    print("=== Production Semantic Search ===\n")
    results = search.search(query, k=3, use_compression=True)

    for i, result in enumerate(results, 1):
        print(f"{i}. Relevance: {result['relevance_score']:.3f}")
        print(f"   File: {result['metadata'].get('filename', 'Unknown')}")

        if 'compression_ratio' in result:
            print(f"   Compression: {result['original_length']} → {result['compressed_length']} chars ({result['compression_ratio']:.1%})")

        print(f"   Content:\n   {result['content'][:200]}...\n")

    # Show stats
    stats = search.get_stats()
    print(f"\n=== Stats ===")
    print(f"Searches: {stats['total_searches']}")
    print(f"Avg docs retrieved: {stats['avg_docs_retrieved']:.1f}")
    print(f"Avg docs after compression: {stats['avg_docs_after_compression']:.1f}")
    print(f"Compression ratio: {stats['avg_compression_ratio']:.1%}")
```

**Output**:
```
INFO:__main__:Loaded 147 documents for BM25
INFO:__main__:Searching for: How do I configure BGP MD5 authentication for AWS peering?
INFO:__main__:Retrieved 20 initial results
INFO:__main__:Re-ranked to top 3 results
INFO:__main__:Compressed to 3 relevant snippets

=== Production Semantic Search ===

1. Relevance: 0.891
   File: aws_bgp_auth.pdf
   Compression: 2847 → 412 chars (14.5%)
   Content:
   BGP MD5 Authentication for AWS Direct Connect

   Configure MD5 authentication on your router:

   router bgp 65001
    neighbor 169.254.1.1 remote-as 7224
    neighbor 169.254.1.1 password MySecurePassword123

   AWS requires MD5 for all BGP sessions. The password must match on both sides...

2. Relevance: 0.763
   File: bgp_security_best_practices.pdf
   Compression: 5231 → 387 chars (7.4%)
   Content:
   MD5 Authentication

   Always enable MD5 authentication for external BGP peers:
   - Prevents TCP session hijacking
   - Authenticates routing updates
   - Required by most enterprise security policies

   Configuration example...

3. Relevance: 0.621
   File: aws_direct_connect_setup.pdf
   Compression: 3912 → 523 chars (13.4%)
   Content:
   Step 4: Configure BGP

   After creating the virtual interface, configure BGP on your router. AWS provides:
   - Peer IP: 169.254.x.x
   - AWS ASN: 7224
   - Your ASN: (specified during VIF creation)
   - BGP authentication key: (from AWS console)

   Apply the authentication key using the neighbor password command...

=== Stats ===
Searches: 1
Avg docs retrieved: 20.0
Avg docs after compression: 3.0
Compression ratio: 15.0%
```

**Production-ready features**:
- ✓ Hybrid search (keyword + semantic)
- ✓ Re-ranking with cross-encoder
- ✓ Contextual compression (2847 chars → 412 chars = 85% reduction)
- ✓ Logging and error handling
- ✓ Statistics tracking

**Evolution summary**:
- **V1**: Basic vector search (similarity only)
- **V2**: Hybrid search (keyword + semantic)
- **V3**: Re-ranking (true relevance)
- **V4**: Compression (extract only relevant snippets) + production features

Start with V1 to understand. Build V4 for production.

---

## Lab 1: Build Hybrid Search

**Objective**: Implement hybrid search combining BM25 and vector search to catch both exact matches and semantic matches.

**Time**: 45 minutes | **Cost**: $0

**Success Criteria**:
- [ ] BM25 finds exact matches (AS numbers, IP addresses)
- [ ] Vector search finds semantic matches (concepts, synonyms)
- [ ] Hybrid outperforms either method alone
- [ ] Test with 10 queries showing improvement

### Quick Start

Use V2 code from Section 2. Test with these query types:

**Exact match test**: "AS 65001 BGP configuration"
- BM25 should rank docs with "AS 65001" at top
- Vector alone might miss the specific AS number

**Semantic test**: "How do I connect to cloud services?"
- Vector should find BGP, Direct Connect, VPN docs
- BM25 alone would miss these (different keywords)

### Verification

<details>
<summary>Q: Why use both BM25 and vector search?</summary>

**Answer**: They catch different things:
- **BM25**: AS numbers (65001), IP addresses (10.1.1.1), exact commands (router bgp)
- **Vector**: Synonyms (cloud = AWS/Azure/GCP), concepts (connectivity = BGP/VPN/DirectConnect)
- **Hybrid**: Combines both using RRF (Reciprocal Rank Fusion)

Example: Query "AS 65001 cloud connection"
- BM25 finds: Docs with exact "AS 65001"
- Vector finds: Docs about cloud connectivity
- Hybrid finds: Docs with AS 65001 AND about cloud (best of both)
</details>

---

## Lab 2: Add Re-ranking

**Objective**: Use cross-encoder to re-rank results by true relevance, not just similarity.

**Time**: 60 minutes | **Cost**: $0 (local model)

**Success Criteria**:
- [ ] Re-ranking moves most relevant docs to top
- [ ] Relevance scores clearly differentiate good vs poor matches
- [ ] Processing time < 500ms for 20 docs

### Implementation

Use V3 code from Section 2. Compare before/after re-ranking.

### Benchmark

Test with: "How do I configure BGP MD5 authentication for AWS?"

**Expected improvement**:
- Before: Generic BGP doc at position 1, AWS-specific at position 3
- After: AWS BGP MD5 doc at position 1, generic at position 5

---

## Lab 3: Production Semantic Search

**Objective**: Build complete production system with compression.

**Time**: 90 minutes | **Cost**: $0.30

**Success Criteria**:
- [ ] Compression reduces token usage by 70-90%
- [ ] Full error handling and logging
- [ ] Stats tracking (searches, compression ratio)
- [ ] 10 test queries complete successfully

### Implementation

Use V4 code from Section 2. This is your production system.

### Extensions

1. **Query analytics**: Log queries with zero results
2. **A/B testing**: Compare V2 vs V4 on same queries
3. **Caching**: Cache compressed snippets for common queries
4. **Async**: Process multiple queries concurrently

---

## Check Your Understanding

<details>
<summary>Q1: Bi-encoder vs Cross-encoder for re-ranking?</summary>

**Answer**:

**Bi-encoder**: Encode query and docs separately, compare with cosine similarity
- Fast: Pre-compute doc embeddings
- Less accurate: Doesn't model query-doc interaction

**Cross-encoder**: Encode query+doc together, output relevance score
- Slow: Must process each query-doc pair
- More accurate: Models interaction directly

**Production pattern**: Bi-encoder retrieves top 100 (fast), cross-encoder re-ranks top 20 (accurate).
</details>

<details>
<summary>Q2: When to use contextual compression?</summary>

**Answer**:

**Use when**:
- Large documents (> 1000 tokens)
- Want specific answers, not full docs
- Cost matters (compression saves LLM input tokens)

**Skip when**:
- Already small chunks (< 500 tokens)
- Need full context
- Latency critical (compression adds 100-300ms)

**ROI**: For 50-page PDF, compression extracts 2 relevant paragraphs → 95% token reduction.
</details>

<details>
<summary>Q3: How does RRF work in hybrid search?</summary>

**Answer**:

RRF (Reciprocal Rank Fusion) formula:
```
score = 1/(k + rank_BM25) + 1/(k + rank_vector)
```

Where k=60 (constant).

**Example**:
- Doc A: BM25 rank 1, vector rank 3 → score = 1/61 + 1/63 = 0.032
- Doc B: BM25 rank 5, vector rank 2 → score = 1/65 + 1/62 = 0.031

Doc A wins (appears high in both rankings).
</details>

<details>
<summary>Q4: Speed vs quality trade-offs?</summary>

**Answer**:

| Method | Speed | Quality |
|--------|-------|---------|
| Vector only | 50ms | Medium |
| Hybrid | 80ms | Good |
| + Re-ranking | 280ms | Excellent |
| + Compression | 480ms | Excellent (focused) |

**Optimization**: Use fast method first, upgrade if confidence is low.
</details>

---

## Lab Time Budget and ROI

### Time Investment

| Lab | Coding | Testing | Total | Difficulty |
|-----|--------|---------|-------|------------|
| **Lab 1: Hybrid** | 25 min | 20 min | **45 min** | Intermediate |
| **Lab 2: Re-ranking** | 35 min | 25 min | **60 min** | Advanced |
| **Lab 3: Production** | 60 min | 30 min | **90 min** | Advanced |
| **Total** | **120 min** | **75 min** | **3 hours 15 min** | - |

### Cost Breakdown

| Component | Cost | Notes |
|-----------|------|-------|
| Embeddings (all-MiniLM-L6-v2) | $0 | Local model |
| BM25 keyword search | $0 | Algorithm only |
| Cross-encoder re-ranking | $0 | Local model (80MB) |
| LLM compression (Claude) | $0.30 | 20 queries × $0.015 |
| **Total** | **$0.30** | For all labs |

### Production Deployment

| Daily Usage | Feature | Monthly Cost |
|-------------|---------|--------------|
| 500 queries | Hybrid search | $0 |
| 500 queries | + Re-ranking | $0 |
| 100 queries | + Compression | $90 |
| **Total** | **500/day** | **$90/month** |

**With caching** (90% hit rate): $9/month

### Business Value

**Time savings per search**:
- Before: 15 min manual doc review
- After: 30 sec with compressed answer
- Savings: 14.5 minutes

**Monthly value** (50 searches/day):
- 50 × 14.5 min × 21 days = 15,225 min/month = 254 hours
- 254 hours × $40/hr = **$10,160/month**

**ROI**: ($10,160 - $90) / $90 = **11,111% monthly ROI**

**Payback period**: < 1 hour of first day

---

## Key Takeaways

1. **Hybrid search combines strengths** - BM25 catches exact terms, vector understands meaning
2. **Re-ranking improves quality** - Cross-encoder scores true relevance, not just similarity
3. **Compression reduces costs** - 85% token reduction by extracting relevant snippets
4. **Trade speed for quality** - 50ms (vector only) → 480ms (full pipeline) but 3x better results
5. **Local models save money** - Only pay for final LLM compression step

Semantic search transforms RAG from "finds keywords" to "understands intent."

Next chapter: RAG production patterns and scaling.

