# Chapter 17: Semantic Search for Network Documentation

## The Problem Every Network Team Faces

You have a Confluence wiki with 2,000 pages, a SharePoint with vendor guides, and a file share with years of post-mortem reports. A new engineer asks: "How do we handle failover when the primary ISP goes down?"

With keyword search, they'd need to guess the exact words someone used years ago. Was it "ISP failover," "WAN redundancy," "default route failover," or "backup link activation"? Traditional search treats each of these as completely different queries.

**Networking analogy**: Keyword search is like a static route — it only works if you know the exact destination. Semantic search is like a routing protocol — it understands the *intent* behind the query and finds the best path to the answer, even if the wording is different.

---

## Beyond Basic Similarity Search

In Chapter 16, we built a basic RAG system that retrieves documents by comparing embeddings. That works well for straightforward queries, but real-world network documentation search needs more sophistication.

### Why Simple Search Falls Short

Consider this scenario: an engineer types "BGP session keeps dropping to our cloud provider." A basic vector search might return documents about:
- BGP configuration basics (relevant concept, wrong context)
- Cloud provider marketing pages (right keywords, wrong content)
- An old change request that mentions BGP and cloud (partially relevant)

What it *should* return is the post-mortem from last year where the team discovered that the cloud provider's BGP hold timer was set to 30 seconds while your routers used 180 seconds, causing flaps during maintenance windows.

### The Search Quality Ladder

```
Level 1: Keyword search     → "Find docs containing these exact words"
Level 2: Semantic search     → "Find docs about this concept"
Level 3: Multi-query search  → "Find docs approaching this from multiple angles"
Level 4: Hybrid + re-ranking → "Find, combine, and prioritize the best results"
```

We're going to build Level 4.

---

## Multi-Query Retrieval

The first technique is to generate multiple search queries from a single user question. Think of it like troubleshooting — you don't just check one thing, you approach the problem from several angles.

**Networking analogy**: This is like ECMP (Equal-Cost Multi-Path) for search. Instead of sending all traffic down one path, you spread queries across multiple paths to maximize your chance of reaching the right answer.

```python
import os
import json
import re
from anthropic import Anthropic

client = Anthropic()  # Uses ANTHROPIC_API_KEY environment variable

def generate_search_queries(user_question: str, num_queries: int = 4) -> list:
    """Generate multiple search queries from a single user question.

    Like running 'show' commands from multiple vantage points —
    each query approaches the problem from a different angle.
    """

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        temperature=0.3,
        messages=[{
            "role": "user",
            "content": f"""Generate {num_queries} different search queries to find
information relevant to this question. Each query should approach the topic
from a different angle — different terminology, different abstraction level,
or different aspect of the problem.

Question: {user_question}

Return ONLY a JSON array of strings, no other text.
Example: ["query 1", "query 2", "query 3", "query 4"]"""
        }]
    )

    text = response.content[0].text.strip()
    # Robust JSON parsing — LLMs sometimes add code fences
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            return json.loads(json_match.group(0))
        return [user_question]  # Fallback to original query


# Example usage
queries = generate_search_queries(
    "How do we handle failover when the primary ISP goes down?"
)

# Typical output:
# [
#     "ISP failover procedure WAN redundancy",
#     "default route tracking primary link failure",
#     "BGP prefix failover backup ISP activation",
#     "WAN circuit failure automated recovery process"
# ]
```

Each of these queries will catch different documents that the others might miss. You then run all queries against your vector database and combine the results.

---

## Hybrid Search: Keyword + Semantic

Semantic search understands meaning, but it can miss exact identifiers. If someone searches for "issue with AS65001 peering," a semantic search understands the *concept* of BGP peering issues but might not prioritize documents containing the exact string "AS65001."

Keyword search excels at finding exact terms — IP addresses, AS numbers, hostnames, interface names — while semantic search excels at understanding intent.

**Networking analogy**: Hybrid search is like running both OSPF and static routes. OSPF (semantic) dynamically finds the best path based on network topology and conditions. Static routes (keyword) guarantee you reach specific, known destinations. Together, they cover all cases.

```python
from typing import List, Dict
import numpy as np

class HybridNetworkSearch:
    """Combines keyword and semantic search for network documentation.

    Keyword search catches exact identifiers (IPs, hostnames, AS numbers).
    Semantic search catches conceptual matches (intent, related topics).
    """

    def __init__(self, documents: List[Dict]):
        """
        Args:
            documents: List of dicts with 'content', 'title', and 'embedding' keys
        """
        self.documents = documents

    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Find documents containing exact query terms.

        Especially important for networking — you need exact matches for:
        - IP addresses (10.0.1.1)
        - AS numbers (AS65001)
        - Hostnames (router-core-01)
        - Interface names (GigabitEthernet0/0/1)
        """
        query_terms = query.lower().split()
        scored = []

        for doc in self.documents:
            content_lower = doc['content'].lower()
            # Count how many query terms appear in the document
            matches = sum(1 for term in query_terms if term in content_lower)
            if matches > 0:
                scored.append({
                    **doc,
                    'keyword_score': matches / len(query_terms)
                })

        scored.sort(key=lambda x: x['keyword_score'], reverse=True)
        return scored[:top_k]

    def semantic_search(
        self, query_embedding: List[float], top_k: int = 10
    ) -> List[Dict]:
        """Find documents by meaning similarity.

        Uses cosine similarity between the query embedding and
        each document's pre-computed embedding.
        """
        query_vec = np.array(query_embedding)
        scored = []

        for doc in self.documents:
            doc_vec = np.array(doc['embedding'])
            # Cosine similarity
            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )
            scored.append({**doc, 'semantic_score': float(similarity)})

        scored.sort(key=lambda x: x['semantic_score'], reverse=True)
        return scored[:top_k]

    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[Dict]:
        """Combine keyword and semantic search results.

        Args:
            keyword_weight: Weight for keyword matches (higher = favor exact terms)
            semantic_weight: Weight for semantic matches (higher = favor meaning)

        For networking docs, you might increase keyword_weight when queries
        contain specific identifiers like IP addresses or hostnames.
        """
        # Detect if query contains network identifiers — if so, boost keyword weight
        import re
        has_identifiers = bool(re.search(
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|'  # IP address
            r'\bAS\d+\b|'                                    # AS number
            r'\b[Gg]ig\w*\d|'                                # Interface name
            r'\b[Tt]en\w*\d',                                # 10G interface
            query
        ))

        if has_identifiers:
            keyword_weight = 0.5
            semantic_weight = 0.5

        keyword_results = self.keyword_search(query, top_k=20)
        semantic_results = self.semantic_search(query_embedding, top_k=20)

        # Merge results using Reciprocal Rank Fusion (RRF)
        doc_scores = {}

        for rank, doc in enumerate(keyword_results):
            doc_id = doc.get('title', doc['content'][:50])
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + \
                keyword_weight * (1 / (rank + 60))

        for rank, doc in enumerate(semantic_results):
            doc_id = doc.get('title', doc['content'][:50])
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + \
                semantic_weight * (1 / (rank + 60))

        # Sort by combined score and return top_k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]
```

### Why Reciprocal Rank Fusion (RRF)?

RRF is a simple but effective way to merge ranked lists. Instead of normalizing raw scores (which is tricky because keyword scores and embedding distances are on different scales), RRF uses *rank position*. A document ranked #1 in both lists gets a higher combined score than one ranked #1 in one and #20 in another.

The formula `1 / (rank + k)` (where k=60 is standard) prevents any single high rank from dominating the combined score.

---

## Contextual Compression

When a search returns a 500-line configuration document, the user doesn't need all 500 lines — they need the 20 lines that answer their question. Contextual compression extracts just the relevant portions.

**Networking analogy**: This is like the difference between a full `show running-config` (everything) and `show running-config | section bgp` (just what you need). We're teaching the AI to automatically apply the right filter.

```python
def compress_search_results(
    query: str,
    documents: List[str],
    max_snippets: int = 3
) -> List[str]:
    """Extract only the relevant portions from retrieved documents.

    Instead of returning full documents, this pulls out just the
    sections that answer the user's question — like piping show
    commands through 'include' or 'section' filters.
    """

    compressed = []

    for doc in documents[:5]:  # Process top 5 results
        response = client.messages.create(
            model="claude-haiku-4-20250514",  # Haiku is fast + cheap for extraction
            max_tokens=500,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Extract ONLY the parts of this document that are
directly relevant to the question. If nothing is relevant, respond with "NOT_RELEVANT".

Question: {query}

Document:
{doc[:3000]}

Return only the relevant excerpt(s), preserving the original text exactly."""
            }]
        )

        result = response.content[0].text.strip()
        if result != "NOT_RELEVANT" and len(result) > 20:
            compressed.append(result)

    return compressed[:max_snippets]
```

---

## Re-Ranking Results

Initial retrieval casts a wide net — it's fast but approximate. Re-ranking is a second pass that carefully evaluates each candidate document against the original question.

**Networking analogy**: Think of initial retrieval as the control plane — it builds the routing table quickly. Re-ranking is the data plane verification — it checks each forwarding entry to ensure traffic actually reaches the right destination.

```python
def rerank_results(
    query: str,
    documents: List[Dict],
    top_k: int = 3
) -> List[Dict]:
    """Re-rank search results using Claude for relevance scoring.

    Initial vector search is fast but approximate (like a hash lookup).
    Re-ranking is slower but more accurate (like a deep packet inspection).
    Use it on your top ~20 candidates, not the whole corpus.
    """

    doc_descriptions = "\n".join([
        f"[Doc {i+1}] {doc['title']}: {doc['content'][:300]}..."
        for i, doc in enumerate(documents[:15])
    ])

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Rank these documents by relevance to the question.
Return ONLY a JSON array of document numbers in order of relevance.

Question: {query}

Documents:
{doc_descriptions}

Example response: [3, 1, 7, 2]"""
        }]
    )

    text = response.content[0].text.strip()
    try:
        rankings = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[[\d,\s]+\]', text)
        rankings = json.loads(match.group(0)) if match else list(range(1, len(documents)+1))

    # Reorder documents by ranking
    reranked = []
    for rank in rankings[:top_k]:
        idx = rank - 1  # Convert to 0-indexed
        if 0 <= idx < len(documents):
            reranked.append(documents[idx])

    return reranked
```

---

## Query Understanding and Classification

Not all queries are equal. "How do I configure OSPF?" is a how-to request. "Why is BGP session down?" is troubleshooting. "What is our MPLS architecture?" is informational. Knowing the query type helps you search the right way.

**Networking analogy**: This is like QoS classification at the network edge. You mark traffic (queries) with the right DSCP value (intent type) so downstream systems (search and retrieval) handle it appropriately.

```python
def classify_and_expand_query(query: str) -> Dict:
    """Classify query intent and expand with related terms.

    Like QoS marking — classify the query type first, then apply
    the right treatment (search strategy) based on the classification.
    """

    response = client.messages.create(
        model="claude-haiku-4-20250514",
        max_tokens=400,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Analyze this network engineering question and return JSON:

Question: {query}

Return ONLY valid JSON:
{{
    "intent": "how-to | troubleshooting | informational | reference",
    "key_concepts": ["list", "of", "key", "technical", "concepts"],
    "expanded_terms": ["synonyms", "and", "related", "networking", "terms"],
    "suggested_queries": ["2-3 alternative phrasings of this question"]
}}"""
        }]
    )

    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        return {"intent": "informational", "key_concepts": [],
                "expanded_terms": [], "suggested_queries": [query]}


# Example
result = classify_and_expand_query("BGP session keeps flapping to our Azure peer")

# Expected output:
# {
#     "intent": "troubleshooting",
#     "key_concepts": ["BGP", "session flapping", "Azure", "cloud peering"],
#     "expanded_terms": ["BGP hold timer", "keepalive", "ExpressRoute",
#                        "peering stability", "route oscillation"],
#     "suggested_queries": [
#         "BGP session instability cloud provider peering",
#         "Azure ExpressRoute BGP flapping troubleshooting",
#         "BGP hold timer mismatch causing session drops"
#     ]
# }
```

---

## Putting It All Together: A Complete Search Pipeline

Here's how all the pieces fit together in a production search system:

```python
class NetworkDocumentSearch:
    """Complete semantic search pipeline for network documentation.

    Combines multi-query, hybrid search, re-ranking, and compression
    into a single search experience.

    Pipeline flow (like a packet through the network):
    1. Query classification  → QoS marking
    2. Query expansion       → ECMP path generation
    3. Hybrid search         → Forwarding lookup (keyword + semantic)
    4. Re-ranking            → Policy-based routing refinement
    5. Compression           → Output filtering (show | section)
    """

    def __init__(self, documents: List[Dict]):
        self.hybrid_searcher = HybridNetworkSearch(documents)
        self.documents = documents

    def search(self, query: str, query_embedding: List[float]) -> Dict:
        """Execute the full search pipeline."""

        # Step 1: Understand the query
        classification = classify_and_expand_query(query)

        # Step 2: Generate multiple search queries
        queries = generate_search_queries(query, num_queries=3)
        queries.extend(classification.get('suggested_queries', []))

        # Step 3: Run hybrid search for each query variation
        all_results = []
        for q in queries:
            results = self.hybrid_searcher.hybrid_search(
                query=q,
                query_embedding=query_embedding,  # In production, embed each query separately
                top_k=10
            )
            all_results.extend(results)

        # Step 4: Deduplicate (same doc found by multiple queries)
        seen = set()
        unique_results = []
        for doc_id, score in all_results:
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append({'title': doc_id, 'score': score})

        # Step 5: Re-rank the combined results
        # (In production, you'd look up full doc content here)
        top_results = unique_results[:15]

        # Step 6: Compress — extract just the relevant parts
        doc_contents = [r['title'] for r in top_results[:5]]
        compressed = compress_search_results(query, doc_contents)

        return {
            "query": query,
            "intent": classification['intent'],
            "num_queries_generated": len(queries),
            "total_candidates": len(unique_results),
            "top_results": top_results[:5],
            "relevant_excerpts": compressed
        }
```

---

## Cost and Performance Considerations

| Component | Model Used | API Calls | Approx. Cost per Query |
|-----------|-----------|-----------|----------------------|
| Query classification | Claude Haiku | 1 | ~$0.001 |
| Multi-query generation | Claude Sonnet | 1 | ~$0.003 |
| Contextual compression | Claude Haiku | 3-5 | ~$0.003-0.005 |
| Re-ranking | Claude Sonnet | 1 | ~$0.005 |
| **Total per search** | | **6-8** | **~$0.01-0.015** |

For a team of 20 engineers doing 50 searches/day: ~$7.50-15/month. Compare that to the engineering time saved by not digging through stale wikis.

**Optimization tips**:
- Cache embeddings — don't re-embed unchanged documents
- Cache frequent queries — same question from different engineers
- Use Haiku for fast operations (classification, compression), Sonnet for quality operations (re-ranking, generation)
- Batch searches when possible

---

## Key Takeaways

1. **Multi-query retrieval** (ECMP for search): Generate multiple query variations to catch documents that different phrasings would miss.

2. **Hybrid search** (OSPF + static routes): Combine semantic understanding with keyword precision — critical for networking docs full of exact identifiers.

3. **Re-ranking** (policy-based routing): A second pass that carefully evaluates candidate quality, not just initial retrieval score.

4. **Contextual compression** (show | section): Extract just the relevant portions from large documents instead of returning everything.

5. **Query classification** (QoS marking): Understand the user's intent to apply the right search strategy.

---

## What's Next

In **Chapter 18**, we'll take these search patterns into production — handling document updates, caching strategies, monitoring search quality, and scaling to enterprise-sized documentation libraries.
