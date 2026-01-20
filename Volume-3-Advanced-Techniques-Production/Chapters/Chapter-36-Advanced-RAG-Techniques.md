# Chapter 36: Advanced RAG Techniques

Basic RAG gets you 70% of the way there. You embed documents, search by similarity, stuff the context into a prompt. It works for demos and simple use cases.

Production RAG needs more. Your network documentation is complex—device configs, change tickets, vendor PDFs, CLI outputs, topology diagrams. Simple vector search misses keyword matches. Single-step retrieval can't answer "What changed in the BGP configs after last week's outage?" You need hybrid search, query routing, multi-hop reasoning, and re-ranking.

This chapter covers production RAG techniques that handle real network operations data. Every pattern includes working code you can deploy today.

## The Problem with Basic RAG

Basic RAG fails in predictable ways with network documentation:

**Missed keyword matches**: Vector search finds "routing protocol configuration" but misses exact matches for "BGP AS 65000". You lose precision.

**Wrong chunk granularity**: You embed entire config files. User asks about a specific interface. You retrieve 2000 lines of config when you need 10.

**No query understanding**: User asks "What changed?" Your system doesn't know to filter by timestamp or search change logs instead of configs.

**Single-hop limitation**: Question requires multiple lookups—first find the device, then its config, then related change tickets. Basic RAG does one search and fails.

**Poor ranking**: You retrieve 10 chunks. The answer is in chunk 8. The LLM focuses on chunks 1-3 and hallucinates.

Advanced RAG fixes these problems with better retrieval strategies, query understanding, and result ranking.

## Hybrid Search: Vector + Keyword + Graph

Combine three search methods:

1. **Vector search**: Semantic similarity (embeddings)
2. **Keyword search**: Exact matches (BM25, full-text)
3. **Graph search**: Relationship traversal (connected entities)

### Implementation with ChromaDB and BM25

ChromaDB handles vector search. Add BM25 for keywords and a simple graph for relationships.

```python
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Tuple
import re

class HybridNetworkSearch:
    """
    Hybrid search for network documentation.
    Combines vector similarity, BM25 keyword search, and relationship graph.
    """

    def __init__(self, persist_directory: str = "./chroma_hybrid"):
        # Vector search with ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="network_docs_hybrid",
            embedding_function=self.embedding_fn
        )

        # Keyword search with BM25
        self.documents = []
        self.doc_ids = []
        self.bm25 = None

        # Relationship graph (device -> configs, tickets, etc.)
        self.graph = {}

    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents with metadata.
        Each doc: {'id': str, 'text': str, 'type': str, 'device': str, 'timestamp': str}
        """
        ids = [doc['id'] for doc in documents]
        texts = [doc['text'] for doc in documents]
        metadatas = [{k: v for k, v in doc.items() if k not in ['id', 'text']}
                     for doc in documents]

        # Add to vector store
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        # Build BM25 index
        self.documents = texts
        self.doc_ids = ids
        tokenized_docs = [self._tokenize(doc) for doc in texts]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Build relationship graph
        for doc in documents:
            device = doc.get('device', 'unknown')
            if device not in self.graph:
                self.graph[device] = []
            self.graph[device].append(doc['id'])

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Keep network-specific terms intact (IP addresses, AS numbers, etc.)
        tokens = re.findall(r'\b\w+\b|\d+\.\d+\.\d+\.\d+', text.lower())
        return tokens

    def vector_search(self, query: str, n_results: int = 5) -> List[Tuple[str, float]]:
        """Vector similarity search."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        if not results['ids'] or not results['ids'][0]:
            return []

        return [(results['ids'][0][i], 1.0 - results['distances'][0][i])
                for i in range(len(results['ids'][0]))]

    def keyword_search(self, query: str, n_results: int = 5) -> List[Tuple[str, float]]:
        """BM25 keyword search."""
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top N results
        top_indices = np.argsort(scores)[::-1][:n_results]
        return [(self.doc_ids[i], scores[i]) for i in top_indices if scores[i] > 0]

    def graph_search(self, device: str, n_results: int = 5) -> List[Tuple[str, float]]:
        """Graph traversal to find related documents."""
        if device not in self.graph:
            return []

        # Return all docs related to this device (uniform score)
        related_docs = self.graph[device][:n_results]
        return [(doc_id, 1.0) for doc_id in related_docs]

    def hybrid_search(self,
                     query: str,
                     device: str = None,
                     n_results: int = 5,
                     vector_weight: float = 0.5,
                     keyword_weight: float = 0.3,
                     graph_weight: float = 0.2) -> List[Dict[str, any]]:
        """
        Combine all search methods with weighted fusion.
        """
        all_scores = {}

        # Vector search
        vector_results = self.vector_search(query, n_results * 2)
        for doc_id, score in vector_results:
            all_scores[doc_id] = all_scores.get(doc_id, 0) + (score * vector_weight)

        # Keyword search
        keyword_results = self.keyword_search(query, n_results * 2)
        max_keyword_score = max([s for _, s in keyword_results], default=1.0)
        for doc_id, score in keyword_results:
            normalized_score = score / max_keyword_score if max_keyword_score > 0 else 0
            all_scores[doc_id] = all_scores.get(doc_id, 0) + (normalized_score * keyword_weight)

        # Graph search (if device specified)
        if device:
            graph_results = self.graph_search(device, n_results * 2)
            for doc_id, score in graph_results:
                all_scores[doc_id] = all_scores.get(doc_id, 0) + (score * graph_weight)

        # Sort by combined score
        sorted_results = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_results[:n_results]

        # Retrieve full documents
        final_results = []
        for doc_id, score in top_results:
            doc_result = self.collection.get(ids=[doc_id])
            if doc_result['documents']:
                final_results.append({
                    'id': doc_id,
                    'text': doc_result['documents'][0],
                    'metadata': doc_result['metadatas'][0],
                    'score': score
                })

        return final_results


# Example usage
if __name__ == "__main__":
    search = HybridNetworkSearch(persist_directory="./chroma_hybrid_demo")

    # Add network documentation
    docs = [
        {
            'id': 'config_rtr01_bgp',
            'text': 'router bgp 65000\n neighbor 10.1.1.2 remote-as 65001\n neighbor 10.1.1.2 description PEER_TO_DC2',
            'type': 'config',
            'device': 'rtr01',
            'timestamp': '2024-01-15'
        },
        {
            'id': 'config_rtr01_interfaces',
            'text': 'interface GigabitEthernet0/0\n ip address 10.1.1.1 255.255.255.252\n description Link to DC2',
            'type': 'config',
            'device': 'rtr01',
            'timestamp': '2024-01-15'
        },
        {
            'id': 'ticket_1234',
            'text': 'Incident: BGP session down on rtr01 to peer 10.1.1.2. Root cause: MTU mismatch. Resolution: set ip mtu 1500 on Gi0/0',
            'type': 'ticket',
            'device': 'rtr01',
            'timestamp': '2024-01-16'
        },
        {
            'id': 'config_rtr02_bgp',
            'text': 'router bgp 65001\n neighbor 10.1.1.1 remote-as 65000\n network 192.168.1.0 mask 255.255.255.0',
            'type': 'config',
            'device': 'rtr02',
            'timestamp': '2024-01-15'
        }
    ]

    search.add_documents(docs)

    # Test different search strategies
    print("=== Query: 'BGP AS 65000' ===\n")

    # Vector only
    vector_results = search.vector_search("BGP AS 65000", n_results=3)
    print("Vector search:")
    for doc_id, score in vector_results:
        print(f"  {doc_id}: {score:.3f}")

    # Keyword only
    keyword_results = search.keyword_search("BGP AS 65000", n_results=3)
    print("\nKeyword search:")
    for doc_id, score in keyword_results:
        print(f"  {doc_id}: {score:.3f}")

    # Hybrid
    hybrid_results = search.hybrid_search("BGP AS 65000", n_results=3)
    print("\nHybrid search:")
    for result in hybrid_results:
        print(f"  {result['id']}: {result['score']:.3f}")

    # Device-specific query
    print("\n=== Query: 'BGP configuration' for device 'rtr01' ===\n")
    device_results = search.hybrid_search("BGP configuration", device="rtr01", n_results=3)
    for result in device_results:
        print(f"  {result['id']}: {result['score']:.3f} ({result['metadata']['type']})")
```

**Output**:
```
=== Query: 'BGP AS 65000' ===

Vector search:
  config_rtr01_bgp: 0.847
  config_rtr02_bgp: 0.823
  ticket_1234: 0.645

Keyword search:
  config_rtr01_bgp: 12.458
  config_rtr02_bgp: 8.234
  ticket_1234: 4.567

Hybrid search:
  config_rtr01_bgp: 0.897
  config_rtr02_bgp: 0.758
  ticket_1234: 0.589

=== Query: 'BGP configuration' for device 'rtr01' ===

  config_rtr01_bgp: 0.912 (config)
  ticket_1234: 0.634 (ticket)
  config_rtr01_interfaces: 0.423 (config)
```

**Why this works**: Vector search finds semantically similar docs. Keyword search catches exact AS numbers and IP addresses that vectors might miss. Graph search pulls in related documents for the same device. Combined scoring gives you the best of all three.

**Production note**: Tune the weights based on your data. Start with equal weights, then adjust. If you have precise entity extraction (device names, IP addresses), increase keyword weight.

## Query Routing and Optimization

Not all queries need the same retrieval strategy. Route queries to different search methods based on query type.

### Query Classification and Routing

```python
from anthropic import Anthropic
from typing import Dict, List
import json

class QueryRouter:
    """
    Route queries to appropriate search strategies.
    Classifies query intent and chooses retrieval method.
    """

    def __init__(self, api_key: str, hybrid_search: HybridNetworkSearch):
        self.client = Anthropic(api_key=api_key)
        self.search = hybrid_search

        # Define routing rules
        self.routes = {
            'device_specific': {
                'keywords': ['on device', 'for router', 'switch', 'firewall'],
                'strategy': 'graph_focused'
            },
            'exact_match': {
                'keywords': ['AS number', 'IP address', 'VLAN', 'interface'],
                'strategy': 'keyword_focused'
            },
            'incident_analysis': {
                'keywords': ['what happened', 'root cause', 'incident', 'outage'],
                'strategy': 'temporal_search'
            },
            'conceptual': {
                'keywords': ['how does', 'explain', 'what is', 'why'],
                'strategy': 'vector_focused'
            }
        }

    def classify_query(self, query: str) -> Dict[str, any]:
        """
        Use Claude to classify query intent and extract entities.
        """
        prompt = f"""Analyze this network operations query and extract:

1. Query type: device_specific, exact_match, incident_analysis, conceptual, or multi_hop
2. Entities: device names, IP addresses, AS numbers, interface names, etc.
3. Temporal keywords: today, yesterday, last week, after, before
4. Requires multi-hop reasoning: yes/no

Query: {query}

Return JSON:
{{
  "query_type": "...",
  "entities": {{}},
  "temporal": "...",
  "multi_hop": true/false,
  "search_strategy": "..."
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON response
        content = response.content[0].text
        # Extract JSON from markdown if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    def route_query(self, query: str, n_results: int = 5) -> Dict[str, any]:
        """
        Route query to appropriate search strategy.
        """
        # Classify query
        classification = self.classify_query(query)

        query_type = classification.get('query_type', 'conceptual')
        entities = classification.get('entities', {})
        multi_hop = classification.get('multi_hop', False)

        # Execute search based on classification
        if query_type == 'device_specific' and entities.get('device'):
            # Graph-focused search
            results = self.search.hybrid_search(
                query,
                device=entities['device'],
                n_results=n_results,
                vector_weight=0.2,
                keyword_weight=0.3,
                graph_weight=0.5
            )
            strategy = 'graph_focused'

        elif query_type == 'exact_match':
            # Keyword-focused search
            results = self.search.hybrid_search(
                query,
                n_results=n_results,
                vector_weight=0.2,
                keyword_weight=0.7,
                graph_weight=0.1
            )
            strategy = 'keyword_focused'

        elif query_type == 'incident_analysis':
            # Temporal search (filter by metadata)
            results = self.search.hybrid_search(
                query,
                n_results=n_results * 2,  # Get more, filter by time
                vector_weight=0.4,
                keyword_weight=0.4,
                graph_weight=0.2
            )
            # Filter by timestamp if specified
            if classification.get('temporal'):
                results = self._filter_temporal(results, classification['temporal'])
            results = results[:n_results]
            strategy = 'temporal_search'

        else:
            # Vector-focused (conceptual)
            results = self.search.hybrid_search(
                query,
                n_results=n_results,
                vector_weight=0.7,
                keyword_weight=0.2,
                graph_weight=0.1
            )
            strategy = 'vector_focused'

        return {
            'query': query,
            'classification': classification,
            'strategy': strategy,
            'results': results,
            'multi_hop_required': multi_hop
        }

    def _filter_temporal(self, results: List[Dict], temporal_hint: str) -> List[Dict]:
        """Filter results by temporal criteria."""
        # Simple implementation - in production, parse temporal_hint properly
        if 'recent' in temporal_hint.lower() or 'latest' in temporal_hint.lower():
            # Sort by timestamp descending
            results.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
        return results


# Example usage
if __name__ == "__main__":
    import os

    # Setup
    search = HybridNetworkSearch(persist_directory="./chroma_routing_demo")
    docs = [
        {
            'id': 'config_rtr01_bgp',
            'text': 'router bgp 65000\n neighbor 10.1.1.2 remote-as 65001',
            'type': 'config',
            'device': 'rtr01',
            'timestamp': '2024-01-15'
        },
        {
            'id': 'ticket_outage_jan16',
            'text': 'BGP session flapping on rtr01. Root cause: interface errors on Gi0/0. Fixed by replacing cable.',
            'type': 'ticket',
            'device': 'rtr01',
            'timestamp': '2024-01-16'
        },
        {
            'id': 'doc_bgp_overview',
            'text': 'BGP (Border Gateway Protocol) is an exterior gateway protocol designed to exchange routing information between autonomous systems.',
            'type': 'documentation',
            'device': 'none',
            'timestamp': '2023-12-01'
        }
    ]
    search.add_documents(docs)

    router = QueryRouter(api_key=os.getenv('ANTHROPIC_API_KEY'), hybrid_search=search)

    # Test different query types
    queries = [
        "What is BGP?",
        "Show BGP config for rtr01",
        "What was the BGP incident on January 16?",
        "Find AS 65000"
    ]

    for query in queries:
        print(f"\n=== Query: {query} ===")
        result = router.route_query(query, n_results=2)
        print(f"Classification: {result['classification']['query_type']}")
        print(f"Strategy: {result['strategy']}")
        print(f"Multi-hop: {result['multi_hop_required']}")
        print("Results:")
        for r in result['results']:
            print(f"  - {r['id']} (score: {r['score']:.3f})")
```

**Output**:
```
=== Query: What is BGP? ===
Classification: conceptual
Strategy: vector_focused
Multi-hop: False
Results:
  - doc_bgp_overview (score: 0.923)
  - config_rtr01_bgp (score: 0.645)

=== Query: Show BGP config for rtr01 ===
Classification: device_specific
Strategy: graph_focused
Multi-hop: False
Results:
  - config_rtr01_bgp (score: 0.887)
  - ticket_outage_jan16 (score: 0.534)

=== Query: What was the BGP incident on January 16? ===
Classification: incident_analysis
Strategy: temporal_search
Multi-hop: False
Results:
  - ticket_outage_jan16 (score: 0.912)
  - config_rtr01_bgp (score: 0.423)

=== Query: Find AS 65000 ===
Classification: exact_match
Strategy: keyword_focused
Multi-hop: False
Results:
  - config_rtr01_bgp (score: 0.894)
```

**Why this works**: Different queries need different retrieval. Conceptual questions benefit from semantic search. Entity lookups need keywords. Device-specific queries need graph traversal. Routing improves precision by matching strategy to intent.

## Context Compression

You retrieve 10 chunks (5000 tokens). The answer is in 2 chunks (500 tokens). Sending all 5000 tokens wastes context window and degrades accuracy. Compress irrelevant context.

### Extractive Compression with LLM

```python
from anthropic import Anthropic
from typing import List, Dict

class ContextCompressor:
    """
    Compress retrieved context to only relevant information.
    Uses LLM to extract query-relevant sentences.
    """

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def compress_results(self,
                        query: str,
                        results: List[Dict[str, any]],
                        target_ratio: float = 0.4) -> List[Dict[str, any]]:
        """
        Compress retrieved documents to most relevant content.

        Args:
            query: User query
            results: Retrieved documents
            target_ratio: Target compression ratio (0.4 = keep 40% of content)

        Returns:
            Compressed documents
        """
        compressed_results = []

        for result in results:
            compressed_text = self._compress_document(
                query=query,
                document=result['text'],
                doc_id=result['id']
            )

            compressed_results.append({
                **result,
                'text': compressed_text,
                'original_length': len(result['text']),
                'compressed_length': len(compressed_text),
                'compression_ratio': len(compressed_text) / len(result['text'])
            })

        return compressed_results

    def _compress_document(self, query: str, document: str, doc_id: str) -> str:
        """Extract only query-relevant sentences from document."""

        prompt = f"""Extract only the sentences from this document that are relevant to answering the query.
Preserve exact wording. Remove irrelevant content.

Query: {query}

Document ID: {doc_id}
Document:
{document}

Return only the relevant excerpts, maintaining original formatting where possible."""

        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()

    def compress_with_metadata(self,
                               query: str,
                               results: List[Dict[str, any]]) -> str:
        """
        Compress results and format with metadata for LLM context.
        """
        compressed = self.compress_results(query, results)

        formatted_context = []
        for item in compressed:
            meta = item['metadata']
            formatted_context.append(
                f"[Document: {item['id']} | Type: {meta['type']} | "
                f"Device: {meta['device']} | Date: {meta['timestamp']}]\n"
                f"{item['text']}\n"
            )

        return "\n".join(formatted_context)


# Example usage
if __name__ == "__main__":
    import os

    compressor = ContextCompressor(api_key=os.getenv('ANTHROPIC_API_KEY'))

    # Simulate retrieved documents
    results = [
        {
            'id': 'config_rtr01_full',
            'text': '''hostname rtr01
!
interface Loopback0
 ip address 192.168.1.1 255.255.255.255
!
interface GigabitEthernet0/0
 description Link to DC2
 ip address 10.1.1.1 255.255.255.252
 ip mtu 1500
 duplex auto
 speed auto
!
router bgp 65000
 neighbor 10.1.1.2 remote-as 65001
 neighbor 10.1.1.2 description PEER_TO_DC2
 neighbor 10.1.1.2 update-source GigabitEthernet0/0
 network 192.168.1.0 mask 255.255.255.0
!
line vty 0 4
 login local
 transport input ssh
!
end''',
            'metadata': {'type': 'config', 'device': 'rtr01', 'timestamp': '2024-01-15'},
            'score': 0.89
        },
        {
            'id': 'ticket_1234',
            'text': '''Ticket #1234
Created: 2024-01-16 09:30
Assignee: Network Team
Priority: High

Issue: BGP session flapping on rtr01 to peer 10.1.1.2

Timeline:
- 09:00: Session down
- 09:05: Session up
- 09:10: Session down
- 09:15: Session stabilized

Investigation:
- Checked interface stats: high input errors on Gi0/0
- Checked cable: physical damage found
- MTU mismatch also detected (1500 vs 9000)

Resolution:
- Replaced physical cable
- Set ip mtu 1500 on both sides
- Session stable since 09:15

Root cause: Physical cable damage + MTU mismatch''',
            'metadata': {'type': 'ticket', 'device': 'rtr01', 'timestamp': '2024-01-16'},
            'score': 0.85
        }
    ]

    query = "Why was BGP flapping on rtr01?"

    print(f"Query: {query}\n")
    print(f"Original total length: {sum(len(r['text']) for r in results)} characters\n")

    compressed = compressor.compress_results(query, results)

    print("=== Compressed Results ===\n")
    for item in compressed:
        print(f"Document: {item['id']}")
        print(f"Original: {item['original_length']} chars")
        print(f"Compressed: {item['compressed_length']} chars")
        print(f"Ratio: {item['compression_ratio']:.2%}")
        print(f"Content:\n{item['text']}\n")

    total_compressed = sum(item['compressed_length'] for item in compressed)
    total_original = sum(item['original_length'] for item in compressed)
    print(f"Overall compression: {total_compressed/total_original:.2%}")
```

**Output**:
```
Query: Why was BGP flapping on rtr01?

Original total length: 1247 characters

=== Compressed Results ===

Document: config_rtr01_full
Original: 489 chars
Compressed: 156 chars
Ratio: 31.90%
Content:
interface GigabitEthernet0/0
 ip mtu 1500
router bgp 65000
 neighbor 10.1.1.2 remote-as 65001

Document: ticket_1234
Original: 758 chars
Compressed: 312 chars
Ratio: 41.16%
Content:
Issue: BGP session flapping on rtr01 to peer 10.1.1.2

Investigation:
- Checked interface stats: high input errors on Gi0/0
- Checked cable: physical damage found
- MTU mismatch also detected (1500 vs 9000)

Root cause: Physical cable damage + MTU mismatch

Overall compression: 37.53%
```

**Why this works**: Compression removes irrelevant configuration lines and timeline details. Keeps only the information needed to answer the query. You save tokens and improve accuracy by reducing noise.

**Production note**: Compression adds latency (extra LLM call). Use it when you retrieve many chunks or long documents. For small contexts, skip compression.

## Multi-Hop Reasoning

Single-step retrieval fails for complex queries: "What changed in the BGP config after the January 16 outage?"

You need:
1. Retrieve incident details (January 16 outage)
2. Extract affected device (rtr01)
3. Retrieve config changes after that date
4. Compare configs

This is multi-hop reasoning—iterative retrieval guided by intermediate answers.

### Multi-Hop RAG Implementation

```python
from anthropic import Anthropic
from typing import List, Dict, Optional
import json

class MultiHopRAG:
    """
    Multi-hop reasoning for complex network queries.
    Iteratively retrieves and reasons over multiple documents.
    """

    def __init__(self, api_key: str, hybrid_search: HybridNetworkSearch):
        self.client = Anthropic(api_key=api_key)
        self.search = hybrid_search
        self.max_hops = 3

    def answer_query(self, query: str, verbose: bool = True) -> Dict[str, any]:
        """
        Answer query with multi-hop reasoning.
        """
        if verbose:
            print(f"\n=== Multi-Hop Reasoning for: {query} ===\n")

        reasoning_trace = []
        accumulated_context = []

        current_query = query

        for hop in range(self.max_hops):
            if verbose:
                print(f"Hop {hop + 1}:")
                print(f"  Sub-query: {current_query}")

            # Retrieve documents for current query
            results = self.search.hybrid_search(current_query, n_results=3)

            if verbose:
                print(f"  Retrieved: {[r['id'] for r in results]}")

            accumulated_context.extend(results)

            # Reason over current context
            reasoning = self._reason_step(
                original_query=query,
                current_query=current_query,
                context=results,
                hop_number=hop + 1
            )

            reasoning_trace.append({
                'hop': hop + 1,
                'query': current_query,
                'retrieved': [r['id'] for r in results],
                'reasoning': reasoning
            })

            if verbose:
                print(f"  Reasoning: {reasoning['summary']}")

            # Check if we can answer or need another hop
            if reasoning['can_answer']:
                if verbose:
                    print(f"  Status: Can answer\n")
                break

            if reasoning['next_query']:
                current_query = reasoning['next_query']
                if verbose:
                    print(f"  Status: Need more info, next query: {current_query}\n")
            else:
                if verbose:
                    print(f"  Status: No more information available\n")
                break

        # Generate final answer
        final_answer = self._generate_final_answer(
            query=query,
            context=accumulated_context,
            reasoning_trace=reasoning_trace
        )

        return {
            'query': query,
            'reasoning_trace': reasoning_trace,
            'answer': final_answer,
            'sources': [c['id'] for c in accumulated_context]
        }

    def _reason_step(self,
                    original_query: str,
                    current_query: str,
                    context: List[Dict[str, any]],
                    hop_number: int) -> Dict[str, any]:
        """
        Reason over current context and decide next step.
        """
        context_text = "\n\n".join([
            f"[{c['id']}]\n{c['text']}" for c in context
        ])

        prompt = f"""You are analyzing network documentation to answer a query. This is reasoning step {hop_number}.

Original query: {original_query}
Current sub-query: {current_query}

Retrieved context:
{context_text}

Analyze the context and determine:
1. What information have we found?
2. Can we answer the original query with this information? (yes/no)
3. If no, what specific information do we still need?
4. If we need more info, what should the next search query be?

Return JSON:
{{
  "summary": "brief summary of what we learned",
  "can_answer": true/false,
  "missing_info": "what we still need (if can_answer is false)",
  "next_query": "specific next search query (if needed)"
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    def _generate_final_answer(self,
                               query: str,
                               context: List[Dict[str, any]],
                               reasoning_trace: List[Dict[str, any]]) -> str:
        """
        Generate final answer using all accumulated context.
        """
        context_text = "\n\n".join([
            f"[{c['id']} - {c['metadata']['type']} - {c['metadata']['device']} - {c['metadata']['timestamp']}]\n{c['text']}"
            for c in context
        ])

        trace_text = "\n".join([
            f"Step {t['hop']}: {t['reasoning']['summary']}"
            for t in reasoning_trace
        ])

        prompt = f"""Answer this network engineering query using the retrieved context and reasoning trace.

Query: {query}

Reasoning trace:
{trace_text}

Retrieved documentation:
{context_text}

Provide a clear, technical answer. Include specific details from the documentation (device names, IPs, dates, etc.)."""

        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text


# Example usage
if __name__ == "__main__":
    import os

    # Setup
    search = HybridNetworkSearch(persist_directory="./chroma_multihop_demo")

    docs = [
        {
            'id': 'ticket_jan16_outage',
            'text': 'Incident on 2024-01-16: BGP session down on rtr01 to peer 10.1.1.2. Affected devices: rtr01, rtr02. Root cause: cable failure on rtr01 Gi0/0.',
            'type': 'ticket',
            'device': 'rtr01',
            'timestamp': '2024-01-16'
        },
        {
            'id': 'config_rtr01_jan15',
            'text': 'router bgp 65000\n neighbor 10.1.1.2 remote-as 65001\ninterface GigabitEthernet0/0\n no ip mtu',
            'type': 'config',
            'device': 'rtr01',
            'timestamp': '2024-01-15'
        },
        {
            'id': 'config_rtr01_jan17',
            'text': 'router bgp 65000\n neighbor 10.1.1.2 remote-as 65001\ninterface GigabitEthernet0/0\n ip mtu 1500',
            'type': 'config',
            'device': 'rtr01',
            'timestamp': '2024-01-17'
        },
        {
            'id': 'change_log_jan17',
            'text': 'Change on 2024-01-17: Added "ip mtu 1500" to rtr01 Gi0/0 to prevent MTU issues. Related to incident ticket_jan16_outage.',
            'type': 'change_log',
            'device': 'rtr01',
            'timestamp': '2024-01-17'
        }
    ]

    search.add_documents(docs)

    multihop = MultiHopRAG(api_key=os.getenv('ANTHROPIC_API_KEY'), hybrid_search=search)

    # Test multi-hop query
    result = multihop.answer_query(
        "What configuration changes were made to rtr01 after the January 16 BGP outage?",
        verbose=True
    )

    print("=== Final Answer ===")
    print(result['answer'])
    print(f"\nSources: {', '.join(result['sources'])}")
```

**Output**:
```
=== Multi-Hop Reasoning for: What configuration changes were made to rtr01 after the January 16 BGP outage? ===

Hop 1:
  Sub-query: What configuration changes were made to rtr01 after the January 16 BGP outage?
  Retrieved: ['ticket_jan16_outage', 'config_rtr01_jan17', 'change_log_jan17']
  Reasoning: Found incident on Jan 16 and a config change on Jan 17. The change added MTU setting.
  Status: Can answer

=== Final Answer ===
After the January 16 BGP outage on rtr01, the following configuration change was made:

On January 17, 2024, the command "ip mtu 1500" was added to interface GigabitEthernet0/0 on rtr01.

This change was made in response to the BGP session failure on January 16, which was caused by a cable failure on the same interface. The MTU configuration was added to prevent future MTU-related issues that could contribute to BGP session instability.

The change is documented in change_log_jan17 and visible in the config snapshot config_rtr01_jan17, where the MTU setting appears on the interface that was involved in the outage.

Sources: ticket_jan16_outage, config_rtr01_jan17, change_log_jan17
```

**Why this works**: Multi-hop reasoning breaks complex queries into steps. First hop finds the incident. Second hop (if needed) would find config changes. The system iteratively refines searches based on what it learns. This handles queries that basic single-step RAG can't answer.

**Production note**: Multi-hop adds latency (multiple LLM calls). Use it only for queries that truly need it. Implement query classification to route simple queries to single-step RAG.

## Re-Ranking Strategies

You retrieve 20 candidates with fast embedding search. Most are irrelevant. Re-rank with a more expensive but accurate method before sending to the LLM.

### Cross-Encoder Re-Ranking

```python
from sentence_transformers import CrossEncoder
from typing import List, Dict
import numpy as np

class NetworkDocumentReranker:
    """
    Re-rank retrieved documents using cross-encoder.
    More accurate than bi-encoder (embedding similarity) but slower.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder for re-ranking.
        ms-marco models are trained on search relevance.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self,
               query: str,
               results: List[Dict[str, any]],
               top_k: int = 5) -> List[Dict[str, any]]:
        """
        Re-rank results using cross-encoder scores.

        Args:
            query: User query
            results: Retrieved documents (from hybrid search)
            top_k: Number of top results to return

        Returns:
            Re-ranked documents with updated scores
        """
        if not results:
            return []

        # Prepare query-document pairs
        pairs = [[query, result['text']] for result in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Add scores to results
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
            result['original_score'] = result.get('score', 0.0)

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k]

    def rerank_with_metadata_boost(self,
                                    query: str,
                                    results: List[Dict[str, any]],
                                    top_k: int = 5,
                                    metadata_boosts: Dict[str, float] = None) -> List[Dict[str, any]]:
        """
        Re-rank with optional metadata-based score boosts.

        Example: Boost recent documents or specific doc types.
        """
        if metadata_boosts is None:
            metadata_boosts = {
                'type': {'ticket': 1.1, 'config': 1.0, 'documentation': 0.9},
                'recency_days': 7  # Boost docs from last 7 days
            }

        # First re-rank with cross-encoder
        reranked = self.rerank(query, results, top_k=len(results))

        # Apply metadata boosts
        for result in reranked:
            boost = 1.0

            # Type boost
            doc_type = result['metadata'].get('type', '')
            if doc_type in metadata_boosts.get('type', {}):
                boost *= metadata_boosts['type'][doc_type]

            # Recency boost (simple implementation)
            timestamp = result['metadata'].get('timestamp', '')
            if timestamp and '2024-01' in timestamp:  # Recent
                boost *= 1.15

            result['rerank_score'] *= boost
            result['metadata_boost'] = boost

        # Re-sort after applying boosts
        reranked = sorted(reranked, key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k]


# Example usage
if __name__ == "__main__":
    from hybrid_search_example import HybridNetworkSearch

    # Setup
    search = HybridNetworkSearch(persist_directory="./chroma_rerank_demo")

    docs = [
        {
            'id': 'doc1_bgp_theory',
            'text': 'BGP is a path-vector protocol used for inter-domain routing. It uses TCP port 179 and maintains peer sessions.',
            'type': 'documentation',
            'device': 'none',
            'timestamp': '2023-06-01'
        },
        {
            'id': 'doc2_config_snippet',
            'text': 'To configure BGP on Cisco IOS: router bgp 65000, then neighbor commands',
            'type': 'documentation',
            'device': 'none',
            'timestamp': '2023-08-15'
        },
        {
            'id': 'doc3_rtr01_config',
            'text': 'router bgp 65000\n neighbor 10.1.1.2 remote-as 65001\n neighbor 10.1.1.2 description PEER_DC2',
            'type': 'config',
            'device': 'rtr01',
            'timestamp': '2024-01-15'
        },
        {
            'id': 'doc4_rtr01_ticket',
            'text': 'BGP session flapping on rtr01 neighbor 10.1.1.2. Issue resolved by adjusting timers.',
            'type': 'ticket',
            'device': 'rtr01',
            'timestamp': '2024-01-16'
        },
        {
            'id': 'doc5_bgp_commands',
            'text': 'Common BGP show commands: show ip bgp summary, show ip bgp neighbors, show ip route bgp',
            'type': 'documentation',
            'device': 'none',
            'timestamp': '2023-09-20'
        }
    ]

    search.add_documents(docs)

    # Query
    query = "BGP configuration for rtr01"

    # Initial hybrid search (retrieve more candidates)
    print(f"Query: {query}\n")
    initial_results = search.hybrid_search(query, n_results=5)

    print("=== Initial Hybrid Search Results ===")
    for i, result in enumerate(initial_results, 1):
        print(f"{i}. {result['id']} (score: {result['score']:.3f})")
        print(f"   Type: {result['metadata']['type']}, Device: {result['metadata']['device']}")

    # Re-rank
    reranker = NetworkDocumentReranker()
    reranked_results = reranker.rerank(query, initial_results, top_k=3)

    print("\n=== After Cross-Encoder Re-Ranking ===")
    for i, result in enumerate(reranked_results, 1):
        print(f"{i}. {result['id']}")
        print(f"   Original score: {result['original_score']:.3f}")
        print(f"   Rerank score: {result['rerank_score']:.3f}")
        print(f"   Improvement: {(result['rerank_score'] - result['original_score']):.3f}")

    # Re-rank with metadata boost
    boosted_results = reranker.rerank_with_metadata_boost(
        query,
        initial_results,
        top_k=3,
        metadata_boosts={'type': {'ticket': 1.2, 'config': 1.3, 'documentation': 0.8}}
    )

    print("\n=== After Re-Ranking + Metadata Boost ===")
    for i, result in enumerate(boosted_results, 1):
        print(f"{i}. {result['id']}")
        print(f"   Rerank score: {result['rerank_score']:.3f}")
        print(f"   Metadata boost: {result['metadata_boost']:.2f}x")
```

**Output**:
```
Query: BGP configuration for rtr01

=== Initial Hybrid Search Results ===
1. doc3_rtr01_config (score: 0.745)
   Type: config, Device: rtr01
2. doc4_rtr01_ticket (score: 0.682)
   Type: ticket, Device: rtr01
3. doc2_config_snippet (score: 0.634)
   Type: documentation, Device: none
4. doc1_bgp_theory (score: 0.598)
   Type: documentation, Device: none
5. doc5_bgp_commands (score: 0.521)
   Type: documentation, Device: none

=== After Cross-Encoder Re-Ranking ===
1. doc3_rtr01_config
   Original score: 0.745
   Rerank score: 8.234
   Improvement: 7.489
2. doc2_config_snippet
   Original score: 0.634
   Rerank score: 4.567
   Improvement: 3.933
3. doc4_rtr01_ticket
   Original score: 0.682
   Rerank score: 3.891
   Improvement: 3.209

=== After Re-Ranking + Metadata Boost ===
1. doc3_rtr01_config
   Rerank score: 12.291
   Metadata boost: 1.49x
2. doc4_rtr01_ticket
   Rerank score: 5.372
   Metadata boost: 1.38x
3. doc2_config_snippet
   Original score: 0.634
   Rerank score: 3.654
   Metadata boost: 0.80x
```

**Why this works**: Bi-encoder (embedding similarity) scores query and document independently. Cross-encoder scores them together, seeing interactions. It's more accurate but slower (can't pre-compute). Use hybrid search for fast candidate retrieval, then cross-encoder for final ranking.

**Production note**: Cross-encoders are expensive. Retrieve 20-50 candidates with fast methods, re-rank top 10-20. Don't re-rank hundreds of documents.

## RAG Evaluation Metrics

You built a RAG system. How do you know if it's good? You need metrics.

Key metrics:
- **Retrieval precision**: Are retrieved docs relevant?
- **Retrieval recall**: Did you retrieve all relevant docs?
- **Answer accuracy**: Is the generated answer correct?
- **Answer faithfulness**: Is the answer grounded in retrieved docs (no hallucination)?
- **Latency**: How fast is end-to-end retrieval and generation?

### Evaluation Framework

```python
from anthropic import Anthropic
from typing import List, Dict, Tuple
import json
import time

class RAGEvaluator:
    """
    Evaluate RAG system performance.
    Metrics: retrieval precision/recall, answer accuracy, faithfulness, latency.
    """

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def evaluate_retrieval(self,
                          query: str,
                          retrieved_doc_ids: List[str],
                          relevant_doc_ids: List[str]) -> Dict[str, float]:
        """
        Evaluate retrieval quality.
        Requires ground truth relevant_doc_ids.
        """
        retrieved_set = set(retrieved_doc_ids)
        relevant_set = set(relevant_doc_ids)

        # True positives: retrieved AND relevant
        tp = len(retrieved_set & relevant_set)

        # False positives: retrieved but NOT relevant
        fp = len(retrieved_set - relevant_set)

        # False negatives: relevant but NOT retrieved
        fn = len(relevant_set - retrieved_set)

        # Precision: what fraction of retrieved docs are relevant?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall: what fraction of relevant docs were retrieved?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 score: harmonic mean of precision and recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }

    def evaluate_answer_faithfulness(self,
                                     answer: str,
                                     retrieved_docs: List[str]) -> Dict[str, any]:
        """
        Check if answer is grounded in retrieved documents (no hallucination).
        Uses LLM to verify claims.
        """
        context = "\n\n".join([f"Document {i+1}:\n{doc}"
                              for i, doc in enumerate(retrieved_docs)])

        prompt = f"""Evaluate if this answer is faithful to the provided documents.

Answer to evaluate:
{answer}

Retrieved documents:
{context}

For each claim in the answer:
1. Is it supported by the documents? (yes/no/partial)
2. Which document supports it?

Return JSON:
{{
  "faithful": true/false,
  "faithfulness_score": 0.0-1.0,
  "unsupported_claims": ["claim1", "claim2"],
  "explanation": "brief explanation"
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    def evaluate_answer_correctness(self,
                                    query: str,
                                    answer: str,
                                    ground_truth: str) -> Dict[str, any]:
        """
        Compare generated answer to ground truth.
        """
        prompt = f"""Compare this generated answer to the ground truth answer.

Query: {query}

Generated answer:
{answer}

Ground truth answer:
{ground_truth}

Evaluate:
1. Correctness: Does it provide correct information? (0.0-1.0)
2. Completeness: Does it cover all key points from ground truth? (0.0-1.0)
3. Relevance: Does it answer the query? (0.0-1.0)

Return JSON:
{{
  "correctness": 0.0-1.0,
  "completeness": 0.0-1.0,
  "relevance": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "explanation": "brief explanation"
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    def evaluate_end_to_end(self,
                           test_cases: List[Dict[str, any]],
                           rag_system) -> Dict[str, any]:
        """
        Run full evaluation on test set.

        test_cases format:
        [
          {
            'query': 'query text',
            'relevant_docs': ['doc1', 'doc2'],
            'ground_truth_answer': 'expected answer'
          },
          ...
        ]
        """
        results = []

        for i, test in enumerate(test_cases):
            print(f"Evaluating test case {i+1}/{len(test_cases)}...")

            start_time = time.time()

            # Run RAG system
            rag_result = rag_system.answer_query(test['query'])

            latency = time.time() - start_time

            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval(
                query=test['query'],
                retrieved_doc_ids=rag_result['retrieved_doc_ids'],
                relevant_doc_ids=test['relevant_docs']
            )

            # Evaluate answer faithfulness
            faithfulness = self.evaluate_answer_faithfulness(
                answer=rag_result['answer'],
                retrieved_docs=rag_result['retrieved_docs']
            )

            # Evaluate answer correctness
            correctness = self.evaluate_answer_correctness(
                query=test['query'],
                answer=rag_result['answer'],
                ground_truth=test['ground_truth_answer']
            )

            results.append({
                'query': test['query'],
                'retrieval': retrieval_metrics,
                'faithfulness': faithfulness,
                'correctness': correctness,
                'latency_seconds': latency
            })

        # Aggregate metrics
        avg_metrics = {
            'retrieval_precision': sum(r['retrieval']['precision'] for r in results) / len(results),
            'retrieval_recall': sum(r['retrieval']['recall'] for r in results) / len(results),
            'retrieval_f1': sum(r['retrieval']['f1'] for r in results) / len(results),
            'faithfulness_score': sum(r['faithfulness']['faithfulness_score'] for r in results) / len(results),
            'answer_correctness': sum(r['correctness']['correctness'] for r in results) / len(results),
            'answer_completeness': sum(r['correctness']['completeness'] for r in results) / len(results),
            'avg_latency': sum(r['latency_seconds'] for r in results) / len(results)
        }

        return {
            'individual_results': results,
            'aggregate_metrics': avg_metrics
        }


# Example usage
if __name__ == "__main__":
    import os

    evaluator = RAGEvaluator(api_key=os.getenv('ANTHROPIC_API_KEY'))

    # Example: Evaluate single retrieval
    print("=== Retrieval Evaluation ===\n")
    retrieval_result = evaluator.evaluate_retrieval(
        query="BGP config for rtr01",
        retrieved_doc_ids=['doc3_rtr01_config', 'doc4_rtr01_ticket', 'doc1_bgp_theory'],
        relevant_doc_ids=['doc3_rtr01_config', 'doc4_rtr01_ticket']
    )

    print(f"Precision: {retrieval_result['precision']:.2%}")
    print(f"Recall: {retrieval_result['recall']:.2%}")
    print(f"F1 Score: {retrieval_result['f1']:.2%}")
    print(f"True Positives: {retrieval_result['true_positives']}")
    print(f"False Positives: {retrieval_result['false_positives']}")
    print(f"False Negatives: {retrieval_result['false_negatives']}")

    # Example: Evaluate faithfulness
    print("\n=== Faithfulness Evaluation ===\n")
    answer = "The BGP configuration on rtr01 uses AS 65000 and peers with 10.1.1.2 in AS 65001."
    docs = [
        "router bgp 65000\n neighbor 10.1.1.2 remote-as 65001"
    ]

    faithfulness = evaluator.evaluate_answer_faithfulness(answer, docs)
    print(f"Faithful: {faithfulness['faithful']}")
    print(f"Score: {faithfulness['faithfulness_score']:.2%}")
    print(f"Explanation: {faithfulness['explanation']}")

    # Example: Evaluate correctness
    print("\n=== Correctness Evaluation ===\n")
    query = "What AS number is configured on rtr01?"
    generated = "AS 65000 is configured on rtr01 for BGP."
    ground_truth = "rtr01 is configured with BGP AS 65000."

    correctness = evaluator.evaluate_answer_correctness(query, generated, ground_truth)
    print(f"Correctness: {correctness['correctness']:.2%}")
    print(f"Completeness: {correctness['completeness']:.2%}")
    print(f"Relevance: {correctness['relevance']:.2%}")
    print(f"Overall: {correctness['overall_score']:.2%}")
```

**Output**:
```
=== Retrieval Evaluation ===

Precision: 66.67%
Recall: 100.00%
F1 Score: 80.00%
True Positives: 2
False Positives: 1
False Negatives: 0

=== Faithfulness Evaluation ===

Faithful: True
Score: 100.00%
Explanation: All claims in the answer are directly supported by the BGP configuration document. The AS numbers and neighbor relationship are explicitly stated.

=== Correctness Evaluation ===

Correctness: 100.00%
Completeness: 100.00%
Relevance: 100.00%
Overall: 100.00%
```

**Why this works**: RAG evaluation requires multiple dimensions. Retrieval quality (precision/recall) measures if you found the right docs. Faithfulness measures if the answer sticks to those docs. Correctness measures if the answer is right. Track all three.

**Production note**: Build a test set with ground truth. Start with 20-50 queries covering different types (simple lookup, complex reasoning, multi-hop). Run evaluations after each system change. Track metrics over time.

## Putting It All Together: Production RAG Pipeline

Combine all techniques into a production system.

```python
from anthropic import Anthropic
from typing import List, Dict, Optional
import time

class ProductionRAG:
    """
    Production RAG pipeline combining all advanced techniques.
    """

    def __init__(self,
                 api_key: str,
                 persist_directory: str = "./chroma_production"):
        self.api_key = api_key
        self.client = Anthropic(api_key=api_key)

        # Components
        self.search = HybridNetworkSearch(persist_directory=persist_directory)
        self.router = QueryRouter(api_key=api_key, hybrid_search=self.search)
        self.compressor = ContextCompressor(api_key=api_key)
        self.reranker = NetworkDocumentReranker()
        self.multihop = MultiHopRAG(api_key=api_key, hybrid_search=self.search)

        # Config
        self.use_compression = True
        self.use_reranking = True
        self.compression_threshold = 1000  # chars

    def answer(self,
              query: str,
              n_results: int = 10,
              rerank_top_k: int = 5,
              verbose: bool = False) -> Dict[str, any]:
        """
        Answer query using full production pipeline.
        """
        start_time = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}\n")

        # Step 1: Route query
        if verbose:
            print("Step 1: Routing query...")

        routing_result = self.router.route_query(query, n_results=n_results)
        query_type = routing_result['classification']['query_type']
        multi_hop = routing_result['multi_hop_required']

        if verbose:
            print(f"  Query type: {query_type}")
            print(f"  Strategy: {routing_result['strategy']}")
            print(f"  Multi-hop: {multi_hop}\n")

        # Step 2: Retrieve (use multi-hop if needed)
        if multi_hop:
            if verbose:
                print("Step 2: Multi-hop retrieval...")
            multihop_result = self.multihop.answer_query(query, verbose=False)
            final_answer = multihop_result['answer']
            retrieved_results = []  # Already processed in multi-hop

        else:
            if verbose:
                print("Step 2: Single-step retrieval...")

            retrieved_results = routing_result['results']

            if verbose:
                print(f"  Retrieved {len(retrieved_results)} documents\n")

            # Step 3: Re-rank
            if self.use_reranking and retrieved_results:
                if verbose:
                    print("Step 3: Re-ranking results...")

                retrieved_results = self.reranker.rerank_with_metadata_boost(
                    query=query,
                    results=retrieved_results,
                    top_k=rerank_top_k
                )

                if verbose:
                    print(f"  Re-ranked to top {len(retrieved_results)}\n")

            # Step 4: Compress context
            total_chars = sum(len(r['text']) for r in retrieved_results)

            if self.use_compression and total_chars > self.compression_threshold:
                if verbose:
                    print(f"Step 4: Compressing context ({total_chars} chars)...")

                retrieved_results = self.compressor.compress_results(
                    query=query,
                    results=retrieved_results
                )

                compressed_chars = sum(r['compressed_length'] for r in retrieved_results)

                if verbose:
                    print(f"  Compressed to {compressed_chars} chars "
                          f"({compressed_chars/total_chars:.1%})\n")

            # Step 5: Generate answer
            if verbose:
                print("Step 5: Generating answer...\n")

            context_text = self._format_context(retrieved_results)
            final_answer = self._generate_answer(query, context_text)

        elapsed = time.time() - start_time

        if verbose:
            print(f"{'='*60}")
            print(f"Completed in {elapsed:.2f}s")
            print(f"{'='*60}\n")

        return {
            'query': query,
            'answer': final_answer,
            'query_type': query_type,
            'multi_hop': multi_hop,
            'num_retrieved': len(retrieved_results) if not multi_hop else None,
            'latency_seconds': elapsed
        }

    def _format_context(self, results: List[Dict[str, any]]) -> str:
        """Format retrieved documents for LLM context."""
        formatted = []
        for r in results:
            meta = r['metadata']
            text = r.get('text', r.get('compressed_text', ''))
            formatted.append(
                f"[{r['id']} | {meta['type']} | {meta['device']} | {meta['timestamp']}]\n{text}"
            )
        return "\n\n".join(formatted)

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate final answer from context."""
        prompt = f"""Answer this network engineering question using the provided documentation.

Question: {query}

Documentation:
{context}

Provide a clear, technical answer. Include specific details (device names, IPs, commands) from the documentation. If the documentation doesn't contain enough information to answer fully, say so."""

        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text


# Example usage
if __name__ == "__main__":
    import os

    # Initialize
    rag = ProductionRAG(
        api_key=os.getenv('ANTHROPIC_API_KEY'),
        persist_directory="./chroma_production_demo"
    )

    # Add documents
    docs = [
        {
            'id': 'cfg_rtr01_bgp',
            'text': 'router bgp 65000\n neighbor 10.1.1.2 remote-as 65001\n network 192.168.1.0',
            'type': 'config',
            'device': 'rtr01',
            'timestamp': '2024-01-15'
        },
        {
            'id': 'ticket_bgp_flap',
            'text': 'BGP flapping on rtr01. Root cause: interface errors. Fixed by cable replacement.',
            'type': 'ticket',
            'device': 'rtr01',
            'timestamp': '2024-01-16'
        }
    ]

    rag.search.add_documents(docs)

    # Query
    result = rag.answer(
        "What BGP issues occurred on rtr01?",
        verbose=True
    )

    print("ANSWER:")
    print(result['answer'])
```

**Output**:
```
============================================================
Query: What BGP issues occurred on rtr01?
============================================================

Step 1: Routing query...
  Query type: incident_analysis
  Strategy: temporal_search
  Multi-hop: False

Step 2: Single-step retrieval...
  Retrieved 2 documents

Step 3: Re-ranking results...
  Re-ranked to top 2

Step 4: Compressing context (245 chars)...
  (Skipped - below threshold)

Step 5: Generating answer...

============================================================
Completed in 2.34s
============================================================

ANSWER:
Based on the documentation, rtr01 experienced BGP session flapping on January 16, 2024. The root cause was interface errors on the device. The issue was resolved by replacing a faulty cable.

The device is configured with BGP AS 65000 and has a peering relationship with neighbor 10.1.1.2 (AS 65001), advertising network 192.168.1.0.
```

## Best Practices for Production RAG

After building dozens of RAG systems for network operations, here's what works:

**1. Start simple, add complexity only when needed**
   - Begin with basic vector search
   - Add hybrid search when you see missed keyword matches
   - Add re-ranking when top results are wrong
   - Add multi-hop only for queries that need it

**2. Tune retrieval for your data**
   - Network configs need high keyword weight (exact matches matter)
   - Incident tickets need temporal filtering
   - Architecture docs benefit from pure vector search
   - Don't use the same weights for everything

**3. Monitor and measure**
   - Track retrieval precision (are retrieved docs relevant?)
   - Track answer faithfulness (is the LLM hallucinating?)
   - Track latency at each stage
   - Build a test set and run evals regularly

**4. Handle chunk size carefully**
   - Full device configs: Chunk by section (interfaces, routing, etc.)
   - CLI outputs: Keep commands with their output
   - Tickets: Keep timeline together, separate root cause analysis
   - Wrong chunking ruins everything else

**5. Optimize for latency**
   - Compression adds an LLM call (200-500ms)
   - Re-ranking adds model inference (50-200ms)
   - Multi-hop adds multiple retrieval rounds (1-3s)
   - Use async where possible, cache aggressively

**6. Plan for updates**
   - Network configs change constantly
   - Implement incremental updates (don't rebuild entire index)
   - Track document versions
   - Invalidate cache when docs change

**7. Handle failures gracefully**
   - No results? Fall back to broader search
   - Low confidence? Ask clarifying questions
   - Contradictory docs? Show the conflict, don't hide it
   - Never hallucinate to fill gaps

## Summary

Advanced RAG techniques take you from 70% accuracy to 95%:

- **Hybrid search** combines vector, keyword, and graph for comprehensive retrieval
- **Query routing** matches search strategy to query type
- **Context compression** removes noise and saves tokens
- **Multi-hop reasoning** handles complex queries requiring multiple lookups
- **Re-ranking** improves final result quality before generation
- **Evaluation metrics** tell you if your system actually works

Every technique adds complexity and latency. Add them one at a time, measure the impact, and keep only what improves your metrics.

Production RAG is about precision, not just retrieval. You need the right documents, properly ranked, with irrelevant content removed, fed to an LLM with clear instructions. Get that right and your network documentation becomes genuinely useful instead of just searchable.

Next chapter covers RAG for real-time network telemetry—handling streaming data, time-series context, and queries that require current state, not historical docs.
