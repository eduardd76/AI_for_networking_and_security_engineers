# Chapter 14: RAG Fundamentals

## Why This Chapter Matters

You generated 500 pages of network documentation (Chapter 13). A engineer asks: "What's the VLAN policy?"

**Without RAG**: Search 500 pages manually, find 5 relevant sections, read them, synthesize answer. **Time**: 20 minutes.

**With RAG**: Ask the question, get instant answer with exact source citations. **Time**: 5 seconds.

**RAG = Retrieval Augmented Generation**

This is the foundation for making AI systems that know your specific network, not just general networking.

This chapter teaches:
- How RAG actually works (step-by-step)
- Embeddings explained for network engineers
- Building your first RAG system
- Choosing vector databases
- Evaluating retrieval quality

**The payoff**: AI that answers questions using YOUR documentation, not generic knowledge.

---

## Section 1: Understanding RAG Architecture

### The Problem LLMs Alone Can't Solve

**LLM without RAG**:
```python
response = llm.invoke("What's our BGP policy for AWS peering?")
# Output: "I don't have access to your specific BGP policies..."
```

The LLM doesn't know your network. It only knows general networking concepts from training data.

**LLM with RAG**:
```python
# 1. Search your documentation
relevant_docs = search("BGP policy AWS peering")

# 2. Add docs to prompt
prompt = f"""Using these documents:
{relevant_docs}

Answer: What's our BGP policy for AWS peering?"""

# 3. LLM answers using YOUR docs
response = llm.invoke(prompt)
# Output: "According to our BGP Peering Guide (section 4.2), AWS peering requires..."
```

Now the LLM has your specific information as context.

### RAG Flow Diagram

```
User Question: "What's the VLAN policy?"
    ↓
Step 1: RETRIEVAL
    ↓
Vector Database Search
    → Find documents about VLANs
    → Score by relevance
    → Return top 5 chunks
    ↓
Relevant Documents:
    - "VLAN Standards.md" (section 2.1)
    - "Network Policy.md" (section 5)
    - "VLAN-Assignment-Guide.md"
    ↓
Step 2: AUGMENTATION
    ↓
Build Enhanced Prompt:
    System: You are a network documentation assistant
    Context: [Insert the 5 relevant document chunks]
    Question: What's the VLAN policy?
    ↓
Step 3: GENERATION
    ↓
LLM Generates Answer
    Using retrieved context
    With source citations
    ↓
Final Answer:
    "VLANs 10-100 are for user access (VLAN Standards.md, section 2.1).
     VLANs 200-300 are for servers (Network Policy.md, section 5)..."
```

**Key Insight**: RAG doesn't train the LLM. It gives the LLM relevant context at query time.

---

## Section 2: Embeddings Explained for Network Engineers

### What Are Embeddings?

**Mental model**: Embeddings are like IP addresses for meaning.

Just as:
- IP address `192.168.1.1` identifies a device's location
- Embedding `[0.23, -0.45, 0.67, ...]` identifies text's semantic location

**Example**:
```python
Text: "BGP routing protocol"
Embedding: [0.23, -0.45, 0.67, 0.12, ..., -0.34]  # 384 numbers

Text: "Border Gateway Protocol"
Embedding: [0.21, -0.43, 0.69, 0.15, ..., -0.32]  # Similar values!

Text: "VLAN configuration"
Embedding: [-0.52, 0.31, -0.22, 0.05, ..., 0.88]  # Different values
```

**Why similar?**: "BGP routing protocol" and "Border Gateway Protocol" mean the same thing, so their embeddings are close.

### How Embeddings Enable Search

**Traditional keyword search**:
```
Query: "BGP configuration"
Matches: Documents containing exact words "BGP" AND "configuration"
Misses: Documents saying "Border Gateway Protocol setup"
```

**Embedding-based semantic search**:
```
Query: "BGP configuration"
Query Embedding: [0.23, -0.45, 0.67, ...]

Compare to all document embeddings:
Doc 1: "BGP setup guide" → embedding [0.21, -0.43, 0.69, ...] → Similarity: 0.95 ✓
Doc 2: "OSPF config" → embedding [-0.12, 0.67, -0.34, ...] → Similarity: 0.23 ✗
Doc 3: "Border Gateway Protocol" → embedding [0.22, -0.44, 0.68, ...] → Similarity: 0.93 ✓

Return: Doc 1 and Doc 3 (high similarity)
```

**The magic**: Finds relevant docs even if they don't contain exact keywords.

### Generating Embeddings

```python
# embeddings_demo.py
from sentence_transformers import SentenceTransformer

# Load embedding model (runs locally, free)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Network documentation snippets
docs = [
    "BGP is a path vector routing protocol",
    "Border Gateway Protocol connects autonomous systems",
    "VLAN 10 is configured for guest WiFi access",
    "OSPF uses Dijkstra's algorithm for route calculation",
]

# Generate embeddings
embeddings = model.encode(docs)

print(f"Document 1: {docs[0]}")
print(f"Embedding shape: {embeddings[0].shape}")  # (384,) - array of 384 numbers
print(f"First 10 values: {embeddings[0][:10]}")

# Compare similarity
from scipy.spatial.distance import cosine

# BGP vs Border Gateway Protocol
sim_bgp = 1 - cosine(embeddings[0], embeddings[1])
print(f"\nSimilarity (BGP descriptions): {sim_bgp:.3f}")  # ~0.85 (very similar)

# BGP vs VLAN
sim_diff = 1 - cosine(embeddings[0], embeddings[2])
print(f"Similarity (BGP vs VLAN): {sim_diff:.3f}")  # ~0.25 (not similar)
```

**Output**:
```
Document 1: BGP is a path vector routing protocol
Embedding shape: (384,)
First 10 values: [ 0.234 -0.451  0.672  0.123 ...]

Similarity (BGP descriptions): 0.847
Similarity (BGP vs VLAN): 0.231
```

**Takeaway**: High similarity = semantically related, even with different words.

---

## Section 3: Building RAG from Scratch

### Step-by-Step RAG Implementation

```python
# rag_from_scratch.py
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Tuple

class SimpleRAG:
    """RAG system built from scratch (no LangChain)."""

    def __init__(self, api_key: str):
        # Embedding model (local, free)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # LLM (API, costs money)
        self.llm = Anthropic(api_key=api_key)

        # Document storage
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents: List[str]):
        """Add documents to the knowledge base."""
        print(f"Adding {len(documents)} documents...")

        # Store documents
        self.documents.extend(documents)

        # Generate embeddings
        new_embeddings = self.embedding_model.encode(documents)
        self.embeddings.extend(new_embeddings)

        print(f"Total documents: {len(self.documents)}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve most relevant documents for a query."""

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Calculate similarity to all documents
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = 1 - cosine(query_embedding, doc_embedding)
            similarities.append((i, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top K documents with scores
        results = []
        for i, score in similarities[:top_k]:
            results.append((self.documents[i], score))

        return results

    def generate_answer(
        self,
        query: str,
        context_docs: List[Tuple[str, float]]
    ) -> str:
        """Generate answer using retrieved context."""

        # Format context
        context = "\n\n".join([
            f"[Source {i+1}, relevance: {score:.2f}]\n{doc}"
            for i, (doc, score) in enumerate(context_docs)
        ])

        # Build prompt
        prompt = f"""Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}

Answer (cite sources using [Source N]):"""

        # Call LLM
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def query(self, question: str, top_k: int = 3) -> dict:
        """Complete RAG query: retrieve + generate."""

        print(f"\nQuery: {question}")
        print("="*60)

        # Step 1: Retrieve relevant documents
        print("\nStep 1: Retrieving relevant documents...")
        relevant_docs = self.retrieve(question, top_k=top_k)

        for i, (doc, score) in enumerate(relevant_docs, 1):
            print(f"  [{i}] Relevance: {score:.3f} - {doc[:60]}...")

        # Step 2: Generate answer
        print("\nStep 2: Generating answer...")
        answer = self.generate_answer(question, relevant_docs)

        print("\nAnswer:")
        print(answer)

        return {
            "question": question,
            "answer": answer,
            "sources": relevant_docs
        }


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = SimpleRAG(api_key="your-api-key")

    # Add network documentation
    documents = [
        """VLAN Assignment Policy
        VLANs 10-100: User access networks
        VLANs 200-300: Server networks
        VLANs 500-600: Guest and IoT devices
        VLAN 999: Quarantine/Remediation
        All VLAN assignments require NetOps approval.""",

        """BGP Peering with AWS
        AS Number for AWS peering: 64512
        Required: MD5 authentication on all BGP sessions
        IP ranges: Use 169.254.0.0/16 for peering
        Enable BFD for fast failure detection
        Contact: network-ops@company.com for new peers""",

        """OSPF Area Design
        Area 0 (Backbone): Core routers only
        Area 1: Branch offices
        Area 2: Datacenters
        All areas must connect to Area 0
        MTU must be 1500 across all areas""",

        """ACL Standard for Management Access
        Permit SSH (port 22) from 10.0.0.0/8 only
        Permit HTTPS (port 443) from 10.0.0.0/8 only
        Deny all other management traffic
        Log all denied attempts
        Review ACL logs weekly""",

        """Switch Port Security Guidelines
        Maximum 2 MAC addresses per access port
        Violation mode: restrict (not shutdown)
        Enable DHCP snooping on all access VLANs
        Enable DAI (Dynamic ARP Inspection)
        Enable IP Source Guard on untrusted ports"""
    ]

    rag.add_documents(documents)

    # Query the system
    questions = [
        "What VLAN should I use for a new server?",
        "How do I configure BGP with AWS?",
        "What's our switch port security policy?",
        "What EIGRP settings do we use?"  # Not in docs
    ]

    for question in questions:
        result = rag.query(question, top_k=3)
        print("\n" + "="*60 + "\n")
```

**Output**:
```
Query: What VLAN should I use for a new server?
============================================================

Step 1: Retrieving relevant documents...
  [1] Relevance: 0.687 - VLAN Assignment Policy...
  [2] Relevance: 0.423 - Switch Port Security Guidelines...
  [3] Relevance: 0.312 - OSPF Area Design...

Step 2: Generating answer...

Answer:
For a new server, you should use VLANs in the range 200-300 [Source 1].
According to our VLAN Assignment Policy, this range is designated for server networks.
Note that all VLAN assignments require NetOps approval before implementation [Source 1].
```

**This is RAG**: Retrieve relevant docs, augment prompt, generate answer.

---

## Section 4: Vector Databases

### Why Not Just Store Embeddings in a List?

**Problem with Python lists**:
```python
# 10,000 documents
# 384-dimensional embeddings
# Search requires: 10,000 similarity calculations

# Time: ~100ms for 10K docs
# Time: ~10 seconds for 1M docs
# Time: Hours for 100M docs
```

**Solution**: Vector databases with optimized search algorithms.

### Vector Database Options

**1. ChromaDB** (Recommended for getting started)
```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Search
results = vectorstore.similarity_search("BGP configuration", k=3)
```

**Pros**:
- Easy to use
- Runs locally
- Persistent storage
- Good for up to 1M documents

**Cons**:
- Not for massive scale (>10M docs)
- Single machine only

**2. FAISS** (For large scale)
```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_texts(
    texts=documents,
    embedding=embeddings
)

# Must manually save/load
vectorstore.save_local("./faiss_index")
```

**Pros**:
- Extremely fast
- Handles billions of vectors
- Facebook-backed

**Cons**:
- In-memory (careful with large datasets)
- More complex setup

**3. Pinecone, Weaviate, Qdrant** (Cloud services)

**Pros**:
- Fully managed
- Scales automatically
- High availability

**Cons**:
- Costs money ($0.10-1.00 per 1M vectors/month)
- Requires internet connection

### Recommendation

**Start**: ChromaDB (easy, local, free)
**Scale**: FAISS (fast, handles billions)
**Production**: Managed service (if budget allows)

---

## Section 5: Evaluating Retrieval Quality

### How to Know If RAG Is Working

**The Problem**: How do you know if retrieval is finding the right documents?

### Metrics That Matter

**1. Precision@K**
```
"Of the K documents retrieved, how many are actually relevant?"

Example:
Query: "VLAN policy"
Retrieved 5 documents
Actually relevant: 3 documents

Precision@5 = 3/5 = 0.60 (60%)
```

**2. Recall@K**
```
"Of all relevant documents, how many did we retrieve?"

Example:
Total relevant documents in database: 4
Retrieved in top 5: 3

Recall@5 = 3/4 = 0.75 (75%)
```

**3. Mean Reciprocal Rank (MRR)**
```
"How high is the first relevant document ranked?"

Example:
Query: "BGP policy"
Results:
  1. OSPF doc (not relevant)
  2. BGP doc (relevant!) ← First relevant at rank 2

MRR = 1/2 = 0.50
```

### Evaluation Framework

```python
# rag_evaluation.py
from typing import List, Dict, Set

class RAGEvaluator:
    """Evaluate RAG retrieval quality."""

    def __init__(self, rag_system):
        self.rag = rag_system

    def precision_at_k(
        self,
        query: str,
        expected_docs: Set[str],
        k: int = 5
    ) -> float:
        """Calculate precision@K."""

        # Retrieve top K documents
        retrieved = self.rag.retrieve(query, top_k=k)
        retrieved_set = set([doc for doc, score in retrieved])

        # How many are actually relevant?
        relevant_retrieved = retrieved_set.intersection(expected_docs)

        precision = len(relevant_retrieved) / k if k > 0 else 0
        return precision

    def recall_at_k(
        self,
        query: str,
        expected_docs: Set[str],
        k: int = 5
    ) -> float:
        """Calculate recall@K."""

        retrieved = self.rag.retrieve(query, top_k=k)
        retrieved_set = set([doc for doc, score in retrieved])

        relevant_retrieved = retrieved_set.intersection(expected_docs)

        recall = len(relevant_retrieved) / len(expected_docs) if expected_docs else 0
        return recall

    def mean_reciprocal_rank(
        self,
        query: str,
        expected_docs: Set[str]
    ) -> float:
        """Calculate MRR."""

        retrieved = self.rag.retrieve(query, top_k=20)  # Check top 20

        for rank, (doc, score) in enumerate(retrieved, 1):
            if doc in expected_docs:
                return 1.0 / rank

        return 0.0  # No relevant doc found

    def evaluate_test_set(
        self,
        test_queries: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate on a test set of queries."""

        precisions = []
        recalls = []
        mrrs = []

        for test_case in test_queries:
            query = test_case['query']
            expected = set(test_case['expected_docs'])

            precision = self.precision_at_k(query, expected, k=5)
            recall = self.recall_at_k(query, expected, k=5)
            mrr = self.mean_reciprocal_rank(query, expected)

            precisions.append(precision)
            recalls.append(recall)
            mrrs.append(mrr)

        return {
            'avg_precision@5': sum(precisions) / len(precisions),
            'avg_recall@5': sum(recalls) / len(recalls),
            'mean_reciprocal_rank': sum(mrrs) / len(mrrs)
        }


# Example usage
if __name__ == "__main__":
    # Test queries with known correct answers
    test_set = [
        {
            'query': 'What VLAN for servers?',
            'expected_docs': {'VLAN Assignment Policy'}
        },
        {
            'query': 'BGP configuration AWS',
            'expected_docs': {'BGP Peering with AWS'}
        },
        {
            'query': 'Switch security settings',
            'expected_docs': {'Switch Port Security Guidelines'}
        }
    ]

    evaluator = RAGEvaluator(rag)
    metrics = evaluator.evaluate_test_set(test_set)

    print("RAG Evaluation Results:")
    print(f"  Precision@5: {metrics['avg_precision@5']:.2f}")
    print(f"  Recall@5: {metrics['avg_recall@5']:.2f}")
    print(f"  MRR: {metrics['mean_reciprocal_rank']:.2f}")

    # Target: Precision > 0.8, Recall > 0.7, MRR > 0.8
```

**Good RAG performance**:
- Precision@5 > 0.80 (80%+ of retrieved docs are relevant)
- Recall@5 > 0.70 (Find 70%+ of relevant docs)
- MRR > 0.80 (First relevant doc in top 2-3 positions)

---

## What Can Go Wrong

**1. Poor document chunking**
- Chunks too small: Lack context
- Chunks too large: Dilute relevance
- Solution: 500-1000 tokens per chunk, with 50-100 overlap

**2. Wrong embedding model**
- Generic models don't understand networking terms
- "trunk" could mean tree trunk, car trunk, or network trunk
- Solution: Use domain-specific embeddings or fine-tune

**3. Too few retrieved documents**
- Miss relevant context
- Answer incomplete
- Solution: Retrieve more (k=5-10), let LLM filter

**4. No source validation**
- Retrieve irrelevant docs
- LLM hallucinates based on bad context
- Solution: Show retrieval scores, set minimum threshold

**5. Stale embeddings**
- Update documents but not embeddings
- RAG returns old information
- Solution: Re-embed when documents change

---

## Key Takeaways

1. **RAG = Retrieve + Augment + Generate** - Three distinct steps
2. **Embeddings enable semantic search** - Find meaning, not just keywords
3. **Vector databases optimize search** - Faster than brute force
4. **Evaluation prevents silent failures** - Measure precision, recall, MRR
5. **RAG doesn't train the LLM** - It provides context at query time

RAG is how you make AI systems that know YOUR network, not just networking in general.

Next chapter: LangChain integration (already written, Chapter 15).
