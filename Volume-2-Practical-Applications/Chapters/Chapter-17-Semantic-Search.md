# Chapter 17: Semantic Search for Network Documentation

## Why This Chapter Matters

Traditional search: Type "BGP configuration" → Finds exact matches

Semantic search: Type "How do I peer with cloud providers?" → Finds BGP, peering policies, cloud integration docs, even if they don't contain those exact words

**The difference**: Keyword search matches words. Semantic search understands meaning.

This chapter builds advanced search capabilities:
- Multi-query retrieval
- Hybrid search (keyword + semantic)
- Re-ranking results
- Query expansion
- Contextual compression

---

## Section 1: Beyond Basic Similarity Search

### The Problem with Simple Retrieval

```python
# Simple retrieval
query = "How do I configure routing?"
results = vectorstore.similarity_search(query, k=3)
# Returns: Generic routing docs

# But user actually needs BGP-specific config
# The query was too vague
```

**Semantic search improvements**:
1. **Understand intent**: Is user asking how-to, troubleshooting, or conceptual?
2. **Expand query**: Add related terms automatically
3. **Re-rank**: Move most relevant results to top
4. **Compress**: Return only relevant parts of documents

### Multi-Query Retrieval

Generate multiple variations of the question to catch more relevant docs.

```python
# multi_query_retriever.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from typing import List, Set
import re

class MultiQueryRetriever:
    """Generate multiple query variations for better retrieval."""

    def __init__(self, vectorstore: Chroma, llm: ChatAnthropic):
        self.vectorstore = vectorstore
        self.llm = llm

        self.query_generation_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant helping generate search queries.
Given a user question about network engineering, generate 3 alternative
versions of the question to retrieve relevant documents.

Provide variations that:
1. Use technical terminology
2. Use different phrasing
3. Focus on different aspects

Original question: {question}

Generate 3 alternative questions (one per line):""")

    def generate_queries(self, original_query: str) -> List[str]:
        """Generate multiple query variations."""
        chain = self.query_generation_prompt | self.llm | StrOutputParser()

        response = chain.invoke({"question": original_query})

        # Parse alternative queries
        alternatives = [q.strip() for q in response.split('\n') if q.strip()]

        # Add original query
        all_queries = [original_query] + alternatives[:3]

        return all_queries

    def retrieve(self, question: str, k_per_query: int = 3) -> List:
        """Retrieve documents using multiple query variations."""
        # Generate query variations
        queries = self.generate_queries(question)
        print(f"Generated {len(queries)} queries:")
        for q in queries:
            print(f"  - {q}")

        # Retrieve docs for each query
        all_docs = []
        seen_content = set()

        for query in queries:
            docs = self.vectorstore.similarity_search(query, k=k_per_query)

            # Deduplicate
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)

        print(f"\nRetrieved {len(all_docs)} unique documents")

        return all_docs


# Example usage
if __name__ == "__main__":
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    # Setup
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key="your-api-key",
        temperature=0.7
    )

    retriever = MultiQueryRetriever(vectorstore, llm)

    # Retrieve with query expansion
    docs = retriever.retrieve("How do I peer with AWS?", k_per_query=2)

    # Example output:
    # Generated 4 queries:
    #   - How do I peer with AWS?
    #   - What are the BGP peering requirements for Amazon Web Services?
    #   - How to configure AWS Direct Connect or VPN tunneling?
    #   - What routing protocols work with AWS cloud connectivity?
```

---

## Section 2: Hybrid Search (Keyword + Semantic)

### Why Hybrid?

**Semantic search**: Great for concepts, but misses exact matches
**Keyword search**: Great for specific terms (AS numbers, IP addresses)

**Hybrid**: Best of both worlds.

```python
# hybrid_search.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from typing import List

class HybridNetworkRetriever:
    """Combine semantic and keyword search."""

    def __init__(self, vectorstore: Chroma, documents: List):
        self.vectorstore = vectorstore
        self.documents = documents

        # Semantic retriever (vector similarity)
        self.semantic_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        # Keyword retriever (BM25 - like Elasticsearch)
        self.keyword_retriever = BM25Retriever.from_documents(documents)
        self.keyword_retriever.k = 5

        # Ensemble combines both
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.semantic_retriever, self.keyword_retriever],
            weights=[0.6, 0.4]  # 60% semantic, 40% keyword
        )

    def search(self, query: str, k: int = 5) -> List:
        """Hybrid search."""
        results = self.ensemble_retriever.get_relevant_documents(query)
        return results[:k]

    def semantic_only(self, query: str, k: int = 5) -> List:
        """Semantic search only."""
        return self.semantic_retriever.get_relevant_documents(query)[:k]

    def keyword_only(self, query: str, k: int = 5) -> List:
        """Keyword search only."""
        return self.keyword_retriever.get_relevant_documents(query)[:k]


# Example: Why hybrid matters
"""
Query: "AS 64512"

Semantic search might find:
- "Autonomous System configuration"
- "BGP AS number policies"
- Generic docs about AS numbers

Keyword search finds:
- Exact matches for "AS 64512" or "AS64512"
- Specific configs mentioning this AS

Hybrid search returns both, ranked by relevance.
"""
```

---

## Section 3: Contextual Compression

### The Problem: Too Much Context

You retrieve 5 documents of 1000 tokens each = 5000 tokens of context.
But only 10% is actually relevant to the question.

**Contextual compression**: Extract only relevant parts.

```python
# contextual_compressor.py
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma

class CompressedRetriever:
    """Retrieve documents and extract only relevant parts."""

    def __init__(self, vectorstore: Chroma, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-haiku-20241022",  # Use Haiku for compression (cheaper)
            api_key=api_key,
            temperature=0.0
        )

        # Base retriever
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        # Compressor extracts relevant parts
        compressor = LLMChainExtractor.from_llm(self.llm)

        # Compressed retriever
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

    def retrieve(self, question: str) -> List:
        """Retrieve compressed documents."""
        return self.retriever.get_relevant_documents(question)


# Example
"""
Before compression:
Doc 1 (1000 tokens): Full BGP configuration guide...
Doc 2 (1200 tokens): Complete OSPF manual...
Doc 3 (800 tokens): Network security policies...
Total: 3000 tokens

After compression:
Doc 1 (150 tokens): "To configure BGP with MD5 auth, use..."
Doc 2 (100 tokens): "OSPF area design requires..."
Doc 3 (80 tokens): "BGP authentication policy states..."
Total: 330 tokens (89% reduction!)

This saves tokens and improves LLM focus.
"""
```

---

## Section 4: Re-Ranking Results

### Initial Retrieval vs Re-Ranking

**Initial retrieval**: Fast, returns 20 candidates
**Re-ranking**: Slow, scores all 20, returns top 5

More accurate but more expensive.

```python
# reranker.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import json

class RankedDocument(BaseModel):
    index: int = Field(description="Original document index")
    relevance_score: int = Field(description="Relevance score 0-10")
    reasoning: str = Field(description="Why this score")

class DocumentReranker:
    """Re-rank retrieved documents for better relevance."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-haiku-20241022",  # Fast model for reranking
            api_key=api_key,
            temperature=0.0
        )

        self.parser = JsonOutputParser(pydantic_object=List[RankedDocument])

    def rerank(
        self,
        query: str,
        documents: List,
        top_k: int = 5
    ) -> List:
        """Re-rank documents by relevance to query."""

        # Format documents for LLM
        doc_texts = []
        for i, doc in enumerate(documents):
            content_preview = doc.page_content[:300] + "..."
            doc_texts.append(f"Document {i}:\n{content_preview}")

        docs_formatted = "\n\n".join(doc_texts)

        prompt = ChatPromptTemplate.from_template("""
Rate the relevance of each document to the user's question.
Score each document from 0-10 (10 = highly relevant, 0 = not relevant).

Question: {query}

Documents:
{documents}

Return a JSON array of objects with fields: index, relevance_score, reasoning

JSON array:""")

        response = self.llm.invoke(
            prompt.format(query=query, documents=docs_formatted)
        )

        # Parse rankings
        try:
            rankings = json.loads(response.content)
        except:
            # Fallback: return original order
            return documents[:top_k]

        # Sort by score
        ranked = sorted(rankings, key=lambda x: x['relevance_score'], reverse=True)

        # Return top K documents
        reranked_docs = []
        for item in ranked[:top_k]:
            idx = item['index']
            if idx < len(documents):
                reranked_docs.append(documents[idx])

        return reranked_docs


# Example usage
if __name__ == "__main__":
    reranker = DocumentReranker(api_key="your-api-key")

    # Simulate retrieved documents
    from langchain.schema import Document

    docs = [
        Document(page_content="VLAN configuration standards for enterprise..."),
        Document(page_content="BGP peering with AWS requires MD5 authentication..."),
        Document(page_content="General network security overview..."),
        Document(page_content="Detailed AWS Direct Connect BGP setup guide..."),
        Document(page_content="OSPF routing in branch offices..."),
    ]

    query = "How to configure BGP peering with AWS?"

    reranked = reranker.rerank(query, docs, top_k=3)

    print("Re-ranked results:")
    for i, doc in enumerate(reranked):
        print(f"{i+1}. {doc.page_content[:80]}...")

    # Output prioritizes docs 2 and 4 (AWS BGP specific)
```

---

## Section 5: Query Understanding and Expansion

### Intent Classification

Understand *what* the user wants before searching.

```python
# query_classifier.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from enum import Enum

class QueryIntent(str, Enum):
    HOW_TO = "how_to"  # "How do I configure X?"
    TROUBLESHOOTING = "troubleshooting"  # "Why isn't X working?"
    CONCEPTUAL = "conceptual"  # "What is X?"
    REFERENCE = "reference"  # "Show me X command syntax"
    POLICY = "policy"  # "What's our policy for X?"

class ClassifiedQuery(BaseModel):
    intent: QueryIntent = Field(description="The query intent")
    entities: list[str] = Field(description="Key networking entities (protocols, devices)")
    expanded_query: str = Field(description="Expanded query with synonyms")
    suggested_filters: dict = Field(description="Metadata filters to apply")

class QueryClassifier:
    """Classify and expand user queries."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            api_key=api_key,
            temperature=0.0
        )

        self.parser = JsonOutputParser(pydantic_object=ClassifiedQuery)

    def classify(self, query: str) -> ClassifiedQuery:
        """Classify query intent and expand."""
        prompt = ChatPromptTemplate.from_template("""
Analyze this network engineering question.

Question: {query}

Identify:
1. Intent (how_to, troubleshooting, conceptual, reference, or policy)
2. Key entities (protocols like BGP/OSPF, devices, technologies)
3. Expanded query (add synonyms and related terms)
4. Suggested metadata filters (e.g., {"doc_type": "configuration_guide"})

Return JSON matching this schema:
{format_instructions}

JSON:""")

        response = self.llm.invoke(
            prompt.format(
                query=query,
                format_instructions=self.parser.get_format_instructions()
            )
        )

        result = json.loads(response.content)
        return ClassifiedQuery(**result)


# Example usage
if __name__ == "__main__":
    classifier = QueryClassifier(api_key="your-api-key")

    queries = [
        "How do I configure BGP with AWS?",
        "Users can't access the internet from VLAN 10",
        "What's the difference between OSPF and EIGRP?",
        "Show me the syntax for ACL configuration",
        "What's our policy for SSH access?"
    ]

    for query in queries:
        result = classifier.classify(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result.intent}")
        print(f"Entities: {result.entities}")
        print(f"Expanded: {result.expanded_query}")
        print(f"Filters: {result.suggested_filters}")

# Expected output:
"""
Query: How do I configure BGP with AWS?
Intent: how_to
Entities: ['BGP', 'AWS']
Expanded: Configure Border Gateway Protocol peering with Amazon Web Services cloud
Filters: {'doc_type': 'configuration_guide', 'protocol': 'BGP'}

Query: Users can't access the internet from VLAN 10
Intent: troubleshooting
Entities: ['VLAN', 'internet connectivity']
Expanded: Troubleshoot Layer 3 routing and gateway issues for VLAN 10 internet access
Filters: {'doc_type': 'troubleshooting_guide', 'category': 'connectivity'}
"""
```

---

## Section 6: Production Semantic Search System

### Complete Implementation

```python
# semantic_search_system.py
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from multi_query_retriever import MultiQueryRetriever
from reranker import DocumentReranker
from query_classifier import QueryClassifier
from typing import List, Dict

class SemanticNetworkSearch:
    """Production semantic search system."""

    def __init__(self, api_key: str, vectorstore_path: str = "./chroma_db"):
        self.api_key = api_key

        # Components
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=self.embeddings
        )

        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.0
        )

        self.multi_query = MultiQueryRetriever(self.vectorstore, self.llm)
        self.reranker = DocumentReranker(api_key)
        self.classifier = QueryClassifier(api_key)

    def search(
        self,
        query: str,
        use_multi_query: bool = True,
        use_reranking: bool = True,
        top_k: int = 5
    ) -> Dict:
        """
        Advanced semantic search with all optimizations.

        Args:
            query: User question
            use_multi_query: Generate query variations
            use_reranking: Re-rank results
            top_k: Number of results to return

        Returns:
            Dict with results, metadata, and search stats
        """
        # Step 1: Classify query
        classification = self.classifier.classify(query)

        print(f"Intent: {classification.intent}")
        print(f"Expanded query: {classification.expanded_query}\n")

        # Step 2: Retrieve candidates
        if use_multi_query:
            # Multi-query retrieval
            candidates = self.multi_query.retrieve(
                classification.expanded_query,
                k_per_query=5
            )
        else:
            # Simple retrieval
            candidates = self.vectorstore.similarity_search(
                classification.expanded_query,
                k=15
            )

        print(f"Retrieved {len(candidates)} candidates")

        # Step 3: Re-rank
        if use_reranking and len(candidates) > top_k:
            final_docs = self.reranker.rerank(query, candidates, top_k=top_k)
            print(f"Re-ranked to top {len(final_docs)}")
        else:
            final_docs = candidates[:top_k]

        # Step 4: Format results
        results = []
        for i, doc in enumerate(final_docs):
            results.append({
                "rank": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "preview": doc.page_content[:200] + "..."
            })

        return {
            "query": query,
            "intent": classification.intent,
            "expanded_query": classification.expanded_query,
            "entities": classification.entities,
            "results": results,
            "num_results": len(results),
            "search_method": {
                "multi_query": use_multi_query,
                "reranking": use_reranking
            }
        }

    def answer_question(self, query: str, top_k: int = 5) -> str:
        """Search and generate answer."""
        # Search
        search_results = self.search(query, top_k=top_k)

        # Format context
        context = "\n\n".join([
            f"[Source {r['rank']}]: {r['content']}"
            for r in search_results['results']
        ])

        # Generate answer
        prompt = f"""Answer the question using the following documentation.
Cite sources using [Source N] notation.

Documentation:
{context}

Question: {query}

Answer:"""

        response = self.llm.invoke(prompt)
        return response.content


# Example usage
if __name__ == "__main__":
    search = SemanticNetworkSearch(api_key="your-api-key")

    # Search only
    results = search.search("How do I configure BGP with AWS?", top_k=3)

    print(f"\nQuery: {results['query']}")
    print(f"Intent: {results['intent']}")
    print(f"Expanded: {results['expanded_query']}\n")

    print("Top results:")
    for result in results['results']:
        print(f"{result['rank']}. {result['preview']}")
        print(f"   Source: {result['metadata'].get('filename', 'Unknown')}\n")

    # Search and answer
    answer = search.answer_question("What's our VLAN policy for servers?")
    print(f"\nAnswer:\n{answer}")
```

---

## What Can Go Wrong

**1. Query expansion too broad**
- Generates queries that are too different from original
- Retrieve irrelevant documents
- Solution: Limit to 2-3 variations, validate relevance

**2. Re-ranking cost**
- Re-ranking 20 documents costs tokens
- Can be expensive at scale
- Solution: Use cheap model (Haiku), cache rankings

**3. Hybrid search imbalance**
- Weight keyword search too high: Misses concepts
- Weight semantic too high: Misses exact matches
- Solution: Tune weights (60/40 is good starting point)

**4. Compression loses context**
- Aggressive compression removes necessary context
- LLM can't answer from fragments
- Solution: Set minimum chunk size (100-150 tokens)

---

## Key Takeaways

1. **Multi-query retrieval** catches more relevant docs
2. **Hybrid search** combines semantic + keyword strengths
3. **Contextual compression** reduces tokens, improves focus
4. **Re-ranking** improves relevance accuracy
5. **Query classification** enables intent-based search

Next chapter: RAG production patterns and optimization.
