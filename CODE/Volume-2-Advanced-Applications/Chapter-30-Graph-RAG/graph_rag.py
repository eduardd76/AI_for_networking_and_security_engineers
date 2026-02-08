"""
Chapter 30: Graph RAG - Combining Graph + Vector Retrieval
Hybrid retrieval using knowledge graphs and vector embeddings
Author: Eduard Dulharu (@eduardd76)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import networkx as nx
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


# ============================================================================
# Data Models
# ============================================================================

class RetrievalStrategy(str, Enum):
    """Retrieval strategy"""
    VECTOR_ONLY = "VECTOR_ONLY"
    GRAPH_ONLY = "GRAPH_ONLY"
    HYBRID = "HYBRID"


@dataclass
class Document:
    """Document with embeddings"""
    doc_id: str
    content: str
    metadata: Dict[str, str]
    embedding: Optional[np.ndarray] = None
    entities: List[str] = field(default_factory=list)


@dataclass
class GraphNode:
    """Graph node representing document or entity"""
    node_id: str
    node_type: str  # "document", "entity", "concept"
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Edge between graph nodes"""
    source_id: str
    target_id: str
    edge_type: str  # "references", "relates_to", "precedes", etc.
    weight: float = 1.0


@dataclass
class RetrievalResult:
    """Single retrieval result"""
    doc_id: str
    content: str
    score: float
    retrieval_method: str  # "vector", "graph", "hybrid"
    metadata: Dict[str, str]
    explanation: str = ""


# ============================================================================
# Pydantic Models for LLM Outputs
# ============================================================================

class DocumentAnalysis(BaseModel):
    """LLM analysis of document"""
    entities: List[str] = Field(description="Key entities mentioned (3-8 items)")
    concepts: List[str] = Field(description="Key concepts/topics (2-5 items)")
    relationships: List[Tuple[str, str, str]] = Field(description="(entity1, relationship, entity2) tuples")
    summary: str = Field(description="One-sentence summary")


class QueryAnalysis(BaseModel):
    """LLM analysis of user query"""
    query_type: str = Field(description="FACTUAL, TROUBLESHOOTING, HOWTO, or COMPARISON")
    key_entities: List[str] = Field(description="Key entities to search for (2-5 items)")
    related_concepts: List[str] = Field(description="Related concepts (2-4 items)")
    recommended_strategy: str = Field(description="VECTOR_ONLY, GRAPH_ONLY, or HYBRID")
    reasoning: str = Field(description="Why this strategy is recommended")


class SynthesizedAnswer(BaseModel):
    """LLM synthesized answer"""
    answer: str = Field(description="Complete answer to the question (2-4 sentences)")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW")
    sources_used: List[str] = Field(description="Which sources were most relevant (2-4 items)")
    additional_context: str = Field(description="Additional relevant context (1-2 sentences)")


# ============================================================================
# Graph RAG System
# ============================================================================

class GraphRAGSystem:
    """Hybrid RAG using knowledge graph + vector search"""

    def __init__(self, api_key_anthropic: Optional[str] = None,
                 api_key_openai: Optional[str] = None):
        self.api_key_anthropic = api_key_anthropic or os.environ.get("ANTHROPIC_API_KEY")
        self.api_key_openai = api_key_openai or os.environ.get("OPENAI_API_KEY")

        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=self.api_key_anthropic,
            temperature=0
        )

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.api_key_openai
        )

        # Storage
        self.documents: Dict[str, Document] = {}
        self.graph = nx.DiGraph()
        self.entity_to_docs: Dict[str, List[str]] = {}  # entity -> list of doc_ids

    def add_document(self, doc_id: str, content: str, metadata: Dict[str, str]):
        """
        Add document to both vector and graph stores

        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Document metadata
        """
        # Create embedding
        embedding = np.array(self.embeddings.embed_query(content))

        # Analyze document to extract entities
        parser = PydanticOutputParser(pydantic_object=DocumentAnalysis)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document analyzer.
Extract entities (devices, protocols, technologies) and relationships.

{format_instructions}"""),
            ("human", """Analyze this network documentation:

{content}

Extract entities, concepts, and relationships.""")
        ])

        chain = prompt | self.llm | parser

        analysis = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "content": content[:2000]  # Limit to 2000 chars
        })

        # Create document
        doc = Document(
            doc_id=doc_id,
            content=content,
            metadata=metadata,
            embedding=embedding,
            entities=analysis.entities
        )

        self.documents[doc_id] = doc

        # Add to graph
        self.graph.add_node(doc_id, node_type="document", content=content[:200], **metadata)

        # Add entities to graph
        for entity in analysis.entities:
            entity_id = f"entity:{entity.lower().replace(' ', '_')}"

            # Add entity node
            if not self.graph.has_node(entity_id):
                self.graph.add_node(entity_id, node_type="entity", content=entity)

            # Link document to entity
            self.graph.add_edge(doc_id, entity_id, edge_type="mentions", weight=1.0)

            # Track entity->doc mapping
            if entity_id not in self.entity_to_docs:
                self.entity_to_docs[entity_id] = []
            self.entity_to_docs[entity_id].append(doc_id)

        # Add concept nodes
        for concept in analysis.concepts:
            concept_id = f"concept:{concept.lower().replace(' ', '_')}"

            if not self.graph.has_node(concept_id):
                self.graph.add_node(concept_id, node_type="concept", content=concept)

            self.graph.add_edge(doc_id, concept_id, edge_type="discusses", weight=1.0)

        # Add relationship edges
        for source, rel_type, target in analysis.relationships:
            source_id = f"entity:{source.lower().replace(' ', '_')}"
            target_id = f"entity:{target.lower().replace(' ', '_')}"

            if self.graph.has_node(source_id) and self.graph.has_node(target_id):
                self.graph.add_edge(source_id, target_id, edge_type=rel_type, weight=1.0)

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query to determine retrieval strategy

        Args:
            query: User query

        Returns:
            Query analysis with recommended strategy
        """
        parser = PydanticOutputParser(pydantic_object=QueryAnalysis)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query analyzer.
Determine the best retrieval strategy:
- VECTOR_ONLY: General questions, broad topics
- GRAPH_ONLY: Specific entity relationships, "what connects X to Y"
- HYBRID: Complex questions needing both context and relationships

{format_instructions}"""),
            ("human", """Analyze this query:

{query}

Determine query type, key entities, and recommended retrieval strategy.""")
        ])

        chain = prompt | self.llm | parser

        analysis = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "query": query
        })

        return analysis

    def vector_retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Vector-based retrieval using cosine similarity

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of retrieval results
        """
        # Get query embedding
        query_embedding = np.array(self.embeddings.embed_query(query))

        # Compute similarities
        similarities = []
        for doc_id, doc in self.documents.items():
            if doc.embedding is not None:
                sim = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    doc.embedding.reshape(1, -1)
                )[0][0]
                similarities.append((doc_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        results = []
        for doc_id, score in similarities[:top_k]:
            doc = self.documents[doc_id]
            results.append(RetrievalResult(
                doc_id=doc_id,
                content=doc.content,
                score=float(score),
                retrieval_method="vector",
                metadata=doc.metadata,
                explanation=f"Vector similarity: {score:.3f}"
            ))

        return results

    def graph_retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Graph-based retrieval using entity relationships

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of retrieval results
        """
        # Analyze query to extract entities
        analysis = self.analyze_query(query)

        # Find relevant graph nodes
        relevant_docs = set()
        doc_scores = {}

        for entity in analysis.key_entities:
            entity_id = f"entity:{entity.lower().replace(' ', '_')}"

            if entity_id in self.entity_to_docs:
                # Direct mentions
                for doc_id in self.entity_to_docs[entity_id]:
                    relevant_docs.add(doc_id)
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 2.0  # Direct mention

            # Also check neighbors in graph
            if self.graph.has_node(entity_id):
                for neighbor in self.graph.neighbors(entity_id):
                    if neighbor.startswith("doc:"):
                        relevant_docs.add(neighbor)
                        doc_scores[neighbor] = doc_scores.get(neighbor, 0) + 1.0  # Related entity

        # Also check for concept matches
        for concept in analysis.related_concepts:
            concept_id = f"concept:{concept.lower().replace(' ', '_')}"

            if self.graph.has_node(concept_id):
                for pred in self.graph.predecessors(concept_id):
                    if pred.startswith("doc:"):
                        relevant_docs.add(pred)
                        doc_scores[pred] = doc_scores.get(pred, 0) + 1.5  # Concept match

        # Sort by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-k
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                results.append(RetrievalResult(
                    doc_id=doc_id,
                    content=doc.content,
                    score=float(score),
                    retrieval_method="graph",
                    metadata=doc.metadata,
                    explanation=f"Graph traversal score: {score:.1f}"
                ))

        return results

    def hybrid_retrieve(self, query: str, top_k: int = 5,
                       vector_weight: float = 0.6, graph_weight: float = 0.4) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining vector and graph

        Args:
            query: Search query
            top_k: Number of results
            vector_weight: Weight for vector scores (0-1)
            graph_weight: Weight for graph scores (0-1)

        Returns:
            List of retrieval results
        """
        # Get results from both methods
        vector_results = self.vector_retrieve(query, top_k * 2)
        graph_results = self.graph_retrieve(query, top_k * 2)

        # Normalize scores
        vector_scores = {r.doc_id: r.score for r in vector_results}
        graph_scores = {r.doc_id: r.score for r in graph_results}

        # Normalize
        max_vector = max(vector_scores.values()) if vector_scores else 1.0
        max_graph = max(graph_scores.values()) if graph_scores else 1.0

        vector_scores = {k: v / max_vector for k, v in vector_scores.items()}
        graph_scores = {k: v / max_graph for k, v in graph_scores.items()}

        # Combine scores
        all_doc_ids = set(vector_scores.keys()) | set(graph_scores.keys())
        hybrid_scores = {}

        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            g_score = graph_scores.get(doc_id, 0.0)
            hybrid_scores[doc_id] = (v_score * vector_weight) + (g_score * graph_weight)

        # Sort by combined score
        sorted_docs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-k
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                results.append(RetrievalResult(
                    doc_id=doc_id,
                    content=doc.content,
                    score=float(score),
                    retrieval_method="hybrid",
                    metadata=doc.metadata,
                    explanation=f"Hybrid: vector={vector_scores.get(doc_id, 0):.2f}, graph={graph_scores.get(doc_id, 0):.2f}"
                ))

        return results

    def retrieve(self, query: str, strategy: Optional[RetrievalStrategy] = None,
                top_k: int = 5) -> List[RetrievalResult]:
        """
        Main retrieval method with automatic strategy selection

        Args:
            query: Search query
            strategy: Retrieval strategy (auto-selected if None)
            top_k: Number of results

        Returns:
            List of retrieval results
        """
        # Auto-select strategy if not provided
        if strategy is None:
            analysis = self.analyze_query(query)
            strategy = RetrievalStrategy[analysis.recommended_strategy]

        # Execute retrieval
        if strategy == RetrievalStrategy.VECTOR_ONLY:
            return self.vector_retrieve(query, top_k)
        elif strategy == RetrievalStrategy.GRAPH_ONLY:
            return self.graph_retrieve(query, top_k)
        else:  # HYBRID
            return self.hybrid_retrieve(query, top_k)

    def generate_answer(self, query: str, results: List[RetrievalResult]) -> Dict:
        """
        Generate answer using retrieved context

        Args:
            query: User query
            results: Retrieved results

        Returns:
            Synthesized answer
        """
        # Build context from results
        context = "\n\n".join([
            f"[Source {i+1}] {r.content[:500]}"
            for i, r in enumerate(results[:3])
        ])

        parser = PydanticOutputParser(pydantic_object=SynthesizedAnswer)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network documentation assistant.
Answer the question using the provided context.
Be specific and cite sources.

{format_instructions}"""),
            ("human", """Question: {query}

Context:
{context}

Provide a comprehensive answer.""")
        ])

        chain = prompt | self.llm | parser

        answer = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "query": query,
            "context": context
        })

        return {
            "query": query,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "sources_used": answer.sources_used,
            "additional_context": answer.additional_context,
            "retrieval_results": [
                {
                    "doc_id": r.doc_id,
                    "score": r.score,
                    "method": r.retrieval_method
                }
                for r in results[:3]
            ]
        }


# ============================================================================
# Examples
# ============================================================================

def example_1_basic_graph_rag():
    """Example 1: Basic Graph RAG setup"""
    print("=" * 80)
    print("Example 1: Basic Graph RAG Setup")
    print("=" * 80)

    rag = GraphRAGSystem()

    # Add sample documents
    docs = [
        {
            "id": "doc:bgp-basics",
            "content": "BGP (Border Gateway Protocol) is used for routing between autonomous systems. Core routers peer with each other using iBGP. The AS number for our network is 65001.",
            "metadata": {"topic": "routing", "protocol": "BGP"}
        },
        {
            "id": "doc:ospf-config",
            "content": "OSPF configuration on distribution switches requires Area 0 (backbone). All interfaces must be in the same area for proper adjacency formation.",
            "metadata": {"topic": "routing", "protocol": "OSPF"}
        },
        {
            "id": "doc:vlan-design",
            "content": "VLAN 100 is used for web servers, VLAN 200 for database servers. Access switches connect to distribution switches using trunk ports.",
            "metadata": {"topic": "switching", "technology": "VLAN"}
        }
    ]

    print("\nAdding documents to Graph RAG system...")
    for doc in docs:
        rag.add_document(doc["id"], doc["content"], doc["metadata"])
        print(f"  ✓ Added: {doc['id']}")

    print(f"\nSystem Statistics:")
    print(f"  Documents: {len(rag.documents)}")
    print(f"  Graph Nodes: {rag.graph.number_of_nodes()}")
    print(f"  Graph Edges: {rag.graph.number_of_edges()}")
    print(f"  Entities Tracked: {len(rag.entity_to_docs)}")


def example_2_vector_retrieval():
    """Example 2: Vector-only retrieval"""
    print("\n" + "=" * 80)
    print("Example 2: Vector-Only Retrieval")
    print("=" * 80)

    rag = GraphRAGSystem()

    # Add documents
    docs = [
        ("doc:1", "BGP peering configuration between core routers", {"type": "config"}),
        ("doc:2", "OSPF area design and best practices", {"type": "design"}),
        ("doc:3", "VLAN trunking between switches", {"type": "config"}),
    ]

    for doc_id, content, metadata in docs:
        rag.add_document(doc_id, content, metadata)

    # Query using vector search
    query = "How do I configure routing between routers?"
    print(f"\nQuery: {query}")
    print(f"Strategy: VECTOR_ONLY")

    results = rag.retrieve(query, strategy=RetrievalStrategy.VECTOR_ONLY, top_k=3)

    print(f"\nResults ({len(results)}):")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.doc_id}")
        print(f"   Score: {result.score:.3f}")
        print(f"   Content: {result.content[:100]}...")
        print(f"   {result.explanation}")


def example_3_graph_retrieval():
    """Example 3: Graph-only retrieval"""
    print("\n" + "=" * 80)
    print("Example 3: Graph-Only Retrieval")
    print("=" * 80)

    rag = GraphRAGSystem()

    # Add documents
    docs = [
        ("doc:1", "Core router connects to distribution switches via BGP", {"type": "topology"}),
        ("doc:2", "Distribution switches use OSPF internally", {"type": "routing"}),
        ("doc:3", "Access switches connect servers using VLANs", {"type": "switching"}),
    ]

    for doc_id, content, metadata in docs:
        rag.add_document(doc_id, content, metadata)

    # Query using graph search
    query = "What connects core router to switches?"
    print(f"\nQuery: {query}")
    print(f"Strategy: GRAPH_ONLY")

    results = rag.retrieve(query, strategy=RetrievalStrategy.GRAPH_ONLY, top_k=3)

    print(f"\nResults ({len(results)}):")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.doc_id}")
        print(f"   Score: {result.score:.1f}")
        print(f"   Content: {result.content[:100]}...")
        print(f"   {result.explanation}")


def example_4_hybrid_retrieval():
    """Example 4: Hybrid retrieval"""
    print("\n" + "=" * 80)
    print("Example 4: Hybrid Retrieval (Vector + Graph)")
    print("=" * 80)

    rag = GraphRAGSystem()

    # Add documents
    docs = [
        ("doc:bgp", "BGP is used between autonomous systems. Configure iBGP between core routers.", {"protocol": "BGP"}),
        ("doc:ospf", "OSPF provides internal routing. Use Area 0 for backbone.", {"protocol": "OSPF"}),
        ("doc:vlan", "VLANs segment traffic. Use trunking for inter-switch links.", {"technology": "VLAN"}),
        ("doc:stp", "Spanning Tree Protocol prevents loops in switched networks.", {"protocol": "STP"}),
    ]

    for doc_id, content, metadata in docs:
        rag.add_document(doc_id, content, metadata)

    # Query using hybrid search
    query = "How should I configure routing in my network?"
    print(f"\nQuery: {query}")
    print(f"Strategy: HYBRID")

    results = rag.retrieve(query, strategy=RetrievalStrategy.HYBRID, top_k=3)

    print(f"\nResults ({len(results)}):")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.doc_id}")
        print(f"   Score: {result.score:.3f}")
        print(f"   Method: {result.retrieval_method}")
        print(f"   Content: {result.content[:80]}...")
        print(f"   {result.explanation}")


def example_5_full_rag_pipeline():
    """Example 5: Complete RAG pipeline with answer generation"""
    print("\n" + "=" * 80)
    print("Example 5: Complete Graph RAG Pipeline")
    print("=" * 80)

    rag = GraphRAGSystem()

    # Add comprehensive documents
    docs = [
        ("doc:bgp-guide", """BGP Configuration Guide:
        Core routers use BGP AS 65001. Configure iBGP peering between core-router-01 and core-router-02.
        Use route reflectors for scalability. Always filter routes using prefix lists.""",
         {"type": "guide", "protocol": "BGP"}),

        ("doc:ospf-setup", """OSPF Setup:
        Distribution switches run OSPF Area 0. Enable OSPF on all trunk interfaces.
        Set reference bandwidth to 100000 for accurate cost calculation.""",
         {"type": "guide", "protocol": "OSPF"}),

        ("doc:vlan-std", """VLAN Standards:
        VLAN 100: Production servers
        VLAN 200: Database servers
        VLAN 999: Management
        All inter-switch links must be 802.1Q trunks.""",
         {"type": "standard", "technology": "VLAN"}),
    ]

    print("\nBuilding knowledge base...")
    for doc_id, content, metadata in docs:
        rag.add_document(doc_id, content, metadata)
        print(f"  ✓ {doc_id}")

    # Query with answer generation
    query = "What routing protocols should I use and where?"
    print(f"\nQuery: {query}")

    # Retrieve relevant documents
    results = rag.retrieve(query, strategy=RetrievalStrategy.HYBRID, top_k=3)

    print(f"\nRetrieved {len(results)} documents")

    # Generate answer
    print("\nGenerating answer using retrieved context...")
    answer = rag.generate_answer(query, results)

    print(f"\n{'='*80}")
    print(f"ANSWER")
    print(f"{'='*80}")
    print(f"\nQuestion: {answer['query']}")
    print(f"\nAnswer: {answer['answer']}")
    print(f"\nConfidence: {answer['confidence']}")

    print(f"\nSources Used:")
    for source in answer['sources_used']:
        print(f"  - {source}")

    print(f"\nAdditional Context:")
    print(f"  {answer['additional_context']}")

    print(f"\nRetrieval Details:")
    for i, result in enumerate(answer['retrieval_results'], 1):
        print(f"  {i}. {result['doc_id']} (score: {result['score']:.3f}, method: {result['method']})")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nChapter 30: Graph RAG - Hybrid Retrieval")
    print("Combining knowledge graphs with vector embeddings")
    print("Author: Eduard Dulharu (@eduardd76)\n")

    # Run all examples
    example_1_basic_graph_rag()
    example_2_vector_retrieval()
    example_3_graph_retrieval()
    example_4_hybrid_retrieval()
    example_5_full_rag_pipeline()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
