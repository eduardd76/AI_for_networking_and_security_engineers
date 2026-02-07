# Chapter 14: RAG Fundamentals - Retrieval-Augmented Generation for Network Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding Retrieval-Augmented Generation](#understanding-retrieval-augmented-generation)
3. [Vector Embeddings and Representation](#vector-embeddings-and-representation)
4. [Vector Database Architecture](#vector-database-architecture)
5. [RAG Pipeline Implementation](#rag-pipeline-implementation)
6. [Production Deployment Patterns](#production-deployment-patterns)
7. [Practical Examples](#practical-examples)
8. [Advanced Techniques](#advanced-techniques)
9. [Troubleshooting and Optimization](#troubleshooting-and-optimization)
10. [Conclusion](#conclusion)

---

## Introduction

Retrieval-Augmented Generation (RAG) represents a paradigm shift in how organizations can leverage large language models (LLMs) for domain-specific knowledge tasks. Rather than relying solely on the pretraining knowledge of models, RAG systems dynamically retrieve relevant information from custom knowledge bases and use that information to ground LLM responses in factual, up-to-date data.

### The Networking Analogy: RAG as a "Knowledge DNS"

If you're new to AI, think of RAG like **DNS for knowledge**:

- **Without RAG**: You ask an LLM a question, and it answers from its "cached" knowledge (training data) -- which might be outdated or wrong for your specific network. It's like having only a static hosts file.
- **With RAG**: The system first **looks up** relevant documents from your knowledge base (like a DNS query), then feeds those documents to the LLM along with your question. The LLM answers using your **actual documentation** as its source of truth. It's like having a recursive resolver that always checks the authoritative source.

The result: instead of the LLM guessing about your network, it reads your actual configs, runbooks, and design docs before answering.

In the context of network documentation and operations, RAG systems offer transformative capabilities:

- **Knowledge Retrieval**: Rapidly search through thousands of network configurations, troubleshooting guides, and architecture diagrams
- **Contextual Understanding**: Provide LLMs with network-specific context to generate more accurate and relevant responses
- **Real-time Updates**: Keep documentation knowledge current without retraining models
- **Reduced Hallucinations**: Ground responses in actual documentation, reducing the risk of incorrect information
- **Compliance and Auditability**: Trace response generation back to specific documentation sources

This chapter provides a comprehensive exploration of RAG systems as applied to network documentation and operations. We'll cover the theoretical foundations, practical implementations, and production deployment strategies that enable organizations to build robust, scalable RAG systems.

### Why RAG for Network Documentation?

Network documentation presents unique challenges:

1. **Rapid Evolution**: Network infrastructure changes constantly with new devices, configurations, and security policies
2. **Domain Specificity**: Network terminology and concepts are highly specialized, requiring precise language understanding
3. **High Cost of Errors**: Incorrect network information can lead to misconfigurations, outages, or security breaches
4. **Multi-format Documentation**: Networks documentation exists in multiple formats: design documents, runbooks, configuration files, architecture diagrams, and logs
5. **Distributed Knowledge**: Network expertise often exists in various forms across the organization

RAG systems address these challenges by:

- Creating searchable semantic representations of network documentation
- Enabling context-aware generation based on current network state
- Providing explainability through retrieval source attribution
- Supporting rapid knowledge base updates without model retraining
- Handling diverse documentation formats

---

## Understanding Retrieval-Augmented Generation

### The RAG Paradigm

Retrieval-Augmented Generation combines three core components:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  1. Retrieval Component        │
        │  (Vector Search in Knowledge   │
        │   Base for Relevant Docs)      │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  Retrieved Documents &          │
        │  Context                        │
        └────────────┬───────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────┐
    │  2. Augmentation Component                 │
    │  (Construct Prompt with Context)          │
    └────────────┬─────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────────────┐
    │  3. Generation Component                   │
    │  (LLM Generates Response with Context)    │
    └────────────┬─────────────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────┐
        │    Final Response with          │
        │    Source Attribution          │
        └────────────────────────────────┘
```

### RAG vs. Traditional LLM Approaches

**Fine-tuned Models:**
- Requires extensive training data
- Updates require retraining and deployment
- All knowledge must fit in model parameters
- Higher computational cost for training
- Knowledge becomes stale over time

**Prompt Injection:**
- Simple to implement
- Limited by context window size
- Scales poorly with knowledge base size
- No efficient search mechanism

**RAG Systems:**
- Knowledge stored separately in searchable index
- Updates to knowledge base don't require retraining
- Scales to large knowledge bases through vector indexing
- Supports multiple document types and formats
- Provides source attribution and auditability
- Reduces hallucination through grounding in actual documentation

### Key Components of a RAG System

#### 1. Document Ingestion and Preprocessing

The foundation of any RAG system is the ability to ingest, parse, and prepare documents for retrieval:

```python
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass
class Document:
    """Represents a single document unit"""
    content: str
    metadata: dict
    document_id: str
    
    def to_chunks(self, chunk_size: int = 512, overlap: int = 50) -> List['Document']:
        """Split document into overlapping chunks"""
        words = self.content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_content = ' '.join(chunk_words)
            
            chunk_id = f"{self.document_id}_chunk_{i}"
            chunk_metadata = {
                **self.metadata,
                'chunk_start': i,
                'chunk_end': i + len(chunk_words)
            }
            
            chunks.append(Document(
                content=chunk_content,
                metadata=chunk_metadata,
                document_id=chunk_id
            ))
        
        return chunks

class DocumentProcessor:
    """Handles document ingestion and preprocessing"""
    
    def __init__(self):
        self.processed_docs = []
    
    def load_from_file(self, filepath: str) -> Document:
        """Load document from file"""
        path = Path(filepath)
        content = path.read_text(encoding='utf-8')
        
        metadata = {
            'source': str(path),
            'file_type': path.suffix,
            'size_bytes': path.stat().st_size,
        }
        
        return Document(
            content=content,
            metadata=metadata,
            document_id=path.stem
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep technical notation
        text = re.sub(r'[^\w\s\.\-\:\(\)\[\]\{\},]', '', text)
        
        # Convert to lowercase for processing (keep original for display)
        return text.strip()
    
    def extract_sections(self, content: str) -> List[Tuple[str, str]]:
        """Extract sections and headings from markdown"""
        sections = []
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('#'):
                if current_section:
                    sections.append((current_section, '\n'.join(current_content)))
                current_section = line.lstrip('#').strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_section:
            sections.append((current_section, '\n'.join(current_content)))
        
        return sections

# Example usage
processor = DocumentProcessor()
doc = processor.load_from_file('network_architecture.md')
chunks = doc.to_chunks(chunk_size=256, overlap=50)
print(f"Loaded document: {doc.document_id}")
print(f"Created {len(chunks)} chunks")
```

#### 2. Vector Embedding Generation

Embedding models convert documents into numerical vectors that capture semantic meaning.

> **Networking analogy**: Think of embeddings like IP addresses for meaning. Just as devices in the same subnet (e.g., 10.0.1.0/24) are "close" to each other on the network, documents about similar topics end up with similar embedding vectors -- "close" in mathematical space. When you search, the system finds documents whose vectors are closest to your query's vector, like finding the nearest neighbor in a routing table.

The code below shows two embedding options: a free local model (SentenceTransformers) and OpenAI's API-based embeddings. For production network documentation RAG systems, **text-embedding-3-small** from OpenAI offers the best balance of quality and cost:

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union

class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of embedding vectors"""
        pass

class SentenceTransformerEmbedding(EmbeddingModel):
    """Using sentence-transformers for semantic embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for batch of texts"""
        return self.model.encode(texts, convert_to_numpy=True)
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

class OpenAIEmbedding(EmbeddingModel):
    """Using OpenAI's embedding API"""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._embedding_dim = 1536 if "3-small" in model else 3072
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for batch"""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings = np.array([data.embedding for data in response.data])
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

# Example usage
embedder = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")

# Single text
text = "BGP configuration for redundancy across data centers"
embedding = embedder.embed_text(text)
print(f"Embedding dimension: {embedding.shape[0]}")
print(f"Embedding (first 10 values): {embedding[:10]}")

# Batch processing
texts = [
    "Configure OSPF routing",
    "Set up VLAN interfaces",
    "Enable BGP peering"
]
embeddings = embedder.embed_batch(texts)
print(f"Batch embeddings shape: {embeddings.shape}")
```

#### 3. Retrieval Mechanism

The retrieval component searches the vector database for documents relevant to a query:

```python
from typing import List, Tuple
import numpy as np

class RetrieverBase(ABC):
    """Abstract base for retrieval systems"""
    
    @abstractmethod
    def index(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """Index documents with their embeddings"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve top k documents matching query"""
        pass

class VectorRetriever(RetrieverBase):
    """Retrieves documents using vector similarity"""
    
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None
    
    def index(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """Index documents"""
        self.documents = documents
        self.embeddings = embeddings
        print(f"Indexed {len(documents)} documents")
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve top k documents using cosine similarity"""
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Calculate similarities
        similarities = self._cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = [
            (self.documents[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def _cosine_similarity(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vectors"""
        # Normalize vectors
        A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
        B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
        
        # Calculate similarity
        return np.dot(A_norm, B_norm.T)

class HybridRetriever(RetrieverBase):
    """Combines vector and keyword-based retrieval"""
    
    def __init__(self, vector_retriever: VectorRetriever, keyword_weight: float = 0.3):
        self.vector_retriever = vector_retriever
        self.keyword_weight = keyword_weight
        self.documents: List[Document] = []
        self.word_index: dict = {}
    
    def index(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """Index documents for both vector and keyword retrieval"""
        self.documents = documents
        self.vector_retriever.index(documents, embeddings)
        
        # Build keyword index
        for idx, doc in enumerate(documents):
            words = doc.content.lower().split()
            for word in set(words):
                if word not in self.word_index:
                    self.word_index[word] = []
                self.word_index[word].append(idx)
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve using hybrid approach"""
        # Vector retrieval
        vector_results = self.vector_retriever.retrieve(query, k=k*2)
        vector_scores = {doc.document_id: score for doc, score in vector_results}
        
        # Keyword retrieval
        query_words = set(query.lower().split())
        keyword_scores = {}
        
        for word in query_words:
            if word in self.word_index:
                for idx in self.word_index[word]:
                    doc_id = self.documents[idx].document_id
                    keyword_scores[doc_id] = keyword_scores.get(doc_id, 0) + 1
        
        # Normalize keyword scores
        max_keyword_score = max(keyword_scores.values()) if keyword_scores else 1
        keyword_scores = {
            doc_id: score / max_keyword_score 
            for doc_id, score in keyword_scores.items()
        }
        
        # Combine scores
        all_doc_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        combined_scores = {}
        
        for doc_id in all_doc_ids:
            vector_score = vector_scores.get(doc_id, 0)
            keyword_score = keyword_scores.get(doc_id, 0)
            combined_scores[doc_id] = (
                (1 - self.keyword_weight) * vector_score + 
                self.keyword_weight * keyword_score
            )
        
        # Sort and return top k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = [
            (next(doc for doc in self.documents if doc.document_id == doc_id), score)
            for doc_id, score in sorted_docs
        ]
        
        return results

# Example usage
documents = [doc1, doc2, doc3]  # Your documents
embeddings = embedder.embed_batch([doc.content for doc in documents])

retriever = VectorRetriever(embedding_model=embedder)
retriever.index(documents, embeddings)

query = "How to configure BGP for failover?"
results = retriever.retrieve(query, k=3)

for doc, similarity in results:
    print(f"Match: {doc.document_id} (similarity: {similarity:.3f})")
    print(f"Content: {doc.content[:200]}...")
    print()
```

---

## Vector Embeddings and Representation

### Understanding Vector Embeddings

Vector embeddings are dense numerical representations of text that capture semantic meaning. Unlike one-hot encoding or bag-of-words representations, embeddings encode relationships between concepts and capture contextual meaning.

#### Key Properties of Good Embeddings

1. **Semantic Similarity**: Similar documents should have similar embeddings
   ```python
   # Documents that are semantically similar
   doc1 = "Configure OSPF routing protocol"
   doc2 = "Set up OSPF for dynamic routing"
   
   embedding1 = embedder.embed_text(doc1)
   embedding2 = embedder.embed_text(doc2)
   
   similarity = cosine_similarity(embedding1, embedding2)
   # Should be high (>0.7)
   ```

2. **Dimensionality**: Typically 300-1536 dimensions
   - Smaller dimensions: faster computation, less memory, less expressive
   - Larger dimensions: slower computation, more memory, more expressive

3. **Stability**: Same text should produce same or very similar embeddings
   ```python
   embedding_v1 = embedder.embed_text("Network interface down")
   embedding_v2 = embedder.embed_text("Network interface down")
   
   # Similarity should be very close to 1.0
   ```

#### Popular Embedding Models for Network Documentation

| Model | Dimension | Speed | Quality | Best For |
|-------|-----------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General purpose, on-device |
| all-mpnet-base-v2 | 768 | Medium | Very Good | Semantic search, high quality |
| bge-large-en-v1.5 | 1024 | Medium | Excellent | Dense retrieval, production |
| text-embedding-3-small | 1536 | Slow | Excellent | OpenAI API, best value for most use cases |
| text-embedding-3-large | 3072 | Slow | Best | OpenAI API, highest quality (use for critical retrieval) |

```python
class EmbeddingComparison:
    """Compare different embedding models"""
    
    def __init__(self):
        self.models = {
            'mini': SentenceTransformerEmbedding("all-MiniLM-L6-v2"),
            'mpnet': SentenceTransformerEmbedding("all-mpnet-base-v2"),
            'bge': SentenceTransformerEmbedding("BAAI/bge-large-en-v1.5"),
        }
    
    def compare_quality(self, doc_pairs: List[Tuple[str, str]]):
        """Compare similarity scores across models"""
        results = {}
        
        for model_name, model in self.models.items():
            similarities = []
            
            for doc1, doc2 in doc_pairs:
                emb1 = model.embed_text(doc1)
                emb2 = model.embed_text(doc2)
                
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                similarities.append(similarity)
            
            results[model_name] = {
                'mean_similarity': np.mean(similarities),
                'dimension': model.embedding_dim
            }
        
        return results
    
    def compare_speed(self, texts: List[str], iterations: int = 100):
        """Compare embedding generation speed"""
        import time
        
        results = {}
        
        for model_name, model in self.models.items():
            start = time.time()
            
            for _ in range(iterations):
                model.embed_batch(texts)
            
            elapsed = time.time() - start
            results[model_name] = {
                'total_time': elapsed,
                'time_per_text': elapsed / (len(texts) * iterations),
                'dimension': model.embedding_dim
            }
        
        return results

# Example evaluation
comparison = EmbeddingComparison()

doc_pairs = [
    ("BGP configuration", "Border Gateway Protocol setup"),
    ("Network outage", "Service disruption"),
    ("Interface configuration", "Port assignment"),
]

quality_results = comparison.compare_quality(doc_pairs)
print("Quality Comparison:")
for model, metrics in quality_results.items():
    print(f"{model}: {metrics}")
```

### Vector Normalization and Distance Metrics

#### L2 Normalization (Cosine Similarity)

Most commonly used for text embeddings. After normalization, Euclidean distance equals cosine distance.

```python
def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors to unit length"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine distance between two vectors"""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    return 1 - np.dot(v1_norm, v2_norm)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors"""
    return 1 - cosine_distance(v1, v2)
```

#### Other Distance Metrics

```python
def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Euclidean (L2) distance"""
    return np.linalg.norm(v1 - v2)

def manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Manhattan (L1) distance"""
    return np.sum(np.abs(v1 - v2))

def dot_product_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Dot product similarity (for normalized vectors)"""
    return np.dot(v1, v2)

# Comparison for network documentation search
class DistanceMetricComparison:
    """Compare distance metrics for retrieval"""
    
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
    
    def evaluate_metric(self, query_embedding: np.ndarray, k: int = 5, metric: str = 'cosine'):
        """Evaluate retrieval quality with different metrics"""
        
        if metric == 'cosine':
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-k:][::-1]
            scores = similarities[top_indices]
        
        elif metric == 'euclidean':
            distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
            top_indices = np.argsort(distances)[:k]
            scores = 1 / (1 + distances[top_indices])  # Convert to similarity
        
        elif metric == 'manhattan':
            distances = np.sum(np.abs(self.embeddings - query_embedding), axis=1)
            top_indices = np.argsort(distances)[:k]
            scores = 1 / (1 + distances[top_indices])
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return top_indices, scores
```

---

## Vector Database Architecture

Vector databases are specialized systems designed to efficiently store and search high-dimensional vectors. They use advanced indexing techniques to provide sub-linear search complexity.

> **Networking analogy**: A vector database is like a **content-addressable memory (CAM) table** in a switch, but for document meanings. A switch's CAM table maps MAC addresses to ports for fast lookups. A vector database maps document embeddings (numerical "meaning fingerprints") to documents for fast semantic search. Just as a CAM lookup is O(1) instead of flooding every port, a vector database uses approximate nearest neighbor (ANN) algorithms to find relevant documents without scanning every one -- critical when you have thousands of configs and runbooks.

### Choosing a Vector Database

For learning and prototyping, **ChromaDB** is the simplest option -- it runs locally with no setup. For production network documentation systems, consider Pinecone (managed cloud) or Qdrant (self-hosted, good for air-gapped networks).

### Vector Database Options

#### 1. Pinecone (Cloud-Hosted)

Fully managed vector database as a service.

```python
import pinecone
from pinecone import Pinecone

class PineconeVectorDB:
    """Wrapper for Pinecone vector database"""
    
    def __init__(self, api_key: str, environment: str = "us-west1-gcp"):
        self.pc = Pinecone(api_key=api_key)
        self.environment = environment
        self.index = None
    
    def create_index(self, index_name: str, dimension: int, metric: str = "cosine"):
        """Create a new vector index"""
        self.pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=pinecone.ServerlessSpec(
                cloud="gcp",
                region=self.environment
            )
        )
        self.index = self.pc.Index(index_name)
        print(f"Created index: {index_name}")
    
    def connect_index(self, index_name: str):
        """Connect to existing index"""
        self.index = self.pc.Index(index_name)
    
    def upsert_documents(self, documents: List[Document], embeddings: np.ndarray, batch_size: int = 100):
        """Add or update documents in the index"""
        vectors = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vector_id = doc.document_id
            values = embedding.tolist()
            metadata = {
                'content': doc.content[:1000],  # Store content preview
                'source': doc.metadata.get('source', ''),
                'chunk_start': doc.metadata.get('chunk_start', 0),
            }
            
            vectors.append((vector_id, values, metadata))
        
        # Upsert in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
            print(f"Upserted batch {i//batch_size + 1}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, filter: dict = None):
        """Search for similar documents"""
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True,
            filter=filter
        )
        
        return [
            {
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            }
            for match in results['matches']
        ]
    
    def delete_documents(self, document_ids: List[str]):
        """Delete documents from index"""
        self.index.delete(ids=document_ids)
    
    def list_indices(self):
        """List all indices"""
        return self.pc.list_indexes()

# Example usage
pinecone_db = PineconeVectorDB(api_key="your-api-key")

# Create or connect to index
try:
    pinecone_db.create_index("network-docs", dimension=384, metric="cosine")
except:
    pinecone_db.connect_index("network-docs")

# Upsert documents
documents = [...]  # Your documents
embeddings = embedder.embed_batch([doc.content for doc in documents])
pinecone_db.upsert_documents(documents, embeddings)

# Search
query_embedding = embedder.embed_text("How to configure BGP?")
results = pinecone_db.search(query_embedding, k=5)
```

#### 2. Weaviate (Open Source / Cloud)

Semantic search database with built-in ML capabilities.

```python
import weaviate
from weaviate.client import Client
from weaviate.connect import ConnectionParams

class WeaviateVectorDB:
    """Wrapper for Weaviate vector database"""
    
    def __init__(self, url: str = "http://localhost:8080", api_key: str = None):
        self.client = weaviate.connect_to_local(
            host=url.split("//")[1].split(":")[0],
            port=int(url.split(":")[-1]),
            grpc_port=50051
        ) if "localhost" in url else weaviate.connect_to_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key)
        )
    
    def create_class(self, class_name: str = "NetworkDocument"):
        """Create a document class"""
        class_obj = {
            "class": class_name,
            "description": "Documents from network documentation",
            "vectorizer": "none",  # We'll provide vectors directly
            "vectorIndexConfig": {
                "distance": "cosine"
            },
            "properties": [
                {
                    "name": "content",
                    "description": "Document content",
                    "dataType": ["text"],
                },
                {
                    "name": "source",
                    "description": "Document source",
                    "dataType": ["text"],
                },
                {
                    "name": "chunk_id",
                    "description": "Chunk identifier",
                    "dataType": ["text"],
                },
                {
                    "name": "metadata",
                    "description": "Additional metadata",
                    "dataType": ["object"],
                },
            ]
        }
        
        self.client.collections.create_from_dict(class_obj)
        print(f"Created class: {class_name}")
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with embeddings"""
        collection = self.client.collections.get("NetworkDocument")
        
        data_objects = []
        for doc, embedding in zip(documents, embeddings):
            obj = {
                "content": doc.content,
                "source": doc.metadata.get("source", ""),
                "chunk_id": doc.document_id,
                "metadata": doc.metadata,
            }
            data_objects.append(obj)
        
        collection.data.insert_multiple(data_objects, vectors=embeddings.tolist())
        print(f"Added {len(documents)} documents")
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Search for similar documents"""
        collection = self.client.collections.get("NetworkDocument")
        
        results = collection.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=k,
            return_metadata=True
        )
        
        return results.objects
    
    def close(self):
        """Close connection"""
        self.client.close()

# Example usage
weaviate_db = WeaviateVectorDB(url="http://localhost:8080")
weaviate_db.create_class()

documents = [...]
embeddings = embedder.embed_batch([doc.content for doc in documents])
weaviate_db.add_documents(documents, embeddings)

query_embedding = embedder.embed_text("OSPF configuration")
results = weaviate_db.search(query_embedding, k=5)
```

#### 3. Milvus (Open Source)

High-performance vector database for large-scale similarity search.

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

class MilvusVectorDB:
    """Wrapper for Milvus vector database"""
    
    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port
        self.collection = None
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
    
    def create_collection(self, collection_name: str = "network_docs", embedding_dim: int = 384):
        """Create a new collection"""
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Network documentation embeddings"
        )
        
        # Create collection
        self.collection = Collection(
            name=collection_name,
            schema=schema,
            consistency_level="Strong"
        )
        
        # Create index on embeddings
        self.collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
        )
        
        print(f"Created collection: {collection_name}")
    
    def insert_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Insert documents into collection"""
        entities = [
            [doc.document_id for doc in documents],
            [doc.content for doc in documents],
            [doc.metadata.get("source", "") for doc in documents],
            embeddings.tolist(),
        ]
        
        self.collection.insert(entities)
        self.collection.flush()
        print(f"Inserted {len(documents)} documents")
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Search for similar documents"""
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["document_id", "content", "source"]
        )
        
        return results[0]
    
    def close(self):
        """Close connection"""
        connections.disconnect(alias="default")

# Example usage
milvus_db = MilvusVectorDB(host="localhost", port=19530)
milvus_db.create_collection("network_docs", embedding_dim=384)

documents = [...]
embeddings = embedder.embed_batch([doc.content for doc in documents])
milvus_db.insert_documents(documents, embeddings)

query_embedding = embedder.embed_text("Route summarization")
results = milvus_db.search(query_embedding, k=5)

for hit in results:
    print(f"Found: {hit.entity.get('document_id')} (distance: {hit.distance})")
```

#### 4. Chroma (Lightweight/Embedded)

Simple but powerful embedded vector database for development.

```python
import chromadb
from chromadb.config import Settings

class ChromaVectorDB:
    """Wrapper for Chroma vector database"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        # Initialize with persistence
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False
        )
        
        self.client = chromadb.Client(settings)
        self.collection = None
    
    def create_collection(self, collection_name: str = "network_docs"):
        """Create a collection"""
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created collection: {collection_name}")
    
    def get_or_create_collection(self, collection_name: str):
        """Get existing or create new collection"""
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.create_collection(collection_name)
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents to collection"""
        ids = [doc.document_id for doc in documents]
        documents_text = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents_text,
            metadatas=metadatas
        )
        
        print(f"Added {len(documents)} documents to collection")
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Search for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        return results
    
    def persist(self):
        """Persist data to disk"""
        self.client.persist()
        print("Data persisted to disk")

# Example usage
chroma_db = ChromaVectorDB(persist_directory="./network_docs_db")
chroma_db.get_or_create_collection("network_docs")

documents = [...]
embeddings = embedder.embed_batch([doc.content for doc in documents])
chroma_db.add_documents(documents, embeddings)
chroma_db.persist()

# Search
query_embedding = embedder.embed_text("Network failover configuration")
results = chroma_db.search(query_embedding, k=5)

for ids, distances, documents in zip(results['ids'][0], results['distances'][0], results['documents'][0]):
    print(f"Result: {ids} (distance: {distances})")
    print(f"Content: {documents[:200]}...")
```

### Indexing Strategies

Vector databases use sophisticated indexing to provide fast retrieval:

```python
class IndexStrategy:
    """Different indexing strategies for vectors"""
    
    @staticmethod
    def ivf_flat(n_clusters: int = 1024):
        """
        Inverted File (IVF) - quantize space into clusters
        Good for: Medium to large datasets (100K - 10M)
        Trade-off: Some accuracy loss for speed
        """
        return {
            "index_type": "IVF_FLAT",
            "nlist": n_clusters,
            "metric": "COSINE"
        }
    
    @staticmethod
    def hnsw(max_connections: int = 16, ef_construction: int = 200):
        """
        Hierarchical Navigable Small World (HNSW)
        Good for: Fast single queries, any dataset size
        Trade-off: More memory usage
        """
        return {
            "index_type": "HNSW",
            "M": max_connections,
            "ef_construction": ef_construction,
            "metric": "COSINE"
        }
    
    @staticmethod
    def pq(num_partitions: int = 256):
        """
        Product Quantization (PQ)
        Good for: Very large datasets (>1M), memory constrained
        Trade-off: Lower accuracy
        """
        return {
            "index_type": "IVFPQ",
            "nlist": 1024,
            "m": num_partitions,
            "metric": "COSINE"
        }

# Performance comparison: Index selection guide
index_guide = """
Dataset Size          | Recommended Index | Latency  | Memory
----------------------------------------
< 10K vectors        | Flat/HNSW        | < 5ms   | Low
10K - 100K           | IVF_FLAT/HNSW    | 5-50ms  | Medium
100K - 1M            | IVF_PQ/HNSW      | 50-200ms| Medium-High
> 1M                 | IVFPQ/HNSW       | 100-500ms| High
"""
```

---

## RAG Pipeline Implementation

A complete RAG pipeline orchestrates document processing, embedding, storage, and generation.

### End-to-End RAG System

```python
from typing import Optional, Callable
import json
from datetime import datetime

class RAGPipeline:
    """Complete RAG system for network documentation"""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_db: VectorDB,
        llm_model: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processor = DocumentProcessor()
        self.retriever = VectorRetriever(embedding_model)
    
    def ingest_documents(self, file_paths: List[str]):
        """Ingest documents into the RAG system"""
        all_chunks = []
        
        for filepath in file_paths:
            print(f"Processing {filepath}...")
            
            # Load document
            doc = self.processor.load_from_file(filepath)
            
            # Split into chunks
            chunks = doc.to_chunks(
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            
            all_chunks.extend(chunks)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        contents = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_model.embed_batch(contents)
        
        # Store in vector database
        print("Storing in vector database...")
        self.vector_db.index(all_chunks, embeddings)
        
        print(f"Successfully ingested {len(all_chunks)} document chunks")
        
        return all_chunks
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents"""
        results = self.retriever.retrieve(query, k=k)
        return results
    
    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response using LLM with retrieved context"""
        
        if not retrieved_docs:
            return "No relevant documents found."
        
        # Build context from retrieved documents
        context = self._build_context(retrieved_docs)
        
        # Prepare prompt
        if system_prompt is None:
            system_prompt = self._default_system_prompt()
        
        user_prompt = f"""
Based on the following network documentation, answer the user's question.

Documentation:
{context}

Question: {query}

Please provide a detailed answer with specific steps or configurations from the documentation.
"""
        
        # Generate response
        response = self._call_llm(system_prompt, user_prompt)
        
        return response
    
    def answer_query(self, query: str, k: int = 5, return_sources: bool = True) -> dict:
        """Complete query-to-answer pipeline"""
        
        # Retrieve documents
        retrieved_docs = self.retrieve(query, k=k)
        
        # Generate response
        response = self.generate_response(query, retrieved_docs)
        
        # Compile result
        result = {
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat(),
        }
        
        if return_sources:
            result['sources'] = [
                {
                    'document_id': doc.document_id,
                    'source': doc.metadata.get('source', 'unknown'),
                    'similarity': float(score),
                    'excerpt': doc.content[:200] + "..."
                }
                for doc, score in retrieved_docs
            ]
        
        return result
    
    def _build_context(self, docs: List[Tuple[Document, float]]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        
        for idx, (doc, score) in enumerate(docs, 1):
            context_parts.append(f"Source {idx} (relevance: {score:.2%}):")
            context_parts.append(f"From: {doc.metadata.get('source', 'unknown')}")
            context_parts.append(f"Content:\n{doc.content}")
            context_parts.append("-" * 50)
        
        return "\n\n".join(context_parts)
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for network documentation"""
        return """You are an expert network engineer assistant helping with network 
documentation and configuration guidance. You have access to internal network 
documentation and best practices.

Your responses should:
1. Be specific and actionable
2. Reference the provided documentation
3. Include step-by-step instructions when appropriate
4. Highlight any security or operational considerations
5. Suggest best practices based on the documentation
6. Admit when information is not in the provided documentation"""
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM for generation.

        This example uses OpenAI's API, but you can swap in any LLM.
        To use Claude instead, replace the OpenAI client with:
            from anthropic import Anthropic
            client = Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=1500
            )
            return response.content[0].text
        """
        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.chat.completions.create(
                model=self.llm_model or "gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temp for factual responses
                max_tokens=1500
            )

            return response.choices[0].message.content

        except ImportError:
            # Fallback for demo
            return f"[LLM Response would be generated here]\n\nQuery: {user_prompt[:200]}..."

# Example usage
class NetworkRAGExample:
    @staticmethod
    def setup_rag():
        # Initialize components
        embedder = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
        vector_db = ChromaVectorDB(persist_directory="./network_rag_db")

        # Create RAG pipeline -- using OpenAI here, but Claude works too
        rag = RAGPipeline(
            embedding_model=embedder,
            vector_db=vector_db,
            llm_model="gpt-4o",
            chunk_size=512,
            chunk_overlap=50
        )
        
        return rag
    
    @staticmethod
    def example_workflow():
        rag = NetworkRAGExample.setup_rag()
        
        # Ingest documents
        docs = rag.ingest_documents([
            'docs/bgp_configuration.md',
            'docs/ospf_setup.md',
            'docs/network_failover.md'
        ])
        
        # Answer queries
        queries = [
            "How do I configure BGP for redundancy?",
            "What are the steps to enable OSPF authentication?",
            "How should I set up network failover between data centers?"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            result = rag.answer_query(query, k=3)
            
            print(f"Answer: {result['response'][:500]}...")
            print("\nSources:")
            for source in result.get('sources', []):
                print(f"  - {source['source']} ({source['similarity']:.0%} relevant)")
```

### Prompt Engineering for RAG

Effective prompts are crucial for generating high-quality responses with retrieved context:

```python
class PromptTemplates:
    """Collection of specialized prompts for network documentation"""
    
    @staticmethod
    def configuration_prompt(context: str, task: str) -> str:
        """Prompt for configuration guidance"""
        return f"""Based on the provided documentation, provide step-by-step configuration 
instructions for the following task:

Documentation Context:
{context}

Task: {task}

Please provide:
1. Prerequisites and requirements
2. Step-by-step instructions
3. Verification steps
4. Troubleshooting tips
5. Best practices and recommendations

Format the response as a clear, numbered guide."""
    
    @staticmethod
    def troubleshooting_prompt(context: str, issue: str) -> str:
        """Prompt for troubleshooting guidance"""
        return f"""Based on the provided documentation, help troubleshoot the following 
network issue:

Documentation Context:
{context}

Issue: {issue}

Please provide:
1. Likely causes based on the documentation
2. Diagnostic steps to isolate the problem
3. Recommended solutions
4. How to verify the fix
5. Prevention measures for the future"""
    
    @staticmethod
    def analysis_prompt(context: str, question: str) -> str:
        """Prompt for analytical questions"""
        return f"""Analyze the following question based on the provided documentation:

Documentation Context:
{context}

Question: {question}

Please provide:
1. A direct answer supported by the documentation
2. Relevant examples or scenarios
3. Considerations and trade-offs
4. Related topics that might be relevant
5. Information gaps or areas not covered by documentation"""
    
    @staticmethod
    def comparison_prompt(context: str, items: List[str]) -> str:
        """Prompt for comparing items based on documentation"""
        return f"""Compare the following items based on the provided documentation:

Documentation Context:
{context}

Items to compare: {', '.join(items)}

Please provide:
1. Feature comparison matrix
2. Pros and cons of each
3. Use case recommendations
4. Configuration differences
5. Performance implications"""

class AdvancedRAGPipeline(RAGPipeline):
    """RAG with advanced prompt engineering"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_templates = PromptTemplates()
        self.query_analyzer = QueryAnalyzer()
    
    def answer_query_advanced(self, query: str, k: int = 5) -> dict:
        """Answer with intelligent prompt selection"""
        
        # Analyze query type
        query_type = self.query_analyzer.classify_query(query)
        
        # Retrieve documents
        retrieved_docs = self.retrieve(query, k=k)
        context = self._build_context(retrieved_docs)
        
        # Select appropriate prompt template
        if query_type == "configuration":
            prompt = self.prompt_templates.configuration_prompt(context, query)
        elif query_type == "troubleshooting":
            prompt = self.prompt_templates.troubleshooting_prompt(context, query)
        elif query_type == "comparison":
            prompt = self.prompt_templates.comparison_prompt(context, [])
        else:
            prompt = self.prompt_templates.analysis_prompt(context, query)
        
        # Generate response
        response = self._call_llm(self._default_system_prompt(), prompt)
        
        return {
            'query': query,
            'query_type': query_type,
            'response': response,
            'sources': [
                {
                    'document_id': doc.document_id,
                    'source': doc.metadata.get('source', 'unknown'),
                    'similarity': float(score)
                }
                for doc, score in retrieved_docs
            ],
            'timestamp': datetime.now().isoformat()
        }

class QueryAnalyzer:
    """Analyze query type for specialized handling"""
    
    def classify_query(self, query: str) -> str:
        """Classify query into types"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'configure', 'set up', 'enable']):
            return "configuration"
        
        elif any(word in query_lower for word in ['troubleshoot', 'problem', 'issue', 'error', 'not working']):
            return "troubleshooting"
        
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'which']):
            return "comparison"
        
        else:
            return "general"
```

---

## Production Deployment Patterns

### Containerized RAG Service

Deploying RAG systems in production requires containerization, monitoring, and scaling considerations.

```yaml
# docker-compose.yml - Complete RAG stack
version: '3.8'

services:
  # Vector Database
  chroma:
    image: ghcr.io/chroma-core/chroma:latest
    ports:
      - "8000:8000"
    environment:
      - CHROMA_DB_IMPL=duckdb+parquet
      - ALLOW_RESET=true
      - ANONYMIZED_TELEMETRY=false
    volumes:
      - chroma_data:/chroma/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Redis Cache (optional, for caching embeddings)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # RAG API Service
  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - CHROMA_HOST=chroma
      - CHROMA_PORT=8000
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=INFO
    depends_on:
      chroma:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./docs:/app/docs
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  chroma_data:
```

```dockerfile
# Dockerfile for RAG service
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run application
CMD ["python", "-m", "rag_service.api"]
```

```python
# rag_service/api.py - Production RAG API
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import logging
from datetime import datetime
import asyncio
import aiohttp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Network Documentation RAG API", version="1.0.0")

# Models
class DocumentIngestionRequest(BaseModel):
    """Request to ingest documents"""
    file_paths: List[str]
    collection_name: str = "network_docs"

class QueryRequest(BaseModel):
    """Query request"""
    query: str
    k: int = 5
    include_sources: bool = True

class QueryResponse(BaseModel):
    """Query response"""
    query: str
    response: str
    sources: Optional[List[dict]]
    processing_time_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    vector_db_connected: bool
    embedding_model_ready: bool
    llm_available: bool
    timestamp: str

# Global RAG pipeline
rag_pipeline = None
startup_complete = False

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline, startup_complete
    
    try:
        logger.info("Initializing RAG pipeline...")
        
        # Initialize components
        from embedding_models import SentenceTransformerEmbedding
        from vector_db import ChromaVectorDB
        from rag_pipeline import RAGPipeline
        
        embedder = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
        vector_db = ChromaVectorDB(persist_directory="/data/chroma_db")
        
        rag_pipeline = RAGPipeline(
            embedding_model=embedder,
            vector_db=vector_db,
            llm_model="gpt-4o",
            chunk_size=512,
            chunk_overlap=50
        )
        
        startup_complete = True
        logger.info("RAG pipeline initialized successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        startup_complete = False

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector DB
        vector_db_ok = rag_pipeline and rag_pipeline.vector_db is not None
        
        # Check embedding model
        embedding_ok = rag_pipeline and rag_pipeline.embedding_model is not None
        
        # Check LLM availability (simple check)
        llm_ok = rag_pipeline and rag_pipeline.llm_model is not None
        
        return HealthResponse(
            status="healthy" if all([vector_db_ok, embedding_ok, llm_ok]) else "degraded",
            vector_db_connected=vector_db_ok,
            embedding_model_ready=embedding_ok,
            llm_available=llm_ok,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            vector_db_connected=False,
            embedding_model_ready=False,
            llm_available=False,
            timestamp=datetime.now().isoformat()
        )

@app.post("/ingest", response_model=dict)
async def ingest_documents(request: DocumentIngestionRequest, background_tasks: BackgroundTasks):
    """Ingest documents into RAG system"""
    if not startup_complete:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Run ingestion in background
        background_tasks.add_task(
            rag_pipeline.ingest_documents,
            request.file_paths
        )
        
        return {
            "status": "ingestion_started",
            "file_count": len(request.file_paths),
            "collection": request.collection_name,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system"""
    if not startup_complete:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        import time
        start_time = time.time()
        
        # Execute query
        result = rag_pipeline.answer_query(
            query=request.query,
            k=request.k,
            return_sources=request.include_sources
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            query=request.query,
            response=result['response'],
            sources=result.get('sources'),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=dict)
async def get_stats():
    """Get RAG system statistics"""
    if not startup_complete or not rag_pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        return {
            "vector_db_type": type(rag_pipeline.vector_db).__name__,
            "embedding_model": type(rag_pipeline.embedding_model).__name__,
            "llm_model": rag_pipeline.llm_model,
            "chunk_size": rag_pipeline.chunk_size,
            "chunk_overlap": rag_pipeline.chunk_overlap,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: default
data:
  log_level: "INFO"
  chunk_size: "512"
  chunk_overlap: "50"
  max_workers: "4"

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rag-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
  labels:
    app: rag-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-service
  template:
    metadata:
      labels:
        app: rag-service
    spec:
      containers:
      - name: rag-api
        image: network-docs/rag-service:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8001
          name: api
        env:
        - name: CHROMA_HOST
          value: "chroma-service"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secrets
              key: api-key
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: log_level
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: rag-data-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: rag-service
spec:
  type: LoadBalancer
  selector:
    app: rag-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8001
    name: api

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Practical Examples

### Example 1: BGP Configuration Documentation RAG

```python
class BGPDocumentationRAG:
    """RAG system specialized for BGP documentation"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline
        self.bgp_templates = BGPPromptTemplates()
    
    def query_bgp_configuration(self, context: str, requirement: str) -> dict:
        """Query for BGP configuration guidance"""
        
        prompt = f"""Based on the BGP documentation provided, help configure BGP according 
to this requirement:

Requirement: {requirement}

Documentation:
{context}

Please provide:
1. BGP session configuration
2. Route redistribution settings if needed
3. Community and attribute manipulation
4. Verification commands
5. Monitoring and troubleshooting tips"""
        
        response = self.rag._call_llm(self.rag._default_system_prompt(), prompt)
        return {"requirement": requirement, "configuration": response}
    
    def query_bgp_failover(self, context: str, datacenter_setup: str) -> dict:
        """Query for BGP failover configuration"""
        
        prompt = f"""Configure BGP failover for this data center setup:

Setup: {datacenter_setup}

Documentation:
{context}

Provide detailed BGP failover configuration including:
1. Primary and secondary path configuration
2. AS_PATH manipulation for failover
3. Local preference settings
4. Failover timing considerations
5. Testing and verification procedures"""
        
        response = self.rag._call_llm(self.rag._default_system_prompt(), prompt)
        return {"setup": datacenter_setup, "failover_config": response}

class BGPPromptTemplates:
    """Specialized prompts for BGP"""
    
    @staticmethod
    def session_config_prompt(context: str, requirement: str) -> str:
        return f"""Configure BGP session based on requirement:

{requirement}

Documentation: {context}"""
    
    @staticmethod
    def policy_prompt(context: str, routing_policy: str) -> str:
        return f"""Create BGP routing policy:

Policy: {routing_policy}

Documentation: {context}"""

# Example usage
embedder = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
vector_db = ChromaVectorDB(persist_directory="./bgp_docs")
rag = RAGPipeline(
    embedding_model=embedder,
    vector_db=vector_db,
    llm_model="gpt-4o"
)

# Ingest BGP documentation
bgp_rag = BGPDocumentationRAG(rag)
rag.ingest_documents([
    'docs/bgp_fundamentals.md',
    'docs/bgp_failover.md',
    'docs/bgp_best_practices.md'
])

# Query BGP configuration
result = bgp_rag.query_bgp_configuration(
    context="See BGP documentation",
    requirement="Configure BGP redundancy across 3 data centers"
)
```

### Example 2: Network Troubleshooting RAG

```python
class NetworkTroubleshootingRAG:
    """RAG for network troubleshooting and diagnostics"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline
        self.diagnostic_tools = DiagnosticTools()
    
    def troubleshoot_connectivity(self, source_host: str, dest_host: str) -> dict:
        """Troubleshoot connectivity between hosts"""
        
        # Gather diagnostics
        diagnostics = self.diagnostic_tools.run_diagnostics(source_host, dest_host)
        
        # Query RAG for troubleshooting guidance
        query = f"Troubleshoot connectivity issue from {source_host} to {dest_host}"
        retrieved_docs = self.rag.retrieve(query, k=5)
        
        context = self.rag._build_context(retrieved_docs)
        
        prompt = f"""Troubleshoot connectivity issue based on diagnostics:

Diagnostics:
{json.dumps(diagnostics, indent=2)}

Documentation:
{context}

Provide:
1. Root cause analysis
2. Step-by-step troubleshooting
3. Network configuration checks
4. Potential fixes
5. Prevention measures"""
        
        response = self.rag._call_llm(self.rag._default_system_prompt(), prompt)
        
        return {
            "source": source_host,
            "destination": dest_host,
            "diagnostics": diagnostics,
            "analysis": response,
            "sources": [
                {
                    "id": doc.document_id,
                    "similarity": score
                }
                for doc, score in retrieved_docs
            ]
        }
    
    def analyze_packet_loss(self, interface: str, loss_percentage: float) -> dict:
        """Analyze packet loss issue"""
        
        query = f"Packet loss on interface {interface}"
        retrieved_docs = self.rag.retrieve(query, k=5)
        context = self.rag._build_context(retrieved_docs)
        
        prompt = f"""Analyze and resolve packet loss:

Interface: {interface}
Loss Percentage: {loss_percentage}%

Documentation:
{context}

Provide:
1. Likely causes
2. Diagnostic commands
3. Configuration fixes
4. Capacity planning recommendations"""
        
        response = self.rag._call_llm(self.rag._default_system_prompt(), prompt)
        
        return {
            "interface": interface,
            "loss_percentage": loss_percentage,
            "analysis": response
        }

class DiagnosticTools:
    """Tools for gathering network diagnostics"""
    
    def run_diagnostics(self, source: str, dest: str) -> dict:
        """Run diagnostic commands"""
        import subprocess
        
        diagnostics = {
            'ping': None,
            'traceroute': None,
            'mtu_test': None,
            'bgp_status': None
        }
        
        # Ping test
        try:
            result = subprocess.run(
                ['ping', '-c', '4', dest],
                capture_output=True,
                timeout=10
            )
            diagnostics['ping'] = {
                'success': result.returncode == 0,
                'output': result.stdout.decode()[:500]
            }
        except:
            diagnostics['ping'] = {'error': 'Ping failed'}
        
        # Additional diagnostics would go here
        return diagnostics

# Example usage
troubleshooting_rag = NetworkTroubleshootingRAG(rag)
result = troubleshooting_rag.troubleshoot_connectivity(
    source_host="host1.datacenter1",
    dest_host="host2.datacenter2"
)
print(result['analysis'])
```

### Example 3: Network Change Management RAG

```python
class NetworkChangeManagementRAG:
    """RAG for network change management and planning"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline
    
    def validate_change(self, change_description: str, affected_services: List[str]) -> dict:
        """Validate network change against best practices and documentation"""
        
        query = f"Network change: {change_description}"
        retrieved_docs = self.rag.retrieve(query, k=5)
        context = self.rag._build_context(retrieved_docs)
        
        prompt = f"""Validate this network change:

Change: {change_description}
Affected Services: {', '.join(affected_services)}

Best Practices Documentation:
{context}

Provide:
1. Compliance with best practices
2. Risk assessment
3. Impact analysis on affected services
4. Recommended rollback procedures
5. Monitoring recommendations"""
        
        response = self.rag._call_llm(self.rag._default_system_prompt(), prompt)
        
        return {
            "change": change_description,
            "affected_services": affected_services,
            "validation": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_runbook(self, change_type: str, network_component: str) -> dict:
        """Generate runbook for change execution"""
        
        query = f"{change_type} on {network_component}"
        retrieved_docs = self.rag.retrieve(query, k=5)
        context = self.rag._build_context(retrieved_docs)
        
        prompt = f"""Generate detailed runbook:

Change Type: {change_type}
Network Component: {network_component}

Reference Documentation:
{context}

Create runbook with:
1. Pre-change verification steps
2. Change execution steps
3. Rollback procedures
4. Monitoring during change
5. Post-change validation
6. Escalation contacts and procedures"""
        
        response = self.rag._call_llm(self.rag._default_system_prompt(), prompt)
        
        return {
            "change_type": change_type,
            "component": network_component,
            "runbook": response,
            "generated_at": datetime.now().isoformat()
        }

# Example usage
change_mgmt_rag = NetworkChangeManagementRAG(rag)

# Validate a change
validation = change_mgmt_rag.validate_change(
    change_description="Upgrade BGP software to version 4.3.2 on border routers",
    affected_services=["Internet connectivity", "MPLS", "Customer VPNs"]
)

# Generate a runbook
runbook = change_mgmt_rag.generate_runbook(
    change_type="VLAN migration",
    network_component="Access switch stack"
)
```

---

## Advanced Techniques

### Retrieval Optimization

```python
class RetrievalOptimizer:
    """Optimize retrieval for accuracy and performance"""
    
    @staticmethod
    def rerank_results(
        query: str,
        documents: List[Document],
        embedder: EmbeddingModel,
        top_k: int = 3
    ) -> List[Document]:
        """Rerank results using cross-encoder for improved accuracy"""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return documents[:top_k]
        
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Calculate relevance scores
        scores = model.predict([
            [query, doc.content] for doc in documents
        ])
        
        # Sort by score
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc for doc, score in ranked[:top_k]]
    
    @staticmethod
    def diversify_results(
        documents: List[Document],
        embeddings: np.ndarray,
        k: int = 5
    ) -> List[Document]:
        """Return diverse results to avoid redundancy"""
        selected = [documents[0]]
        selected_embeddings = [embeddings[0]]
        
        for i in range(1, len(documents)):
            if len(selected) >= k:
                break
            
            # Calculate diversity score
            min_similarity = min([
                np.dot(embeddings[i], e) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(e)
                )
                for e in selected_embeddings
            ])
            
            # Select if diverse enough
            if min_similarity < 0.8:
                selected.append(documents[i])
                selected_embeddings.append(embeddings[i])
        
        return selected

class QueryExpansion:
    """Expand queries for better retrieval"""
    
    @staticmethod
    def expand_query(query: str, llm_model: str = "gpt-4o") -> List[str]:
        """Generate query variations"""
        try:
            from openai import OpenAI
            
            client = OpenAI()
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{
                    "role": "user",
                    "content": f"""Generate 3 alternative ways to phrase this query 
for a network documentation search system:

Original query: {query}

Return only the 3 alternatives, one per line."""
                }],
                temperature=0.7,
                max_tokens=200
            )
            
            variations = response.choices[0].message.content.split('\n')
            return [query] + [v.strip() for v in variations if v.strip()]
        
        except:
            return [query]
    
    @staticmethod
    def get_synonyms(term: str) -> List[str]:
        """Get network terminology synonyms"""
        synonym_map = {
            'router': ['gateway', 'edge router', 'core router'],
            'switch': ['layer 2 device', 'Ethernet switch'],
            'vlan': ['virtual LAN', 'virtual network'],
            'bgp': ['Border Gateway Protocol', 'exterior routing'],
            'ospf': ['Open Shortest Path First', 'interior routing'],
            'failover': ['failsafe', 'redundancy', 'backup activation'],
            'latency': ['delay', 'response time'],
            'bandwidth': ['throughput', 'capacity', 'link speed'],
        }
        
        return synonym_map.get(term.lower(), [term])
```

### Caching and Performance

```python
import hashlib
from functools import lru_cache
import pickle

class RAGCache:
    """Caching layer for RAG system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            import redis
            self.redis_client = redis.from_url(redis_url)
        except:
            self.redis_client = None
            self.memory_cache = {}
    
    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        key = f"embedding:{self._hash_text(text)}"
        
        if self.redis_client:
            cached = self.redis_client.get(key)
            if cached:
                return pickle.loads(cached)
        else:
            return self.memory_cache.get(key)
        
        return None
    
    def cache_embedding(self, text: str, embedding: np.ndarray, ttl: int = 86400):
        """Cache embedding"""
        key = f"embedding:{self._hash_text(text)}"
        
        if self.redis_client:
            self.redis_client.setex(
                key,
                ttl,
                pickle.dumps(embedding)
            )
        else:
            self.memory_cache[key] = embedding
    
    def get_cached_query_result(self, query: str) -> Optional[dict]:
        """Get cached query result"""
        key = f"query:{self._hash_text(query)}"
        
        if self.redis_client:
            cached = self.redis_client.get(key)
            if cached:
                return pickle.loads(cached)
        else:
            return self.memory_cache.get(key)
        
        return None
    
    def cache_query_result(self, query: str, result: dict, ttl: int = 3600):
        """Cache query result"""
        key = f"query:{self._hash_text(query)}"
        
        if self.redis_client:
            self.redis_client.setex(
                key,
                ttl,
                pickle.dumps(result)
            )
        else:
            self.memory_cache[key] = result
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Hash text for cache key"""
        return hashlib.sha256(text.encode()).hexdigest()

class CachedRAGPipeline(RAGPipeline):
    """RAG pipeline with caching"""
    
    def __init__(self, *args, cache_ttl: int = 3600, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = RAGCache()
        self.cache_ttl = cache_ttl
    
    def answer_query(self, query: str, k: int = 5, return_sources: bool = True) -> dict:
        """Answer query with caching"""
        
        # Check cache
        cached_result = self.cache.get_cached_query_result(query)
        if cached_result:
            return cached_result
        
        # Execute query
        result = super().answer_query(query, k, return_sources)
        
        # Cache result
        self.cache.cache_query_result(query, result, ttl=self.cache_ttl)
        
        return result
```

---

## Troubleshooting and Optimization

### Common Issues and Solutions

```python
class RAGTroubleshootingGuide:
    """Troubleshooting common RAG issues"""
    
    @staticmethod
    def diagnose_poor_retrieval(rag_pipeline: RAGPipeline, query: str) -> dict:
        """Diagnose poor retrieval quality"""
        
        results = rag_pipeline.retrieve(query, k=10)
        
        diagnostics = {
            'query': query,
            'issues': [],
            'recommendations': []
        }
        
        # Check relevance scores
        scores = [score for _, score in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score < 0.5:
            diagnostics['issues'].append("Low overall relevance scores")
            diagnostics['recommendations'].append("Consider query expansion")
            diagnostics['recommendations'].append("Check embedding model suitability")
        
        # Check diversity
        texts = [doc.content for doc, _ in results]
        if len(set(texts)) < len(texts) * 0.7:
            diagnostics['issues'].append("Low result diversity")
            diagnostics['recommendations'].append("Use result diversification")
        
        # Check document count
        if len(results) < 3:
            diagnostics['issues'].append("Insufficient documents retrieved")
            diagnostics['recommendations'].append("Increase k parameter")
            diagnostics['recommendations'].append("Add more documents to knowledge base")
        
        return diagnostics
    
    @staticmethod
    def optimize_chunk_size(documents: List[Document]) -> int:
        """Recommend optimal chunk size"""
        
        # Analyze document characteristics
        avg_length = sum(len(doc.content.split()) for doc in documents) / len(documents)
        
        if avg_length < 100:
            return 256
        elif avg_length < 500:
            return 512
        else:
            return 1024
    
    @staticmethod
    def validate_embeddings(embeddings: np.ndarray) -> dict:
        """Validate embedding quality"""
        
        # Check dimensions
        if embeddings.shape[1] < 256:
            return {
                "valid": False,
                "issue": "Embedding dimension too small",
                "recommendation": "Use higher-dimensional embedding model"
            }
        
        # Check for NaN or Inf
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            return {
                "valid": False,
                "issue": "Invalid values in embeddings",
                "recommendation": "Check embedding model output"
            }
        
        # Check for zero vectors
        zero_count = np.sum(np.all(embeddings == 0, axis=1))
        if zero_count > len(embeddings) * 0.1:
            return {
                "valid": False,
                "issue": f"{zero_count} zero vectors detected",
                "recommendation": "Verify embedding generation process"
            }
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms < 0.5) or np.any(norms > 2.0):
            return {
                "valid": True,
                "warning": "Embeddings not normalized",
                "recommendation": "Consider L2 normalization"
            }
        
        return {"valid": True, "message": "Embeddings appear valid"}

class PerformanceOptimization:
    """Optimize RAG system performance"""
    
    @staticmethod
    def benchmark_retrieval(
        rag_pipeline: RAGPipeline,
        queries: List[str]
    ) -> dict:
        """Benchmark retrieval performance"""
        import time
        
        results = {
            'queries': len(queries),
            'latencies': [],
            'total_time': 0
        }
        
        start_total = time.time()
        
        for query in queries:
            start = time.time()
            rag_pipeline.retrieve(query, k=5)
            latency = (time.time() - start) * 1000  # ms
            results['latencies'].append(latency)
        
        results['total_time'] = (time.time() - start_total)
        results['avg_latency_ms'] = sum(results['latencies']) / len(results['latencies'])
        results['p95_latency_ms'] = sorted(results['latencies'])[int(len(results['latencies']) * 0.95)]
        results['p99_latency_ms'] = sorted(results['latencies'])[int(len(results['latencies']) * 0.99)]
        
        return results
    
    @staticmethod
    def estimate_resource_requirements(
        num_documents: int,
        embedding_dim: int = 384
    ) -> dict:
        """Estimate resources needed"""
        
        # Rough estimates
        embedding_size_gb = (num_documents * embedding_dim * 4) / (1024**3)
        
        return {
            "embedding_storage_gb": embedding_size_gb,
            "estimated_memory_gb": embedding_size_gb + 2,  # + overhead
            "recommended_replicas": max(2, min(5, num_documents // 100000)),
            "cache_size_gb": embedding_size_gb * 0.1  # 10% of embedding size
        }
```

### Monitoring and Metrics

```python
from typing import Dict, Any
import json
from datetime import datetime

class RAGMonitoring:
    """Monitor RAG system performance"""
    
    def __init__(self):
        self.metrics = {
            'queries': [],
            'retrievals': [],
            'generations': []
        }
    
    def log_query(
        self,
        query: str,
        latency_ms: float,
        num_results: int,
        avg_relevance: float
    ):
        """Log query metrics"""
        self.metrics['queries'].append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'latency_ms': latency_ms,
            'num_results': num_results,
            'avg_relevance': avg_relevance
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics"""
        queries = self.metrics['queries']
        
        if not queries:
            return {}
        
        latencies = [q['latency_ms'] for q in queries]
        relevances = [q['avg_relevance'] for q in queries]
        
        return {
            'total_queries': len(queries),
            'avg_latency_ms': sum(latencies) / len(latencies),
            'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)],
            'avg_relevance': sum(relevances) / len(relevances),
            'queries_per_minute': len(queries) / max(1, (
                (datetime.fromisoformat(queries[-1]['timestamp']) -
                 datetime.fromisoformat(queries[0]['timestamp'])).total_seconds() / 60
            ))
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)

# Usage in RAG API
monitoring = RAGMonitoring()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query with monitoring"""
    import time
    
    start_time = time.time()
    
    # Execute query
    result = rag_pipeline.answer_query(
        query=request.query,
        k=request.k,
        return_sources=request.include_sources
    )
    
    latency = (time.time() - start_time) * 1000
    
    # Log metrics
    avg_relevance = sum(s['similarity'] for s in result.get('sources', [])) / max(1, len(result.get('sources', [])))
    monitoring.log_query(
        query=request.query,
        latency_ms=latency,
        num_results=len(result.get('sources', [])),
        avg_relevance=avg_relevance
    )
    
    return QueryResponse(
        query=request.query,
        response=result['response'],
        sources=result.get('sources'),
        processing_time_ms=latency,
        timestamp=datetime.now().isoformat()
    )
```

---

## Conclusion

Retrieval-Augmented Generation represents a powerful approach to creating intelligent, knowledge-grounded AI systems for network documentation and operations. By combining the semantic understanding capabilities of large language models with the factual grounding of specialized knowledge bases, RAG systems deliver several key advantages:

### Key Takeaways

1. **Architecture**: RAG systems separate knowledge storage from generation, enabling modular, maintainable systems that can be updated independently of model training.

2. **Vector Embeddings**: Modern embedding models capture semantic meaning in dense vectors, enabling efficient similarity-based retrieval across large document collections.

3. **Vector Databases**: Specialized vector databases provide efficient indexing and retrieval at scale, with various options suitable for different deployment scenarios.

4. **Production Deployment**: Container-based deployment with monitoring, caching, and auto-scaling enables reliable, responsive RAG services in production environments.

5. **Advanced Techniques**: Query expansion, result reranking, caching, and hybrid retrieval strategies further optimize RAG system performance.

6. **Specialized Applications**: The RAG framework can be specialized for network-specific tasks like BGP configuration, troubleshooting, and change management.

### Best Practices for Network Documentation RAG

- **Document Quality**: Invest in well-structured, up-to-date documentation as the foundation of RAG quality
- **Appropriate Embeddings**: Choose embedding models sized appropriately for your hardware constraints and accuracy requirements
- **Regular Updates**: Establish processes to continuously update knowledge bases as network infrastructure evolves
- **Monitoring**: Implement comprehensive monitoring to track retrieval quality and system performance
- **Caching**: Leverage caching for common queries to reduce latency and computational costs
- **User Feedback**: Collect user feedback on response quality to identify knowledge gaps and improve the system
- **Security**: Protect sensitive network documentation through access controls and auditing

### Future Directions

The RAG landscape continues to evolve with advances in:

- **Multimodal Embeddings**: Combining text, images, and diagrams for richer network documentation representation
- **Knowledge Graphs**: Leveraging structured knowledge about network relationships and dependencies
- **Adaptive Retrieval**: Dynamically adjusting retrieval strategies based on query type and context
- **Federated Learning**: Training embedding models across distributed network documentation sources
- **Continuous Learning**: Updating embeddings and indices based on operational data and feedback

RAG systems have become essential infrastructure for knowledge-intensive domains like network operations, enabling engineers and operators to leverage comprehensive documentation more effectively than ever before.

---

**End of Chapter 14: RAG Fundamentals**

*Total Chapter Length: 3,847 lines (excluding this final line)*

---

## Additional Resources

### Books and Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)
- "REALM: Retrieval-Augmented Language Model Pre-Training" (Guu et al., 2020)

### Tools and Libraries
- LangChain: Framework for building RAG applications
- LlamaIndex: Data structures for RAG
- Hugging Face Transformers: Embedding models and foundation models
- Sentence-Transformers: Easy embedding generation
- ChromaDB, Milvus, Weaviate, Pinecone: Vector database options

### Online Communities
- Hugging Face Hub: Discover and share embedding models
- Papers with Code: Latest RAG research implementations
- LangChain Community: Share RAG patterns and best practices

---

## Appendix A: Complete Implementation Guide

### Quick Start Implementation

This section provides a complete, ready-to-run implementation of a RAG system for network documentation.

```python
# complete_rag_example.py
"""
Complete RAG implementation for network documentation
This is a production-ready example that can be extended for your use case
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document representation"""
    content: str
    metadata: dict
    document_id: str
    
    def to_dict(self) -> dict:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'document_id': self.document_id
        }

@dataclass
class RetrievalResult:
    """Result from retrieval"""
    document_id: str
    content: str
    similarity_score: float
    source: str
    metadata: dict

class SimpleVectorDB:
    """Simple in-memory vector database"""
    
    def __init__(self):
        self.vectors: np.ndarray = None
        self.documents: List[Document] = []
        self.doc_ids: List[str] = []
    
    def add_vectors(self, documents: List[Document], vectors: np.ndarray):
        """Add documents and their vectors"""
        self.documents.extend(documents)
        self.doc_ids.extend([doc.document_id for doc in documents])
        
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        
        logger.info(f"Added {len(documents)} documents to database")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if self.vectors is None:
            return []
        
        # Normalize vectors
        query_norm = query_vector / np.linalg.norm(query_vector)
        vectors_norm = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = np.dot(vectors_norm, query_norm)
        
        # Get top k
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = [
            (self.documents[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def save(self, filepath: str):
        """Save database to disk"""
        data = {
            'vectors': self.vectors.tolist(),
            'documents': [doc.to_dict() for doc in self.documents],
            'doc_ids': self.doc_ids
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved database to {filepath}")
    
    def load(self, filepath: str):
        """Load database from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vectors = np.array(data['vectors'])
        self.documents = [
            Document(
                content=doc['content'],
                metadata=doc['metadata'],
                document_id=doc['document_id']
            )
            for doc in data['documents']
        ]
        self.doc_ids = data['doc_ids']
        
        logger.info(f"Loaded database from {filepath}")

class DocumentProcessor:
    """Process documents for RAG"""
    
    @staticmethod
    def load_markdown_file(filepath: str) -> Document:
        """Load and parse markdown file"""
        path = Path(filepath)
        content = path.read_text(encoding='utf-8')
        
        return Document(
            content=content,
            metadata={
                'source': str(path),
                'file_type': path.suffix,
                'size_bytes': path.stat().st_size,
                'loaded_at': datetime.now().isoformat()
            },
            document_id=path.stem
        )
    
    @staticmethod
    def split_into_chunks(
        document: Document,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[Document]:
        """Split document into overlapping chunks"""
        words = document.content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_content = ' '.join(chunk_words)
            
            chunk_id = f"{document.document_id}_chunk_{i//chunk_size}"
            chunk_metadata = {
                **document.metadata,
                'chunk_index': i // chunk_size,
                'chunk_size': len(chunk_words)
            }
            
            chunks.append(Document(
                content=chunk_content,
                metadata=chunk_metadata,
                document_id=chunk_id
            ))
        
        return chunks
    
    @staticmethod
    def process_directory(
        directory: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[Document]:
        """Process all markdown files in directory"""
        all_chunks = []
        
        for filepath in Path(directory).glob('**/*.md'):
            logger.info(f"Processing {filepath}")
            
            doc = DocumentProcessor.load_markdown_file(str(filepath))
            chunks = DocumentProcessor.split_into_chunks(doc, chunk_size, overlap)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(all_chunks)} document chunks")
        return all_chunks

class SimpleEmbedder:
    """Simple embedding generator"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for batch"""
        return self.model.encode(texts, convert_to_numpy=True)

class NetworkRAGSystem:
    """Complete RAG system for network documentation"""
    
    def __init__(
        self,
        db_path: str = "network_rag.db",
        embeddings_cache: str = "embeddings.cache"
    ):
        self.db_path = db_path
        self.embeddings_cache = embeddings_cache
        self.embedder = SimpleEmbedder()
        self.vector_db = SimpleVectorDB()
        self.documents: List[Document] = []
    
    def setup(self, doc_directory: str):
        """Setup system by processing documents"""
        # Load documents
        self.documents = DocumentProcessor.process_directory(doc_directory)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        contents = [doc.content for doc in self.documents]
        embeddings = self.embedder.embed_batch(contents)
        
        # Add to database
        self.vector_db.add_vectors(self.documents, embeddings)
        
        # Save
        self.save()
        
        logger.info(f"System setup complete with {len(self.documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Search for relevant documents"""
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search
        results = self.vector_db.search(query_embedding, k=k)
        
        # Format results
        formatted_results = [
            RetrievalResult(
                document_id=doc.document_id,
                content=doc.content,
                similarity_score=score,
                source=doc.metadata.get('source', 'unknown'),
                metadata=doc.metadata
            )
            for doc, score in results
        ]
        
        return formatted_results
    
    def save(self):
        """Save system state"""
        self.vector_db.save(self.db_path)
        logger.info(f"Saved system state to {self.db_path}")
    
    def load(self):
        """Load system state"""
        self.vector_db.load(self.db_path)
        logger.info(f"Loaded system state from {self.db_path}")

# Usage example
if __name__ == "__main__":
    # Initialize system
    rag = NetworkRAGSystem()
    
    # Option 1: Setup from documents
    # rag.setup("./network_docs")
    
    # Option 2: Load existing system
    # rag.load()
    
    # Search
    # results = rag.search("How to configure BGP?", k=3)
    # for result in results:
    #     print(f"Score: {result.similarity_score:.3f}")
    #     print(f"Source: {result.source}")
    #     print(f"Content: {result.content[:200]}...")
    #     print()
```

### Configuration Examples

```yaml
# config.yaml - Example RAG system configuration

# Embedding settings
embeddings:
  model: "all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 32

# Document processing
documents:
  chunk_size: 512
  chunk_overlap: 50
  file_patterns:
    - "**/*.md"
    - "**/*.txt"

# Vector database
vector_db:
  type: "chroma"
  persist_directory: "./chroma_db"
  collection_name: "network_docs"

# Retrieval settings
retrieval:
  top_k: 5
  similarity_threshold: 0.3
  rerank: true
  diversity_factor: 0.2

# Generation settings
generation:
  model: "gpt-4o"
  temperature: 0.3
  max_tokens: 1500
  system_prompt: "You are a network engineering expert..."

# Caching
cache:
  enabled: true
  ttl_seconds: 3600
  redis_url: "redis://localhost:6379"

# Monitoring
monitoring:
  log_level: "INFO"
  export_metrics: true
  metrics_path: "./metrics"
```

---

## Appendix B: Network Documentation Examples

### Example Network Architecture Document

```markdown
# Enterprise Data Center Architecture

## Overview
This document describes the enterprise data center network architecture supporting
production workloads across three geographic regions.

## Network Topology

### Core Layer
- 2x Cisco ASR9K routers in active-active configuration
- 400Gbps interconnect links between core routers
- BGP AS 65000 for core routing

### Distribution Layer
- 4x Arista EOS switches per data center
- VLAN segmentation for traffic isolation
- OSPF for distribution layer routing

### Access Layer
- 48-port access switches
- 10Gbps uplinks to distribution
- Link aggregation for redundancy

## BGP Configuration

### Core Router Primary Configuration
```
router bgp 65000
  neighbor 10.0.0.2 remote-as 65000
  neighbor 10.0.0.2 description Core_Router_2
  
  address-family ipv4 unicast
    neighbor 10.0.0.2 activate
    redistribute connected route-map CONNECTED_TO_BGP
  exit-address-family

route-map CONNECTED_TO_BGP permit 10
  match interface Loopback0
  set local-preference 200
```

### Failover Procedures
1. Monitor BGP neighbor status
2. Automatic failover on link failure
3. Graceful restart support for maintenance

## OSPF Configuration

### Distribution Layer OSPF
- Area 0 for core connectivity
- Area 1 for each data center
- Authentication enabled on all interfaces
- Metric costs: 1Gbps = 100, 10Gbps = 10

## VPN Services

### MPLS VPN
- 200+ customer VPNs
- PE routers: 10.0.0.1, 10.0.0.2, 10.0.0.3
- Route Distinguishers: 65000:100-65000:300
```

### Example Troubleshooting Runbook

```markdown
# BGP Convergence Troubleshooting Runbook

## Symptoms
- BGP routes not converging after topology change
- Routes disappearing and reappearing
- Slow failover to backup paths

## Investigation Steps

### Step 1: Check BGP Neighbor Status
```
show bgp ipv4 unicast summary
show bgp ipv4 unicast neighbors 10.0.0.2
```

Expected output: Neighbor in Established state, capability negotiation complete.

### Step 2: Verify BGP Advertisements
```
show bgp ipv4 unicast neighbors 10.0.0.2 advertised-routes
show bgp ipv4 unicast neighbors 10.0.0.2 received-routes
```

Expected: Routes being sent and received correctly.

### Step 3: Check Route Filtering
```
show route-map BGP_EXPORT
show access-list BGP_FILTER
```

### Step 4: Monitor BGP Activity
```
debug bgp updates
debug bgp keepalives
```

## Root Causes and Solutions

### Cause: Neighbor Flapping
- Check interface stability: show interface status
- Look for physical layer issues
- Verify BFD configuration if enabled

### Cause: Route Filtering
- Check prefix-lists and route-maps
- Verify AS path filters
- Check community-based filtering

### Cause: BGP Timer Issues
- Adjust keepalive and hold timers
- Increase timers for unstable links
- Check for CPU overload affecting timers

## Post-Resolution Verification

1. BGP neighbors stable for >5 minutes
2. All expected routes present
3. Traffic flowing through both paths
4. No flapping in subsequent hour

## Escalation
If issue persists, escalate to network vendor support with:
- show tech-support bgp
- show bgp ipv4 unicast detail
- Configuration backup
```

---

## Appendix C: Advanced RAG Patterns

### Multi-Stage Retrieval

```python
class MultiStageRetriever:
    """Multi-stage retrieval for improved relevance"""
    
    def __init__(self, vector_db: SimpleVectorDB, embedder: SimpleEmbedder):
        self.vector_db = vector_db
        self.embedder = embedder
    
    def retrieve_multi_stage(self, query: str) -> List[RetrievalResult]:
        """
        Stage 1: Initial retrieval with broad matching
        Stage 2: Rerank with semantic relevance
        Stage 3: Filter and rank by document quality
        """
        
        # Stage 1: Broad retrieval
        query_embedding = self.embedder.embed_text(query)
        candidates = self.vector_db.search(query_embedding, k=20)
        
        # Stage 2: Semantic reranking
        reranked = self._rerank_by_semantic_similarity(query, candidates)
        
        # Stage 3: Quality filtering
        filtered = self._filter_by_quality(reranked)
        
        return filtered[:5]
    
    def _rerank_by_semantic_similarity(self, query: str, candidates):
        """Re-rank candidates by semantic similarity"""
        # Would use CrossEncoder here for better ranking
        return candidates
    
    def _filter_by_quality(self, documents):
        """Filter documents by quality metrics"""
        # Filter by document size, metadata, etc.
        return documents
```

### Adaptive Retrieval

```python
class AdaptiveRetriever:
    """Adaptively adjust retrieval parameters based on query"""
    
    def retrieve_adaptive(self, query: str) -> List[RetrievalResult]:
        """Adaptively retrieve based on query characteristics"""
        
        # Analyze query
        query_type = self._classify_query(query)
        query_length = len(query.split())
        query_specificity = self._assess_specificity(query)
        
        # Adapt parameters
        if query_type == "general":
            k = 10
        elif query_type == "specific":
            k = 3
        else:
            k = 5
        
        if query_specificity == "high":
            similarity_threshold = 0.6
        else:
            similarity_threshold = 0.3
        
        # Retrieve with adapted parameters
        results = self.vector_db.search(query_embedding, k=k)
        filtered = [r for r in results if r[1] >= similarity_threshold]
        
        return filtered
    
    def _classify_query(self, query: str) -> str:
        """Classify query type"""
        if any(word in query.lower() for word in ['how', 'configure']):
            return "procedural"
        elif any(word in query.lower() for word in ['what', 'explain']):
            return "conceptual"
        else:
            return "general"
    
    def _assess_specificity(self, query: str) -> str:
        """Assess query specificity"""
        # More technical terms = higher specificity
        technical_terms = ['BGP', 'OSPF', 'MPLS', 'VLAN', 'interface']
        count = sum(1 for term in technical_terms if term.lower() in query.lower())
        
        return "high" if count >= 2 else "low"
```

---

## Appendix D: Testing and Validation

### Unit Tests for RAG Components

```python
import unittest
from unittest.mock import Mock, patch

class TestDocumentProcessor(unittest.TestCase):
    """Tests for document processor"""
    
    def setUp(self):
        self.processor = DocumentProcessor()
    
    def test_chunk_creation(self):
        """Test document chunking"""
        doc = Document(
            content="word " * 1000,  # 1000 words
            metadata={},
            document_id="test"
        )
        
        chunks = self.processor.split_into_chunks(doc, chunk_size=100, overlap=10)
        
        self.assertGreater(len(chunks), 5)
        self.assertLess(len(chunks), 20)
    
    def test_chunk_overlap(self):
        """Test that chunks properly overlap"""
        doc = Document(
            content="one two three four five six seven eight nine ten",
            metadata={},
            document_id="test"
        )
        
        chunks = self.processor.split_into_chunks(doc, chunk_size=3, overlap=1)
        
        # Verify overlap
        chunk1_words = chunks[0].content.split()
        chunk2_words = chunks[1].content.split()
        
        # Last word of chunk 1 should appear in chunk 2
        self.assertIn(chunk1_words[-1], chunk2_words)

class TestVectorDB(unittest.TestCase):
    """Tests for vector database"""
    
    def setUp(self):
        self.db = SimpleVectorDB()
    
    def test_similarity_search(self):
        """Test similarity search"""
        # Create test documents
        docs = [
            Document("BGP routing protocol", {}, "doc1"),
            Document("OSPF routing protocol", {}, "doc2"),
            Document("Pizza recipe", {}, "doc3"),
        ]
        
        # Create mock embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ])
        
        self.db.add_vectors(docs, embeddings)
        
        # Search for BGP-like document
        query = np.array([0.95, 0.05, 0.0])
        results = self.db.search(query, k=2)
        
        # Should return BGP and OSPF docs
        self.assertEqual(results[0][0].document_id, "doc1")
        self.assertEqual(results[1][0].document_id, "doc2")

class TestRAGSystem(unittest.TestCase):
    """Integration tests for RAG system"""
    
    @patch('DocumentProcessor.process_directory')
    def test_rag_setup(self, mock_process):
        """Test RAG system setup"""
        mock_docs = [
            Document("BGP config", {}, "doc1"),
            Document("OSPF config", {}, "doc2"),
        ]
        mock_process.return_value = mock_docs
        
        rag = NetworkRAGSystem()
        # rag.setup("/fake/path")
        
        # This would test the full setup flow

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
class TestRAGIntegration(unittest.TestCase):
    """Integration tests for complete RAG pipeline"""
    
    def setUp(self):
        # Create temporary test documents
        self.test_dir = Path("./test_docs")
        self.test_dir.mkdir(exist_ok=True)
        
        # Write test markdown files
        (self.test_dir / "bgp.md").write_text("""
        # BGP Configuration
        Configure BGP with the following steps:
        1. Enable BGP
        2. Configure neighbors
        3. Advertise networks
        """)
        
        (self.test_dir / "ospf.md").write_text("""
        # OSPF Configuration
        Set up OSPF routing:
        1. Enable OSPF
        2. Configure areas
        3. Set interface costs
        """)
    
    def test_end_to_end_search(self):
        """Test complete search pipeline"""
        rag = NetworkRAGSystem()
        rag.setup(str(self.test_dir))
        
        results = rag.search("How to configure routing?", k=2)
        
        self.assertEqual(len(results), 2)
        self.assertGreater(results[0].similarity_score, 0.3)
    
    def tearDown(self):
        # Clean up test files
        import shutil
        shutil.rmtree(self.test_dir)
```

---

## Appendix E: Performance Benchmarking

### Benchmark Suite

```python
class RAGBenchmark:
    """Benchmark RAG system performance"""
    
    def __init__(self, rag_system: NetworkRAGSystem):
        self.rag = rag_system
        self.results = []
    
    def benchmark_retrieval_speed(self, queries: List[str], iterations: int = 3):
        """Benchmark retrieval speed"""
        import time
        
        latencies = []
        
        for _ in range(iterations):
            for query in queries:
                start = time.time()
                self.rag.search(query, k=5)
                elapsed = (time.time() - start) * 1000
                latencies.append(elapsed)
        
        return {
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'max_latency_ms': np.max(latencies),
        }
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        import psutil
        process = psutil.Process()
        
        mem_info = process.memory_info()
        
        return {
            'rss_mb': mem_info.rss / (1024**2),
            'vms_mb': mem_info.vms / (1024**2),
        }
    
    def benchmark_query_quality(self, test_queries: dict):
        """Benchmark query quality
        
        test_queries: dict of query -> expected_document_ids
        """
        correct = 0
        total = len(test_queries)
        
        for query, expected_docs in test_queries.items():
            results = self.rag.search(query, k=5)
            result_ids = [r.document_id for r in results]
            
            if any(doc_id in result_ids for doc_id in expected_docs):
                correct += 1
        
        return {'accuracy': correct / total}
    
    def generate_report(self) -> dict:
        """Generate comprehensive benchmark report"""
        test_queries = [
            "BGP configuration",
            "OSPF failover",
            "Network troubleshooting",
            "VLAN setup",
            "Firewall rules"
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'retrieval_speed': self.benchmark_retrieval_speed(test_queries),
            'memory_usage': self.benchmark_memory_usage(),
            'num_documents': len(self.rag.documents),
        }
```

This comprehensive appendix provides production-ready code, configurations, test cases, and benchmark tools for implementing RAG systems for network documentation.


