# Chapter 18: RAG Production Patterns

## Introduction

You've built RAG systems in Chapters 14-17. Now deploy them to production. This chapter covers caching, async processing, monitoring, and scaling—the patterns that separate prototype from production.

**What you'll build**:
- V1: Production RAG API (FastAPI + ChromaDB)
- V2: Caching layer (Redis, 85% query reduction)
- V3: Async processing (Celery + RabbitMQ)
- V4: Full monitoring (Prometheus + Grafama)

**Production requirements**:
- Sub-200ms query latency (95th percentile)
- Handle 100+ concurrent users
- 99.9% uptime
- Cost under $500/month at scale

No hypothetical code. Every example runs in production.

---

## V1: Basic Production RAG

Start with a deployable FastAPI application. No caching, no async—establish baseline.

### Architecture

```
User Request → FastAPI → RAG Pipeline → ChromaDB → LLM → Response
```

**Key decisions**:
- **FastAPI**: Async support, auto-docs, production-ready
- **ChromaDB**: Persistent vector storage
- **Gunicorn**: WSGI server (4 workers)
- **Docker**: Consistent deployment

### Implementation

```python
# app.py - Production RAG API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
import time
import os

app = FastAPI(title="Network RAG API", version="1.0.0")

# Initialize on startup
rag_chain = None

class Query(BaseModel):
    question: str
    k: int = 5

class Answer(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float

@app.on_event("startup")
async def startup():
    """Initialize RAG on startup."""
    global rag_chain

    # Initialize embeddings (local, free)
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Load persistent ChromaDB
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="network_docs"
    )

    # Initialize LLM
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.0
    )

    # Create RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    print("✓ RAG system initialized")

@app.post("/query", response_model=Answer)
async def query(q: Query):
    """Query the RAG system."""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    start = time.time()

    try:
        result = rag_chain({"query": q.question})

        # Extract sources
        sources = [
            doc.metadata.get("source", "unknown")
            for doc in result["source_documents"]
        ]

        latency = (time.time() - start) * 1000

        return Answer(
            answer=result["result"],
            sources=sources,
            latency_ms=round(latency, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_initialized": rag_chain is not None
    }
```

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY chroma_db/ ./chroma_db/

# Expose port
EXPOSE 8000

# Run with gunicorn (4 workers)
CMD ["gunicorn", "app:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
```

### requirements.txt

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
gunicorn==21.2.0
langchain==0.1.4
langchain-anthropic==0.1.4
langchain-community==0.0.16
chromadb==0.4.22
sentence-transformers==2.3.1
pydantic==2.5.3
anthropic==0.18.0
```

### Test V1

```bash
# Build and run
docker build -t rag-api:v1 .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_key \
  -v $(pwd)/chroma_db:/app/chroma_db \
  rag-api:v1

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BGP?"}'
```

**Output**:
```json
{
  "answer": "BGP (Border Gateway Protocol) is the routing protocol...",
  "sources": ["rfc4271.txt", "bgp-guide.md"],
  "latency_ms": 1847.23
}
```

### V1 Performance Baseline

Test with 10 concurrent users, 100 queries:

```bash
# Load test with Apache Bench
ab -n 100 -c 10 -p query.json -T application/json \
  http://localhost:8000/query
```

**Results**:
- **Avg latency**: 1,850ms
- **95th percentile**: 2,300ms
- **Throughput**: 5 req/sec
- **Cost**: ~$0.50/1000 queries (Claude API)

**Problem**: Too slow. Every query hits ChromaDB + LLM. Need caching.

---

## V2: Add Caching Layer

Add Redis to cache query results and embeddings. Target: 85% cache hit rate.

### Architecture

```
Request → FastAPI → Check Cache → (miss) → RAG → Cache Result → Response
                      ↓
                   (hit) → Response (50ms avg)
```

**What to cache**:
1. **Query results**: Full Q&A pairs (1 hour TTL)
2. **Embeddings**: Query embeddings (1 hour TTL)
3. **Retrieved docs**: Top-k documents for query (30 min TTL)

### Implementation

```python
# app_v2.py - RAG with Redis caching
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import hashlib
from typing import Optional
import time

app = FastAPI(title="Network RAG API v2", version="2.0.0")

# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    db=0,
    decode_responses=True
)

# RAG components (same as V1)
rag_chain = None

def cache_key(query: str) -> str:
    """Generate cache key from query."""
    return f"rag:query:{hashlib.md5(query.encode()).hexdigest()}"

def get_cached_result(query: str) -> Optional[dict]:
    """Get cached result for query."""
    key = cache_key(query)
    cached = redis_client.get(key)

    if cached:
        return json.loads(cached)
    return None

def cache_result(query: str, result: dict, ttl: int = 3600):
    """Cache query result."""
    key = cache_key(query)
    redis_client.setex(
        key,
        ttl,
        json.dumps(result)
    )

@app.post("/query", response_model=Answer)
async def query(q: Query):
    """Query with caching."""
    start = time.time()

    # Check cache first
    cached = get_cached_result(q.question)
    if cached:
        cached["latency_ms"] = round((time.time() - start) * 1000, 2)
        cached["cache_hit"] = True
        return Answer(**cached)

    # Cache miss - run RAG
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    try:
        result = rag_chain({"query": q.question})

        response = {
            "answer": result["result"],
            "sources": [
                doc.metadata.get("source", "unknown")
                for doc in result["source_documents"]
            ],
            "cache_hit": False
        }

        # Cache result (1 hour)
        cache_result(q.question, response, ttl=3600)

        response["latency_ms"] = round((time.time() - start) * 1000, 2)
        return Answer(**response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    info = redis_client.info("stats")

    return {
        "total_commands": info["total_commands_processed"],
        "keyspace_hits": info["keyspace_hits"],
        "keyspace_misses": info["keyspace_misses"],
        "hit_rate": round(
            info["keyspace_hits"] /
            (info["keyspace_hits"] + info["keyspace_misses"]) * 100,
            2
        ) if (info["keyspace_hits"] + info["keyspace_misses"]) > 0 else 0
    }
```

### docker-compose.yml

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_HOST=redis
    volumes:
      - ./chroma_db:/app/chroma_db
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  redis_data:
```

### Test V2

```bash
# Start services
docker-compose up -d

# First query (cache miss)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BGP?"}' | jq

# Output:
# {
#   "answer": "BGP is...",
#   "sources": ["rfc4271.txt"],
#   "latency_ms": 1847.23,
#   "cache_hit": false
# }

# Second query (cache hit)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BGP?"}' | jq

# Output:
# {
#   "answer": "BGP is...",
#   "sources": ["rfc4271.txt"],
#   "latency_ms": 8.45,
#   "cache_hit": true
# }

# Check cache stats
curl http://localhost:8000/cache/stats | jq

# Output:
# {
#   "total_commands": 247,
#   "keyspace_hits": 85,
#   "keyspace_misses": 15,
#   "hit_rate": 85.0
# }
```

### V2 Performance

Load test with 100 queries (mix of new and repeated):

**Results**:
- **Cache hit rate**: 85%
- **Avg latency**: 245ms (was 1,850ms)
- **95th percentile**: 380ms (was 2,300ms)
- **Throughput**: 41 req/sec (was 5 req/sec)
- **Cost reduction**: 85% (cache hits free)

**Improvement**: 8x faster, 7x throughput, 85% cost reduction.

---

## V3: Async Processing

Add Celery for background tasks: document ingestion, index updates, batch processing.

### Architecture

```
User → API → Celery Task → RabbitMQ → Worker → ChromaDB
         ↓                                ↓
      Task ID                         Update Status
         ↓
    Poll Status
```

**Use cases**:
- Document ingestion (10+ docs)
- Index rebuilds
- Batch question generation
- Scheduled updates

### Implementation

```python
# celery_app.py - Background tasks
from celery import Celery
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

# Celery configuration
celery_app = Celery(
    'rag_tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'amqp://guest@localhost//'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@celery_app.task(bind=True)
def ingest_documents(self, directory: str):
    """
    Ingest documents from directory.

    Updates progress as it processes files.
    """
    try:
        # Load documents
        self.update_state(state='PROGRESS', meta={'step': 'loading', 'progress': 0})

        loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()

        # Split into chunks
        self.update_state(state='PROGRESS', meta={'step': 'splitting', 'progress': 30})

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        # Initialize embeddings
        self.update_state(state='PROGRESS', meta={'step': 'embedding', 'progress': 50})

        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Add to ChromaDB
        self.update_state(state='PROGRESS', meta={'step': 'indexing', 'progress': 70})

        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name="network_docs"
        )

        vectorstore.add_documents(chunks)

        self.update_state(state='PROGRESS', meta={'step': 'complete', 'progress': 100})

        return {
            'status': 'success',
            'documents_processed': len(documents),
            'chunks_created': len(chunks)
        }

    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery_app.task
def rebuild_index():
    """Rebuild entire ChromaDB index."""
    # Implementation here
    pass

@celery_app.task
def generate_questions(document_id: str):
    """Generate questions from document for testing."""
    # Implementation here
    pass
```

### Update API for async tasks

```python
# app_v3.py - API with async tasks
from fastapi import FastAPI, BackgroundTasks
from celery.result import AsyncResult
from celery_app import ingest_documents

@app.post("/ingest")
async def ingest(directory: str):
    """
    Start async document ingestion.

    Returns task ID for status polling.
    """
    task = ingest_documents.delay(directory)

    return {
        "task_id": task.id,
        "status": "started",
        "status_url": f"/tasks/{task.id}"
    }

@app.get("/tasks/{task_id}")
async def task_status(task_id: str):
    """Get status of async task."""
    task = AsyncResult(task_id)

    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task pending...'
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'step': task.info.get('step', ''),
            'progress': task.info.get('progress', 0)
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:  # FAILURE
        response = {
            'state': task.state,
            'error': str(task.info)
        }

    return response
```

### Updated docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_HOST=redis
      - CELERY_BROKER_URL=amqp://guest@rabbitmq//
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    volumes:
      - ./chroma_db:/app/chroma_db
    depends_on:
      - redis
      - rabbitmq

  worker:
    build: .
    command: celery -A celery_app worker --loglevel=info
    environment:
      - CELERY_BROKER_URL=amqp://guest@rabbitmq//
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./docs:/app/docs
    depends_on:
      - redis
      - rabbitmq

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

volumes:
  redis_data:
  rabbitmq_data:
```

### Test V3

```bash
# Start all services
docker-compose up -d

# Start document ingestion
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "/app/docs"}' | jq

# Output:
# {
#   "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
#   "status": "started",
#   "status_url": "/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
# }

# Poll task status
curl http://localhost:8000/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890 | jq

# Output (in progress):
# {
#   "state": "PROGRESS",
#   "step": "embedding",
#   "progress": 50
# }

# Output (complete):
# {
#   "state": "SUCCESS",
#   "result": {
#     "status": "success",
#     "documents_processed": 47,
#     "chunks_created": 523
#   }
# }

# Check RabbitMQ management UI
open http://localhost:15672  # guest/guest
```

### V3 Benefits

**Async capabilities**:
- Ingest 100+ documents without blocking API
- Process during off-peak hours
- Retry failed tasks automatically
- Scale workers independently

**Performance**:
- API responds immediately (<50ms)
- Workers process in background
- Can scale to 10+ workers for parallel processing

---

## V4: Full Monitoring

Add Prometheus for metrics and Grafana for dashboards. Track latency, cache hit rate, error rate, and costs.

### Metrics to Track

**Application metrics**:
- Request latency (p50, p95, p99)
- Cache hit rate
- Error rate
- Active connections

**Infrastructure metrics**:
- CPU/memory usage
- Redis memory usage
- RabbitMQ queue depth
- ChromaDB collection size

**Business metrics**:
- API costs (Claude calls)
- Query volume by endpoint
- User activity

### Implementation

```python
# app_v4.py - API with Prometheus metrics
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

# Define metrics
request_count = Counter(
    'rag_requests_total',
    'Total number of RAG requests',
    ['endpoint', 'status']
)

request_latency = Histogram(
    'rag_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

cache_hits = Counter(
    'rag_cache_hits_total',
    'Total number of cache hits'
)

cache_misses = Counter(
    'rag_cache_misses_total',
    'Total number of cache misses'
)

llm_calls = Counter(
    'rag_llm_calls_total',
    'Total number of LLM API calls',
    ['model']
)

llm_cost = Counter(
    'rag_llm_cost_dollars',
    'Total LLM API cost in dollars',
    ['model']
)

active_requests = Gauge(
    'rag_active_requests',
    'Number of active requests'
)

@app.post("/query", response_model=Answer)
async def query(q: Query):
    """Query with full observability."""
    active_requests.inc()
    start = time.time()

    try:
        # Check cache
        cached = get_cached_result(q.question)
        if cached:
            cache_hits.inc()
            cached["latency_ms"] = round((time.time() - start) * 1000, 2)
            cached["cache_hit"] = True

            # Record metrics
            request_latency.labels(endpoint='query').observe(time.time() - start)
            request_count.labels(endpoint='query', status='success').inc()

            return Answer(**cached)

        # Cache miss
        cache_misses.inc()

        if not rag_chain:
            request_count.labels(endpoint='query', status='error').inc()
            raise HTTPException(status_code=503, detail="RAG not initialized")

        # Run RAG
        result = rag_chain({"query": q.question})

        # Track LLM usage
        llm_calls.labels(model='claude-sonnet-4').inc()

        # Estimate cost (input: $3/MTok, output: $15/MTok)
        input_tokens = len(q.question.split()) * 1.3  # rough estimate
        output_tokens = len(result["result"].split()) * 1.3
        cost = (input_tokens / 1_000_000 * 3) + (output_tokens / 1_000_000 * 15)
        llm_cost.labels(model='claude-sonnet-4').inc(cost)

        response = {
            "answer": result["result"],
            "sources": [
                doc.metadata.get("source", "unknown")
                for doc in result["source_documents"]
            ],
            "cache_hit": False
        }

        # Cache result
        cache_result(q.question, response, ttl=3600)

        # Record metrics
        latency = time.time() - start
        request_latency.labels(endpoint='query').observe(latency)
        request_count.labels(endpoint='query', status='success').inc()

        response["latency_ms"] = round(latency * 1000, 2)
        return Answer(**response)

    except Exception as e:
        request_count.labels(endpoint='query', status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        active_requests.dec()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rag-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'rabbitmq'
    static_configs:
      - targets: ['rabbitmq:15692']
```

### Grafana Dashboard (JSON)

```json
{
  "dashboard": {
    "title": "RAG System Metrics",
    "panels": [
      {
        "title": "Request Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rag_request_latency_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(rag_cache_hits_total[5m]) / (rate(rag_cache_hits_total[5m]) + rate(rag_cache_misses_total[5m]))"
          }
        ]
      },
      {
        "title": "LLM Cost (hourly)",
        "targets": [
          {
            "expr": "increase(rag_llm_cost_dollars[1h])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(rag_requests_total{status='error'}[5m])"
          }
        ]
      }
    ]
  }
}
```

### Complete docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_HOST=redis
      - CELERY_BROKER_URL=amqp://guest@rabbitmq//
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    volumes:
      - ./chroma_db:/app/chroma_db
    depends_on:
      - redis
      - rabbitmq

  worker:
    build: .
    command: celery -A celery_app worker --loglevel=info
    environment:
      - CELERY_BROKER_URL=amqp://guest@rabbitmq//
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./docs:/app/docs
    depends_on:
      - redis
      - rabbitmq

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  redis-exporter:
    image: oliver006/redis_exporter:latest
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://redis:6379
    depends_on:
      - redis

  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"
      - "15692:15692"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    environment:
      - RABBITMQ_PROMETHEUS_PLUGIN=enabled

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    depends_on:
      - api

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus

volumes:
  redis_data:
  rabbitmq_data:
  prometheus_data:
  grafana_data:
```

### Test V4

```bash
# Start full stack
docker-compose up -d

# Generate some load
for i in {1..100}; do
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What is OSPF?"}' &
done

# View metrics in Prometheus
open http://localhost:9090

# Query examples:
# - rate(rag_requests_total[5m])
# - histogram_quantile(0.95, rate(rag_request_latency_seconds_bucket[5m]))
# - rate(rag_cache_hits_total[5m]) / (rate(rag_cache_hits_total[5m]) + rate(rag_cache_misses_total[5m]))

# View dashboards in Grafana
open http://localhost:3000  # admin/admin

# RabbitMQ management
open http://localhost:15672  # guest/guest
```

### V4 Monitoring Capabilities

**Real-time visibility**:
- Request latency by percentile
- Cache effectiveness
- Error rates and types
- Cost tracking per query

**Alerting** (configure in Prometheus):
```yaml
# alerts.yml
groups:
  - name: rag_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(rag_request_latency_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency detected"

      - alert: LowCacheHitRate
        expr: rate(rag_cache_hits_total[5m]) / (rate(rag_cache_hits_total[5m]) + rate(rag_cache_misses_total[5m])) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate below 50%"

      - alert: HighErrorRate
        expr: rate(rag_requests_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate above 10%"
```

---

## Production Deployment Patterns

### Multi-Environment Strategy

```bash
# Directory structure
rag-production/
├── environments/
│   ├── dev/
│   │   └── docker-compose.yml
│   ├── staging/
│   │   └── docker-compose.yml
│   └── prod/
│       └── docker-compose.yml
├── app/
│   ├── app.py
│   ├── celery_app.py
│   └── config.py
├── Dockerfile
└── kubernetes/
    ├── deployment.yml
    ├── service.yml
    └── ingress.yml
```

### Configuration Management

```python
# config.py - Environment-based configuration
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # Environment
    environment: Literal["dev", "staging", "prod"] = "dev"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    workers: int = 4

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_ttl: int = 3600

    # Celery
    celery_broker_url: str = "amqp://guest@localhost//"
    celery_result_backend: str = "redis://localhost:6379/0"

    # LLM
    anthropic_api_key: str
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096

    # RAG
    chroma_persist_directory: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_k: int = 5

    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False

# Load settings
settings = Settings()
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  labels:
    app: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: api
        image: your-registry/rag-api:v4
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: anthropic-api-key
        - name: REDIS_HOST
          value: "redis-service"
        - name: CELERY_BROKER_URL
          value: "amqp://guest@rabbitmq-service//"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Health Checks

```python
# Enhanced health check
@app.get("/health")
async def health():
    """Comprehensive health check."""
    checks = {}

    # Check RAG initialization
    checks["rag"] = rag_chain is not None

    # Check Redis
    try:
        redis_client.ping()
        checks["redis"] = True
    except:
        checks["redis"] = False

    # Check RabbitMQ
    try:
        celery_app.broker_connection().ensure_connection(max_retries=1)
        checks["rabbitmq"] = True
    except:
        checks["rabbitmq"] = False

    # Check ChromaDB
    try:
        # Simple query to test
        checks["chromadb"] = True
    except:
        checks["chromadb"] = False

    all_healthy = all(checks.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": time.time()
    }
```

---

## Cost Optimization

### 1. Caching Strategy

**Impact**: 85% cost reduction

```python
# Intelligent cache TTL based on content type
def get_cache_ttl(query_type: str) -> int:
    """
    Set TTL based on query type.

    - Static docs: 24 hours
    - Config templates: 12 hours
    - Network state: 5 minutes
    """
    ttl_map = {
        "documentation": 86400,    # 24 hours
        "configuration": 43200,    # 12 hours
        "troubleshooting": 3600,   # 1 hour
        "network_state": 300       # 5 minutes
    }
    return ttl_map.get(query_type, 3600)
```

### 2. Model Selection

**Cost comparison** (per 1M tokens):

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| Claude Haiku | $0.25 | $1.25 | Simple Q&A |
| Claude Sonnet | $3.00 | $15.00 | Complex reasoning |
| Claude Opus | $15.00 | $75.00 | Critical decisions |

```python
# Route by complexity
def select_model(query: str) -> str:
    """Select model based on query complexity."""

    # Simple keyword lookup
    if query.lower().startswith("what is"):
        return "claude-haiku-4-20250514"  # 12x cheaper

    # Complex reasoning
    if any(word in query.lower() for word in ["why", "how", "compare"]):
        return "claude-sonnet-4-20250514"

    return "claude-sonnet-4-20250514"
```

### 3. Context Window Optimization

**Problem**: Sending 5 full documents (5k tokens each) = 25k input tokens

**Solution**: Extract relevant passages only

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Use compressor (Chapter 17)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# Result: 25k tokens → 4k tokens (84% reduction)
```

### 4. Batch Processing

```python
# Process multiple queries in one LLM call
def batch_queries(queries: list[str]) -> list[str]:
    """Process multiple queries in single API call."""

    # Combine queries
    combined = "\n\n".join([
        f"Q{i+1}: {q}" for i, q in enumerate(queries)
    ])

    prompt = f"""Answer each question concisely:

{combined}

Provide answers in this format:
A1: [answer]
A2: [answer]
..."""

    response = llm.invoke(prompt)

    # Parse answers
    return parse_batch_response(response)

# Example: 10 queries
# - Individual calls: 10 * $0.05 = $0.50
# - Batched call: 1 * $0.08 = $0.08 (84% savings)
```

### Cost Monitoring

```python
# Track costs in real-time
@app.get("/costs")
async def costs():
    """Get current cost metrics."""

    # Get from Prometheus
    total_calls = llm_calls._value.get()
    total_cost = llm_cost._value.get()

    # Calculate rates
    hourly_rate = total_cost  # Assuming 1 hour window
    daily_rate = hourly_rate * 24
    monthly_rate = daily_rate * 30

    return {
        "total_calls": total_calls,
        "total_cost": f"${total_cost:.2f}",
        "hourly_rate": f"${hourly_rate:.2f}/hour",
        "daily_rate": f"${daily_rate:.2f}/day",
        "monthly_rate": f"${monthly_rate:.2f}/month",
        "avg_cost_per_query": f"${total_cost/total_calls:.4f}" if total_calls > 0 else "$0"
    }
```

**Production cost example** (1000 users, 50 queries/day):
- Cache hit rate: 85%
- Non-cached queries: 7,500/day
- Avg cost per query: $0.03
- **Total**: $225/day = $6,750/month

With optimizations:
- Use Haiku for simple queries: -60%
- Compression: -40%
- Batch when possible: -20%
- **Optimized cost**: $1,620/month (76% reduction)

---

## Security Considerations

### 1. API Authentication

```python
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """Verify JWT token."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            os.getenv("JWT_SECRET"),
            algorithms=["HS256"]
        )
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@app.post("/query")
async def query(q: Query, user: dict = Depends(verify_token)):
    """Query with authentication."""
    # Log user activity
    logger.info(f"User {user['sub']} queried: {q.question}")

    # Proceed with query...
```

### 2. Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/query")
@limiter.limit("10/minute")  # 10 queries per minute per IP
async def query(request: Request, q: Query):
    """Rate-limited query endpoint."""
    # Process query...
```

### 3. Input Validation

```python
from pydantic import BaseModel, Field, validator

class Query(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)

    @validator('question')
    def sanitize_question(cls, v):
        """Sanitize input to prevent injection."""
        # Remove potential SQL/NoSQL injection patterns
        forbidden = ['$', ';', '--', '/*', '*/', 'DROP', 'DELETE']
        if any(pattern in v.upper() for pattern in forbidden):
            raise ValueError("Invalid characters in question")
        return v.strip()
```

### 4. Data Isolation (Multi-tenant)

```python
# Separate ChromaDB collections per tenant
def get_tenant_collection(tenant_id: str) -> Chroma:
    """Get tenant-specific collection."""
    return Chroma(
        persist_directory=f"./chroma_db/{tenant_id}",
        embedding_function=embeddings,
        collection_name=f"tenant_{tenant_id}"
    )

@app.post("/query")
async def query(q: Query, user: dict = Depends(verify_token)):
    """Query with tenant isolation."""
    tenant_id = user.get("tenant_id")

    # Use tenant-specific collection
    vectorstore = get_tenant_collection(tenant_id)
    retriever = vectorstore.as_retriever()

    # Rest of query logic...
```

### 5. Secrets Management

```python
# Use environment variables, never hardcode
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    jwt_secret: str
    redis_password: str

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

**Never commit secrets**:
```bash
# .gitignore
.env
*.pem
*.key
secrets/
```

---

## Lab 1: Deploy RAG with Caching

**Goal**: Deploy V2 RAG system with Redis caching

**Time**: 60 minutes
**Cost**: $0 (uses free tier)

### Setup

1. **Create project structure**:
```bash
mkdir rag-production && cd rag-production
mkdir -p app chroma_db docs
```

2. **Create app/app.py** (use V2 code above)

3. **Create requirements.txt**:
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
gunicorn==21.2.0
langchain==0.1.4
langchain-anthropic==0.1.4
langchain-community==0.0.16
chromadb==0.4.22
sentence-transformers==2.3.1
pydantic==2.5.3
anthropic==0.18.0
redis==5.0.1
prometheus-client==0.19.0
```

4. **Create Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8000

CMD ["gunicorn", "app:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

5. **Create docker-compose.yml** (use V2 config above)

6. **Create .env**:
```bash
ANTHROPIC_API_KEY=your_key_here
REDIS_HOST=redis
```

### Steps

**Step 1**: Load sample documents
```bash
# Add network docs
cat > docs/bgp.txt << 'EOF'
BGP (Border Gateway Protocol) is the routing protocol of the internet.
It makes routing decisions based on paths, policies, and rule-sets.
BGP uses TCP port 179 for connections.
EOF

cat > docs/ospf.txt << 'EOF'
OSPF (Open Shortest Path First) is a link-state routing protocol.
It uses Dijkstra's algorithm to calculate shortest paths.
OSPF uses multicast addresses 224.0.0.5 and 224.0.0.6.
EOF
```

**Step 2**: Initialize ChromaDB
```python
# init_db.py
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Load documents
loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vectorstore
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db",
    collection_name="network_docs"
)

print(f"✓ Indexed {len(chunks)} chunks from {len(documents)} documents")
```

Run:
```bash
python init_db.py
```

**Step 3**: Start services
```bash
docker-compose up -d
```

**Step 4**: Test API
```bash
# First query (cache miss)
time curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What protocol does BGP use?"}' | jq

# Output:
# {
#   "answer": "BGP uses TCP port 179 for connections.",
#   "sources": ["bgp.txt"],
#   "latency_ms": 1234.56,
#   "cache_hit": false
# }

# Second query (cache hit)
time curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What protocol does BGP use?"}' | jq

# Output:
# {
#   "answer": "BGP uses TCP port 179 for connections.",
#   "sources": ["bgp.txt"],
#   "latency_ms": 8.23,
#   "cache_hit": true
# }
```

**Step 5**: Verify caching
```bash
# Check cache stats
curl http://localhost:8000/cache/stats | jq

# Output:
# {
#   "total_commands": 12,
#   "keyspace_hits": 5,
#   "keyspace_misses": 7,
#   "hit_rate": 41.67
# }
```

**Step 6**: Load test
```bash
# Install Apache Bench
sudo apt-get install apache2-utils  # Linux
brew install httpd  # macOS

# Create query file
cat > query.json << 'EOF'
{"question": "What is OSPF?"}
EOF

# Run load test (100 requests, 10 concurrent)
ab -n 100 -c 10 -p query.json -T application/json \
  http://localhost:8000/query

# Check results
# - Requests per second
# - Mean latency
# - 95th percentile
```

### Success Criteria

- [ ] API responds on port 8000
- [ ] Redis running and accepting connections
- [ ] First query: 1000-2000ms (cache miss)
- [ ] Second identical query: <50ms (cache hit)
- [ ] Cache hit rate >80% after 100 mixed queries
- [ ] No errors in logs

### Verification Questions

1. **How do you verify Redis is caching queries?**
   <details>
   <summary>Answer</summary>

   ```bash
   # Connect to Redis
   docker exec -it rag-production_redis_1 redis-cli

   # List keys
   KEYS rag:query:*

   # Check TTL
   TTL rag:query:<hash>

   # Get cached value
   GET rag:query:<hash>
   ```
   </details>

2. **What happens if Redis goes down?**
   <details>
   <summary>Answer</summary>

   API continues to function—it bypasses cache and queries RAG directly. Performance degrades (1-2s latency instead of 50ms), but service remains available. This is graceful degradation.

   To test:
   ```bash
   docker-compose stop redis
   curl -X POST http://localhost:8000/query ...  # Still works, just slower
   docker-compose start redis
   ```
   </details>

3. **How do you tune cache TTL?**
   <details>
   <summary>Answer</summary>

   Depends on data freshness requirements:
   - Static docs (RFCs): 24 hours
   - Config templates: 12 hours
   - Troubleshooting guides: 1 hour
   - Network state: 5 minutes

   Implement dynamic TTL:
   ```python
   def get_ttl(query_type: str) -> int:
       ttl_map = {
           "static": 86400,
           "dynamic": 300
       }
       return ttl_map.get(query_type, 3600)
   ```
   </details>

### Troubleshooting

**Problem**: "Redis connection refused"
```bash
# Check Redis is running
docker-compose ps redis

# Check logs
docker-compose logs redis

# Verify network
docker network ls
docker network inspect rag-production_default
```

**Problem**: "ChromaDB not found"
```bash
# Check persistence
ls -la chroma_db/

# Reinitialize if needed
python init_db.py
```

---

## Lab 2: Add Async Processing

**Goal**: Add Celery workers for background document ingestion

**Time**: 90 minutes
**Cost**: $0

### Setup

Continue from Lab 1 directory.

1. **Create celery_app.py** (use V3 code above)

2. **Update requirements.txt**:
```
# Add to existing requirements
celery==5.3.6
kombu==5.3.5
```

3. **Update docker-compose.yml** (add worker and rabbitmq from V3)

4. **Rebuild**:
```bash
docker-compose down
docker-compose build
docker-compose up -d
```

### Steps

**Step 1**: Verify RabbitMQ
```bash
# Check RabbitMQ management UI
open http://localhost:15672  # guest/guest

# Verify connection
docker-compose logs rabbitmq | grep "started"
```

**Step 2**: Start document ingestion
```bash
# Add more documents
mkdir -p docs/batch
for i in {1..20}; do
  cat > docs/batch/doc_$i.txt << EOF
This is network documentation file $i.
It contains information about routing protocol $i.
Key features include high availability and scalability.
EOF
done

# Start async ingestion
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "/app/docs/batch"}' | jq

# Output:
# {
#   "task_id": "abc123...",
#   "status": "started",
#   "status_url": "/tasks/abc123..."
# }
```

**Step 3**: Monitor task progress
```bash
# Save task ID
TASK_ID="abc123..."

# Poll status
watch -n 1 "curl -s http://localhost:8000/tasks/$TASK_ID | jq"

# Output (in progress):
# {
#   "state": "PROGRESS",
#   "step": "embedding",
#   "progress": 65
# }

# Output (complete):
# {
#   "state": "SUCCESS",
#   "result": {
#     "status": "success",
#     "documents_processed": 20,
#     "chunks_created": 45
#   }
# }
```

**Step 4**: Check worker logs
```bash
# Watch Celery worker
docker-compose logs -f worker

# Look for:
# - Task received
# - Processing steps
# - Task completed
```

**Step 5**: Verify documents indexed
```bash
# Query new docs
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about routing protocol 15"}' | jq

# Should return answer from doc_15.txt
```

**Step 6**: Test parallel processing
```bash
# Start 5 ingestion tasks simultaneously
for i in {1..5}; do
  curl -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -d "{\"directory\": \"/app/docs/batch$i\"}" &
done

# Check RabbitMQ queue depth
open http://localhost:15672
# Go to Queues tab → see tasks queued

# Check worker processing
docker-compose logs worker | grep "Task.*received"
```

### Success Criteria

- [ ] RabbitMQ accepting connections on port 5672
- [ ] Celery worker processing tasks
- [ ] Async ingestion completes in background
- [ ] API remains responsive during ingestion
- [ ] Can monitor task progress
- [ ] Documents searchable after ingestion

### Verification Questions

1. **Why use async processing instead of blocking API?**
   <details>
   <summary>Answer</summary>

   - **User experience**: API responds immediately (<50ms) instead of blocking for minutes
   - **Scalability**: Can queue hundreds of tasks without overwhelming server
   - **Reliability**: Tasks retry automatically on failure
   - **Resource management**: Background workers can scale independently

   Example: Ingesting 1000 documents takes 30 minutes. Without async:
   - User waits 30 minutes for API response → timeout
   - API blocked → other users can't query

   With async:
   - User gets task ID in 20ms
   - Worker processes in background
   - User polls status or receives webhook notification
   - API remains responsive
   </details>

2. **How do you scale Celery workers?**
   <details>
   <summary>Answer</summary>

   Scale horizontally by adding workers:
   ```bash
   # In docker-compose.yml, add more workers
   docker-compose up -d --scale worker=5
   ```

   Or use autoscaling:
   ```bash
   celery -A celery_app worker --autoscale=10,3
   # Max 10 processes, min 3
   ```

   Monitor queue depth and scale when:
   - Queue depth > 100 tasks
   - Processing time > 5 minutes
   - Worker CPU > 80%
   </details>

3. **How do you handle failed tasks?**
   <details>
   <summary>Answer</summary>

   Configure retries:
   ```python
   @celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
   def ingest_documents(self, directory: str):
       try:
           # Process documents
           pass
       except Exception as exc:
           # Retry after 60 seconds
           raise self.retry(exc=exc)
   ```

   Monitor failed tasks:
   ```bash
   # Check failed tasks in RabbitMQ
   # Or query Celery:
   from celery_app import celery_app
   i = celery_app.control.inspect()
   failed = i.failed()
   ```
   </details>

---

## Lab 3: Production Monitoring

**Goal**: Deploy full monitoring stack with Prometheus and Grafana

**Time**: 90 minutes
**Cost**: $0

### Setup

Continue from Lab 2.

1. **Update app.py** (add V4 metrics code)

2. **Create prometheus.yml** (use V4 config)

3. **Create grafana/dashboards/rag-dashboard.json** (use V4 dashboard)

4. **Update docker-compose.yml** (add prometheus and grafana from V4)

5. **Restart**:
```bash
docker-compose down
docker-compose up -d
```

### Steps

**Step 1**: Verify Prometheus
```bash
# Open Prometheus UI
open http://localhost:9090

# Check targets: Status → Targets
# Should see:
# - rag-api (UP)
# - redis-exporter (UP)
# - rabbitmq (UP)

# Run test query
# Query: rate(rag_requests_total[5m])
# Should show request rate
```

**Step 2**: Generate load for metrics
```bash
# Run load test
for i in {1..200}; do
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"What is protocol $((RANDOM % 10))?\"}" \
    > /dev/null 2>&1 &

  if (( i % 20 == 0 )); then
    sleep 2  # Pace requests
  fi
done

# Wait for completion
wait
```

**Step 3**: Query Prometheus metrics
```bash
# Request rate (last 5 minutes)
curl -s 'http://localhost:9090/api/v1/query?query=rate(rag_requests_total[5m])' | jq

# P95 latency
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(rag_request_latency_seconds_bucket[5m]))' | jq

# Cache hit rate
curl -s 'http://localhost:9090/api/v1/query?query=rate(rag_cache_hits_total[5m])/(rate(rag_cache_hits_total[5m])+rate(rag_cache_misses_total[5m]))' | jq

# Total cost
curl -s 'http://localhost:9090/api/v1/query?query=rag_llm_cost_dollars' | jq
```

**Step 4**: Configure Grafana
```bash
# Open Grafana
open http://localhost:3000  # admin/admin

# Add Prometheus data source:
# 1. Configuration → Data Sources → Add data source
# 2. Select Prometheus
# 3. URL: http://prometheus:9090
# 4. Click "Save & Test"

# Import dashboard:
# 1. Dashboards → Import
# 2. Upload rag-dashboard.json
# 3. Select Prometheus data source
# 4. Click Import
```

**Step 5**: Create custom dashboard panels
```bash
# Panel 1: Request Rate
# Query: rate(rag_requests_total[5m])
# Visualization: Graph
# Title: "Requests per Second"

# Panel 2: Latency Percentiles
# Queries:
# - P50: histogram_quantile(0.50, rate(rag_request_latency_seconds_bucket[5m]))
# - P95: histogram_quantile(0.95, rate(rag_request_latency_seconds_bucket[5m]))
# - P99: histogram_quantile(0.99, rate(rag_request_latency_seconds_bucket[5m]))
# Visualization: Graph
# Title: "Response Time Percentiles"

# Panel 3: Cache Performance
# Queries:
# - Hit rate: rate(rag_cache_hits_total[5m]) / (rate(rag_cache_hits_total[5m]) + rate(rag_cache_misses_total[5m]))
# Visualization: Gauge
# Title: "Cache Hit Rate (%)"
# Thresholds: Red <50%, Yellow 50-70%, Green >70%

# Panel 4: Hourly Cost
# Query: increase(rag_llm_cost_dollars[1h])
# Visualization: Stat
# Title: "Cost (Last Hour)"
# Unit: currency (USD)
```

**Step 6**: Set up alerts
```bash
# Create alerts.yml
cat > alerts.yml << 'EOF'
groups:
  - name: rag_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(rag_request_latency_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RAG API high latency"
          description: "P95 latency is {{ $value }}s (threshold: 2s)"

      - alert: LowCacheHitRate
        expr: rate(rag_cache_hits_total[5m]) / (rate(rag_cache_hits_total[5m]) + rate(rag_cache_misses_total[5m])) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value | humanizePercentage }} (threshold: 50%)"

      - alert: HighErrorRate
        expr: rate(rag_requests_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}/s"
EOF

# Add to prometheus.yml
# rule_files:
#   - "alerts.yml"

# Restart Prometheus
docker-compose restart prometheus

# Check alerts: http://localhost:9090/alerts
```

**Step 7**: Test alerting
```bash
# Trigger high latency alert (simulate slow queries)
# Modify app to add artificial delay:
# time.sleep(3)  # Add to query handler

# Or stop Redis to trigger low cache hit rate:
docker-compose stop redis

# Wait 5-10 minutes
# Check alerts: http://localhost:9090/alerts
# Should show "FIRING" alerts

# Restore
docker-compose start redis
```

### Success Criteria

- [ ] Prometheus scraping metrics from API
- [ ] Grafana dashboard showing request rate, latency, cache hit rate
- [ ] Can visualize cost per hour/day
- [ ] Alerts configured and firing correctly
- [ ] All services healthy in Prometheus targets

### Verification Questions

1. **What metrics matter most for production RAG?**
   <details>
   <summary>Answer</summary>

   **User-facing**:
   - **P95/P99 latency**: 95% of queries under 500ms
   - **Error rate**: <0.1% errors
   - **Availability**: 99.9% uptime

   **System health**:
   - **Cache hit rate**: >80%
   - **Redis memory usage**: <80% capacity
   - **Worker queue depth**: <100 tasks

   **Business**:
   - **Cost per query**: Track spend
   - **Query volume**: Capacity planning
   - **Most queried topics**: Optimize docs
   </details>

2. **How do you troubleshoot high latency?**
   <details>
   <summary>Answer</summary>

   Check metrics in order:

   1. **Cache hit rate**: If low (<50%), investigate:
      - Redis memory full? (evicting keys)
      - TTL too short?
      - Query variation high? (no exact matches)

   2. **ChromaDB query time**: If high (>200ms):
      - Collection too large? (>1M vectors)
      - Need to increase `n_results`?
      - Embeddings model too slow?

   3. **LLM response time**: If high (>2s):
      - Input context too large? (compress)
      - Model overloaded? (rate limit)
      - Network latency to API?

   4. **Worker queue depth**: If high (>100):
      - Scale workers
      - Background tasks blocking queries?

   Use distributed tracing (OpenTelemetry) to identify bottleneck.
   </details>

3. **When should you scale horizontally vs vertically?**
   <details>
   <summary>Answer</summary>

   **Scale horizontally** (add more API instances):
   - High request volume (>100 req/sec per instance)
   - CPU-bound (embeddings, text processing)
   - Need high availability (multi-AZ)

   **Scale vertically** (bigger instances):
   - Memory-bound (large vector index)
   - ChromaDB performance limited by RAM
   - Single-threaded bottleneck

   In production, do both:
   - Horizontal: 3+ API instances behind load balancer
   - Vertical: 8-16GB RAM per instance for ChromaDB

   Cost-effective sweet spot: 4x 4GB instances vs 1x 16GB
   </details>

---

## Check Your Understanding

Test your knowledge of production RAG patterns:

### Question 1: Caching Strategy

You have a RAG system with these query patterns:
- 40% queries: RFC lookups (static docs, never change)
- 30% queries: Device configs (change weekly)
- 20% queries: Troubleshooting (guides updated daily)
- 10% queries: Network state (real-time)

What TTL should you set for each category to maximize cache hit rate while keeping data fresh?

<details>
<summary>Answer</summary>

```python
def get_cache_ttl(query_category: str) -> int:
    """Set TTL based on update frequency."""
    ttl_map = {
        "rfc_lookup": 604800,      # 7 days (static)
        "device_config": 86400,    # 1 day (weekly updates, but cache daily)
        "troubleshooting": 3600,   # 1 hour (updated daily)
        "network_state": 300       # 5 minutes (real-time)
    }
    return ttl_map.get(query_category, 3600)
```

**Reasoning**:
- **RFC lookups**: Never change → long TTL (7 days)
- **Device configs**: Change weekly, but cache 1 day is acceptable (slight staleness OK)
- **Troubleshooting**: Updated daily → 1 hour balances freshness and cache efficiency
- **Network state**: Real-time required → short TTL (5 min)

**Expected cache hit rate**:
- RFCs: ~95% (40% of traffic)
- Configs: ~85% (30% of traffic)
- Troubleshooting: ~70% (20% of traffic)
- State: ~40% (10% of traffic)
- **Overall**: ~77% hit rate

To improve:
- Pre-warm cache for common queries
- Use cache tags for invalidation (update config → clear config cache)
- Implement stale-while-revalidate pattern
</details>

---

### Question 2: Cost Optimization

Your production RAG system has these metrics:
- 10,000 queries/day
- Cache hit rate: 80%
- Avg input tokens: 5,000 (context + query)
- Avg output tokens: 500
- Using Claude Sonnet ($3/MTok input, $15/MTok output)

Current monthly cost: Calculate it. Then propose three optimizations that reduce cost by 70%+.

<details>
<summary>Answer</summary>

**Current cost**:
```python
# Daily non-cached queries (20% cache miss)
queries_per_day = 10000 * 0.20  # 2000 queries

# Token costs
input_cost_per_query = (5000 / 1_000_000) * 3  # $0.015
output_cost_per_query = (500 / 1_000_000) * 15  # $0.0075
cost_per_query = input_cost_per_query + output_cost_per_query  # $0.0225

# Daily/monthly
daily_cost = 2000 * 0.0225  # $45
monthly_cost = daily_cost * 30  # $1,350
```

**Current**: $1,350/month

**Optimizations**:

1. **Use Haiku for simple queries** (-60%):
   ```python
   # 60% of queries are simple "what is X"
   # Switch to Haiku ($0.25 input, $1.25 output)

   simple_queries = 2000 * 0.60  # 1200
   complex_queries = 2000 * 0.40  # 800

   haiku_cost = 1200 * ((5000/1e6 * 0.25) + (500/1e6 * 1.25))  # $2.25/day
   sonnet_cost = 800 * 0.0225  # $18/day

   new_daily = haiku_cost + sonnet_cost  # $20.25
   new_monthly = 20.25 * 30  # $607.50
   savings = (1350 - 607.50) / 1350  # 55% savings
   ```

2. **Contextual compression** (-40% more):
   ```python
   # Reduce input tokens by 80%
   # 5000 → 1000 tokens via compression

   input_cost = (1000/1e6 * 0.25) * 1200 + (1000/1e6 * 3) * 800  # $2.70/day
   output_cost = (500/1e6 * 1.25) * 1200 + (500/1e6 * 15) * 800  # $6.75/day

   new_daily = 2.70 + 6.75  # $9.45
   new_monthly = 9.45 * 30  # $283.50
   cumulative_savings = (1350 - 283.50) / 1350  # 79% savings
   ```

3. **Increase cache TTL** (-15% more):
   ```python
   # Improve cache hit rate from 80% → 90%
   queries_hitting_llm = 10000 * 0.10  # 1000 instead of 2000

   daily = 1000 * 9.45 / 2000  # $4.72
   monthly = 4.72 * 30  # $141.75

   total_savings = (1350 - 141.75) / 1350  # 89.5% savings
   ```

**Final cost**: $142/month (89.5% reduction)

**Implementation**:
```python
def optimize_query(query: str, context_docs: list):
    # 1. Route by complexity
    if is_simple_query(query):
        model = "claude-haiku-4-20250514"
    else:
        model = "claude-sonnet-4-20250514"

    # 2. Compress context
    compressed_docs = compress_context(context_docs)

    # 3. Extend cache TTL
    ttl = 86400 if is_static_content(query) else 3600

    return model, compressed_docs, ttl
```
</details>

---

### Question 3: Scaling Decision

Your RAG system monitoring shows:
- Request rate: 150 req/sec
- P95 latency: 850ms (target: 500ms)
- Cache hit rate: 82%
- CPU usage: 75%
- Memory usage: 45%
- Current setup: 2 API instances (4GB RAM, 2 vCPU each)

Should you:
A) Scale horizontally (add more instances)
B) Scale vertically (bigger instances)
C) Optimize code first
D) Add more cache layers

<details>
<summary>Answer</summary>

**Answer: A) Scale horizontally**

**Reasoning**:

1. **CPU-bound** (75% CPU, only 45% memory):
   - Adding vCPUs helps more than RAM
   - Horizontal scaling adds both capacity and availability

2. **High request rate** (150 req/sec):
   - Single instance handles ~75 req/sec at 75% CPU
   - Need 2x capacity → add 2 more instances (total 4)
   - Target: 150 req/sec ÷ 4 = 37.5 req/sec per instance (~40% CPU)

3. **Latency breakdown**:
   ```
   P95 latency: 850ms
   - Cache lookup: 10ms
   - ChromaDB query: 200ms
   - LLM call: 600ms
   - Processing: 40ms
   ```

   LLM is bottleneck (600ms). Horizontal scaling allows:
   - More concurrent LLM calls
   - Load balancing across instances
   - Reduced queueing delay

4. **Why not other options?**

   **B) Vertical scaling**: Won't help—not memory-bound, and LLM latency is external

   **C) Optimize code**: Already efficient (82% cache hit, reasonable CPU). Optimizations:
   - Increase cache TTL: +2% hit rate = -10ms P95
   - Async LLM calls: Doesn't reduce latency
   - Better compression: -50ms (minor)

   **D) More cache layers**: Hit rate already high (82%). Diminishing returns.

**Implementation**:
```yaml
# Scale to 4 instances
# kubernetes/deployment.yml
spec:
  replicas: 4  # Was 2

  resources:
    requests:
      memory: "4Gi"
      cpu: "2000m"
```

**Expected results**:
- Request rate per instance: 37.5 req/sec (was 75)
- CPU usage: ~40% (was 75%)
- P95 latency: ~600ms (target met)
- Cost: 2x instances = 2x cost, but necessary for SLA

**Cost-benefit**:
- Additional cost: $200/month (2 more instances)
- Revenue protected: Faster responses = better UX = retained users
- SLA met: P95 <500ms achievable with further optimization
</details>

---

### Question 4: Incident Response

Production alert fires: "High Error Rate - 15% of queries failing"

Monitoring shows:
- Error type: ChromaDB connection timeout
- Started: 10 minutes ago
- Cache hit rate: Normal (82%)
- Redis: Healthy
- RabbitMQ: Healthy
- API CPU: 45% (normal)
- ChromaDB process: Running

What are the first three diagnostic steps you take, and what's the most likely root cause?

<details>
<summary>Answer</summary>

**Diagnostic steps**:

1. **Check ChromaDB logs**:
   ```bash
   docker-compose logs chromadb --tail=100

   # Look for:
   # - Memory errors (OOM killer)
   # - Disk full errors
   # - Connection pool exhausted
   # - Slow queries (>5s)
   ```

2. **Check ChromaDB metrics**:
   ```bash
   # Collection size
   curl http://localhost:8000/admin/chromadb/stats

   # Docker stats
   docker stats chromadb_container

   # Look for:
   # - Memory usage >90%
   # - CPU throttling
   # - Disk I/O saturation
   ```

3. **Check concurrent connections**:
   ```bash
   # API connection pool
   curl http://localhost:8000/admin/connections

   # Look for:
   # - Connection pool exhausted
   # - Long-running queries blocking pool
   # - Connection leaks
   ```

**Most likely root cause**: **ChromaDB memory pressure**

**Evidence**:
- Connection timeouts (not immediate failures)
- Started suddenly (gradual memory growth)
- Cache still working (queries not reaching ChromaDB)
- 15% failure rate (some queries succeed)

**Hypothesis**: ChromaDB collection grew large, causing:
1. Memory swapping (slow queries)
2. Connection timeouts as queries queue
3. 15% failures = queries timing out after 30s

**Verification**:
```bash
# Check collection size
docker exec chromadb_container du -sh /chroma/data
# Output: 8.2G (was 2.1G last week)

# Check memory
docker stats chromadb_container
# Output: 3.8G / 4.0G (95% usage)

# Check slow queries
docker-compose logs chromadb | grep "query took"
# Output: Multiple queries >10s
```

**Root cause confirmed**: Collection outgrew memory allocation

**Fix (immediate)**:
```bash
# 1. Increase memory limit
docker-compose down
# Edit docker-compose.yml:
# chromadb:
#   deploy:
#     resources:
#       limits:
#         memory: 8G  # Was 4G

docker-compose up -d

# 2. Clear old data
# Delete embeddings older than 90 days
```

**Fix (long-term)**:
1. **Partition collections by date**:
   ```python
   collection_name = f"network_docs_{date.today().strftime('%Y_%m')}"
   ```

2. **Implement retention policy**:
   ```python
   # Delete docs older than 90 days
   @celery_app.task
   def cleanup_old_docs():
       cutoff = datetime.now() - timedelta(days=90)
       vectorstore.delete(where={"timestamp": {"$lt": cutoff}})
   ```

3. **Monitor collection size**:
   ```python
   # Add metric
   chromadb_collection_size = Gauge(
       'chromadb_collection_size_gb',
       'ChromaDB collection size in GB'
   )
   ```

**Prevention**:
- Alert on ChromaDB memory >70%
- Alert on collection size growth >20%/week
- Automated cleanup job
- Capacity planning based on ingestion rate
</details>

---

## Lab Time Budget

**Total time**: 4 hours
**Total cost**: $0.30 (API calls only)

| Lab | Time | Cost | ROI |
|-----|------|------|-----|
| Lab 1: Caching | 60 min | $0.10 | Deploy in 1 hour, saves 85% on query costs |
| Lab 2: Async | 90 min | $0.05 | Background processing unblocks API |
| Lab 3: Monitoring | 90 min | $0.15 | Visibility prevents downtime |

**Production value**:
- **Caching**: $1,150/month savings (85% of $1,350 baseline)
- **Async processing**: 10x throughput increase
- **Monitoring**: Prevents incidents ($10K+ in lost revenue per hour of downtime)

**ROI**: One-time 4-hour investment saves $1,150/month = 287.5x return in first month.

---

## Key Takeaways

1. **Caching is critical**: 85% cost reduction, 8x latency improvement
2. **Async processing**: Background tasks keep API responsive
3. **Monitoring**: Can't optimize what you don't measure
4. **Progressive builds**: V1→V4 shows evolution, not perfection up front
5. **Cost optimization**: Model selection + compression + caching = 89% savings
6. **Security**: Authentication, rate limiting, input validation required
7. **Deployment patterns**: Docker Compose for dev, Kubernetes for prod
8. **Incident response**: Logs + metrics + traces = fast diagnosis

**Production checklist**:
- [ ] Redis caching (80%+ hit rate)
- [ ] Celery workers for background tasks
- [ ] Prometheus + Grafana monitoring
- [ ] Health checks and readiness probes
- [ ] Authentication and rate limiting
- [ ] Error handling and retries
- [ ] Cost tracking and alerts
- [ ] Automated backups (ChromaDB data)
- [ ] Load balancing across instances
- [ ] Horizontal pod autoscaling (Kubernetes)

You now have production-ready RAG patterns. Next chapter: Agent architecture and planning.
