# Chapter 51: Scaling AI Systems

## Introduction

Your AI-powered config generator works great for 10 switches. You deploy it company-wide: 5,000 devices. Users submit 50 config generation requests simultaneously. Your single-threaded Python script processes them one at a time. Request 1 takes 3 seconds. Request 50 waits 147 seconds (almost 3 minutes) in queue before processing even starts.

**Users complain. System is "too slow."**

The problem isn't the AI—it's the architecture. Single-threaded processing doesn't scale. You need:
- **Queue systems** to buffer requests
- **Worker pools** to process in parallel
- **Batch processing** to handle bulk operations efficiently
- **Caching** to avoid redundant API calls
- **Database design** that scales to millions of records

This chapter shows you how to scale AI systems from 10 devices to 10,000+ devices, handling thousands of concurrent requests with sub-second response times.

**What You'll Build**:
- Queue-based architecture with Redis and Celery
- Parallel worker pools (process 50 requests simultaneously)
- Batch processing system (1,000 configs in 5 minutes vs. 50 minutes)
- High-performance database design (SQLite → PostgreSQL)
- Caching layer (80% cache hit rate = 5x speedup)
- Rate limiting and backpressure management
- Complete scalable architecture (handles 10,000 devices)

**Prerequisites**: Chapter 22 (Config Generation), Chapter 48 (Monitoring), Chapter 34 (Multi-Agent Systems)

---

## The Scaling Problem

### Performance at Different Scales

**Single-threaded processing (baseline)**:

| Devices | Requests/Day | Avg Processing Time | Total Time/Day | User Experience |
|---------|--------------|---------------------|----------------|-----------------|
| 10 | 50 | 3s | 2.5 min | ✓ Excellent |
| 100 | 500 | 3s | 25 min | ⚠️ Acceptable |
| 1,000 | 5,000 | 3s | 4.2 hours | ✗ Poor |
| 10,000 | 50,000 | 3s | 42 hours | ✗ Unusable |

**Problem**: At 10,000 devices, you can't even process one day's requests in a day.

**With parallel processing (10 workers)**:

| Devices | Requests/Day | Parallel Time | Total Time/Day | Improvement |
|---------|--------------|---------------|----------------|-------------|
| 10,000 | 50,000 | 3s | 4.2 hours | **10x faster** |

**With caching (80% cache hit)**:

| Devices | Cache Hits | API Calls | Total Time/Day | Improvement |
|---------|------------|-----------|----------------|-------------|
| 10,000 | 40,000 | 10,000 | 50 min | **50x faster** |

**Goal**: Handle 10,000 devices with <1 second latency for most requests.

---

## Architecture Pattern: Queue-Based Processing

Instead of processing requests synchronously, queue them and process with worker pools.

### Architecture Diagram

```
User Request
     │
     ▼
┌─────────────┐
│   API       │  (Fast: accepts request, returns job ID)
│   Server    │
└─────────────┘
     │
     ▼
┌─────────────┐
│   Redis     │  (Queue: holds pending requests)
│   Queue     │
└─────────────┘
     │
     ├──────┬──────┬──────┬──────┐
     ▼      ▼      ▼      ▼      ▼
  [Worker][Worker][Worker][Worker][Worker]  (10+ workers process in parallel)
     │      │      │      │      │
     └──────┴──────┴──────┴──────┘
              │
              ▼
        ┌──────────┐
        │PostgreSQL│  (Store results)
        └──────────┘
              │
              ▼
    User polls for result
```

**Benefits**:
- API responds instantly (queues request, doesn't wait for processing)
- Multiple workers process requests in parallel
- Automatic retry on failure
- Rate limiting at worker level (don't overwhelm API)
- Scalable (add more workers as needed)

---

## Implementation: Queue-Based System with Celery

### Setup

```bash
# Install dependencies
pip install celery redis sqlalchemy psycopg2-binary

# Start Redis (queue backend)
docker run -d -p 6379:6379 redis:latest

# Start PostgreSQL (result storage)
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:latest
```

### Implementation: Celery Worker

```python
"""
Celery Worker for Scalable AI Processing
File: scaling/celery_worker.py
"""
from celery import Celery
from anthropic import Anthropic
import os
import time
from typing import Dict

# Initialize Celery
app = Celery(
    'ai_worker',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Configure Celery
app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minute timeout
    worker_prefetch_multiplier=1,  # Don't prefetch tasks (better for long-running tasks)
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks (prevent memory leaks)
)

# Initialize Anthropic client
anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


@app.task(bind=True, max_retries=3, default_retry_delay=60)
def generate_config(self, device_name: str, requirements: str) -> Dict:
    """
    Generate network device configuration (Celery task).

    Args:
        device_name: Device hostname
        requirements: Config requirements

    Returns:
        Dict with generated config and metadata
    """
    try:
        print(f"[Worker {self.request.id[:8]}] Generating config for {device_name}")

        start_time = time.time()

        # Generate config using Claude
        response = anthropic_client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": f"""Generate network configuration for device: {device_name}

Requirements:
{requirements}

Return only configuration commands, no explanations."""
                }
            ]
        )

        config = response.content[0].text.strip()
        duration = time.time() - start_time

        print(f"[Worker {self.request.id[:8]}] Config generated in {duration:.2f}s")

        return {
            'device_name': device_name,
            'config': config,
            'status': 'success',
            'duration_seconds': duration,
            'tokens_used': response.usage.input_tokens + response.usage.output_tokens
        }

    except Exception as e:
        print(f"[Worker {self.request.id[:8]}] Error: {e}")

        # Retry up to 3 times
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

        return {
            'device_name': device_name,
            'status': 'error',
            'error': str(e)
        }


@app.task
def analyze_config(device_name: str, config: str) -> Dict:
    """Analyze network configuration for issues."""
    try:
        print(f"[Worker] Analyzing config for {device_name}")

        response = anthropic_client.messages.create(
            model='claude-3-haiku-20240307',  # Use faster model for analysis
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze this configuration for security issues and misconfigurations:

Device: {device_name}
Config:
{config[:2000]}

List critical issues only."""
                }
            ]
        )

        analysis = response.content[0].text.strip()

        return {
            'device_name': device_name,
            'analysis': analysis,
            'status': 'success'
        }

    except Exception as e:
        return {
            'device_name': device_name,
            'status': 'error',
            'error': str(e)
        }


@app.task
def batch_generate_configs(devices: list) -> Dict:
    """
    Generate configs for multiple devices in one task (batch processing).

    More efficient than individual tasks for bulk operations.
    """
    print(f"[Batch Worker] Processing {len(devices)} devices")

    results = []
    for device in devices:
        result = generate_config.apply(
            args=[device['name'], device['requirements']]
        ).get()
        results.append(result)

    return {
        'batch_size': len(devices),
        'results': results,
        'status': 'complete'
    }
```

### Implementation: API Server

```python
"""
API Server for Queued Processing
File: scaling/api_server.py
"""
from flask import Flask, request, jsonify
from celery_worker import generate_config, analyze_config
import uuid

app = Flask(__name__)


@app.route('/api/generate-config', methods=['POST'])
def api_generate_config():
    """
    Submit config generation request (returns immediately with job ID).

    Request body:
    {
        "device_name": "switch-access-01",
        "requirements": "Access switch with VLANs 10, 20, 30"
    }

    Response:
    {
        "job_id": "abc123...",
        "status": "queued"
    }
    """
    data = request.json

    device_name = data.get('device_name')
    requirements = data.get('requirements')

    if not device_name or not requirements:
        return jsonify({'error': 'Missing device_name or requirements'}), 400

    # Queue the task (non-blocking)
    task = generate_config.delay(device_name, requirements)

    return jsonify({
        'job_id': task.id,
        'status': 'queued',
        'device_name': device_name
    }), 202


@app.route('/api/job-status/<job_id>', methods=['GET'])
def api_job_status(job_id):
    """
    Check status of a queued job.

    Returns:
    {
        "job_id": "abc123...",
        "status": "pending" | "started" | "success" | "failure",
        "result": {...} (if complete)
    }
    """
    from celery.result import AsyncResult

    task = AsyncResult(job_id, app=generate_config.app)

    if task.state == 'PENDING':
        response = {
            'job_id': job_id,
            'status': 'pending'
        }
    elif task.state == 'STARTED':
        response = {
            'job_id': job_id,
            'status': 'started'
        }
    elif task.state == 'SUCCESS':
        response = {
            'job_id': job_id,
            'status': 'success',
            'result': task.result
        }
    else:  # FAILURE
        response = {
            'job_id': job_id,
            'status': 'failure',
            'error': str(task.info)
        }

    return jsonify(response)


@app.route('/api/bulk-generate', methods=['POST'])
def api_bulk_generate():
    """
    Submit bulk config generation (batch processing).

    Request body:
    {
        "devices": [
            {"name": "switch-01", "requirements": "..."},
            {"name": "switch-02", "requirements": "..."},
            ...
        ]
    }
    """
    data = request.json
    devices = data.get('devices', [])

    if not devices:
        return jsonify({'error': 'No devices provided'}), 400

    # Process in batches of 100 to avoid overwhelming workers
    batch_size = 100
    job_ids = []

    for i in range(0, len(devices), batch_size):
        batch = devices[i:i+batch_size]
        task = batch_generate_configs.delay(batch)
        job_ids.append(task.id)

    return jsonify({
        'batch_count': len(job_ids),
        'job_ids': job_ids,
        'total_devices': len(devices)
    }), 202


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### Usage Example

```python
"""
Client Usage Example
File: scaling/client_example.py
"""
import requests
import time

# Submit config generation request
response = requests.post('http://localhost:5000/api/generate-config', json={
    'device_name': 'switch-access-floor3-01',
    'requirements': 'Access switch with VLAN 10 (Data), VLAN 20 (Voice), trunk uplink on Gi0/48'
})

job_data = response.json()
job_id = job_data['job_id']

print(f"Job submitted: {job_id}")
print(f"Status: {job_data['status']}")

# Poll for completion
while True:
    time.sleep(2)

    status_response = requests.get(f'http://localhost:5000/api/job-status/{job_id}')
    status = status_response.json()

    print(f"Status: {status['status']}")

    if status['status'] == 'success':
        print("\nConfig generated:")
        print(status['result']['config'])
        break

    elif status['status'] == 'failure':
        print(f"Error: {status['error']}")
        break
```

### Starting Workers

```bash
# Start Celery worker (single worker)
celery -A celery_worker worker --loglevel=info

# Start multiple workers for parallel processing
celery -A celery_worker worker --concurrency=10 --loglevel=info

# Start worker on separate machine for distributed processing
celery -A celery_worker worker --hostname=worker2@%h --concurrency=10
```

**Performance**:
- Single worker: 3s per request = 20 requests/minute
- 10 workers: 3s per request × 10 = 200 requests/minute
- 100 workers: 2,000 requests/minute

---

## Batch Processing for Bulk Operations

For bulk operations (generate configs for 1,000 switches), batch processing is more efficient than individual tasks.

### Implementation: Batch Processor

```python
"""
Batch Processor for Bulk Operations
File: scaling/batch_processor.py
"""
from anthropic import Anthropic
from typing import List, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class BatchProcessor:
    """
    Process bulk operations efficiently with batching and parallelism.
    """

    def __init__(self, api_key: str, max_workers: int = 10, batch_size: int = 50):
        """
        Args:
            api_key: Anthropic API key
            max_workers: Number of parallel workers
            batch_size: Number of items per batch
        """
        self.client = Anthropic(api_key=api_key)
        self.max_workers = max_workers
        self.batch_size = batch_size

    def process_batch(self, items: List[Dict], process_func) -> List[Dict]:
        """
        Process a batch of items in parallel.

        Args:
            items: List of items to process
            process_func: Function to process each item

        Returns:
            List of results
        """
        results = []
        total_items = len(items)

        print(f"\nProcessing {total_items} items with {self.max_workers} workers...")

        start_time = time.time()

        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_func, item): item
                for item in items
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_item):
                item = future_to_item[future]

                try:
                    result = future.result()
                    results.append(result)

                except Exception as e:
                    print(f"Error processing {item}: {e}")
                    results.append({
                        'item': item,
                        'status': 'error',
                        'error': str(e)
                    })

                completed += 1

                # Progress indicator
                if completed % 100 == 0 or completed == total_items:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (total_items - completed) / rate if rate > 0 else 0

                    print(f"  Progress: {completed}/{total_items} "
                          f"({completed/total_items*100:.1f}%) "
                          f"Rate: {rate:.1f} items/sec "
                          f"ETA: {eta:.0f}s")

        duration = time.time() - start_time

        print(f"\n✓ Batch processing complete in {duration:.1f}s")
        print(f"  Throughput: {total_items/duration:.1f} items/sec")

        return results

    def generate_configs_bulk(self, devices: List[Dict]) -> List[Dict]:
        """
        Generate configs for multiple devices in parallel.

        Args:
            devices: List of dicts with 'name' and 'requirements'

        Returns:
            List of generated configs
        """
        def generate_single(device: Dict) -> Dict:
            """Generate config for single device."""
            try:
                response = self.client.messages.create(
                    model='claude-sonnet-4-20250514',
                    max_tokens=2000,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Generate config for {device['name']}

Requirements: {device['requirements']}

Return only configuration commands."""
                        }
                    ]
                )

                return {
                    'device': device['name'],
                    'config': response.content[0].text.strip(),
                    'status': 'success',
                    'tokens': response.usage.input_tokens + response.usage.output_tokens
                }

            except Exception as e:
                return {
                    'device': device['name'],
                    'status': 'error',
                    'error': str(e)
                }

        return self.process_batch(devices, generate_single)


# Example Usage
if __name__ == "__main__":
    import os

    # Create 1,000 test devices
    devices = [
        {
            'name': f'switch-floor{floor:02d}-{num:02d}',
            'requirements': f'Access switch for floor {floor}, VLANs 10, 20, 30'
        }
        for floor in range(1, 21)  # 20 floors
        for num in range(1, 51)    # 50 switches per floor
    ]

    print(f"Generated {len(devices)} test devices")

    processor = BatchProcessor(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_workers=20,  # 20 parallel workers
        batch_size=50
    )

    # Process all devices
    results = processor.generate_configs_bulk(devices)

    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    total_tokens = sum(r.get('tokens', 0) for r in results if r['status'] == 'success')

    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print('='*70)
    print(f"Total Devices: {len(devices)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total Tokens: {total_tokens:,}")
```

### Performance Comparison

**Sequential Processing (1,000 devices)**:
```
Processing time: 3s × 1,000 = 3,000s (50 minutes)
Throughput: 20 devices/minute
```

**Parallel Processing (10 workers)**:
```
Processing time: 3s × (1,000 / 10) = 300s (5 minutes)
Throughput: 200 devices/minute
Speedup: 10x
```

**Parallel Processing (20 workers)**:
```
Processing time: 150s (2.5 minutes)
Throughput: 400 devices/minute
Speedup: 20x
```

---

## Caching Layer for Performance

80% of requests are for the same data (common configs, topology queries). Cache them.

### Implementation: Redis Cache

```python
"""
Redis Caching Layer
File: scaling/cache_layer.py
"""
import redis
import json
import hashlib
from typing import Optional, Any
import time

class CacheLayer:
    """
    Redis-based caching for AI responses.

    Dramatically reduces API calls for repeated queries.
    """

    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379,
                 default_ttl: int = 3600):
        """
        Args:
            redis_host: Redis hostname
            redis_port: Redis port
            default_ttl: Default cache TTL in seconds (1 hour)
        """
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.default_ttl = default_ttl
        self.stats = {'hits': 0, 'misses': 0}

    def _generate_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model."""
        key_data = f"{model}:{prompt}"
        return f"cache:{hashlib.sha256(key_data.encode()).hexdigest()}"

    def get(self, prompt: str, model: str) -> Optional[str]:
        """
        Get cached response.

        Returns:
            Cached response if exists, None otherwise
        """
        cache_key = self._generate_cache_key(prompt, model)

        cached_value = self.redis.get(cache_key)

        if cached_value:
            self.stats['hits'] += 1
            return json.loads(cached_value)
        else:
            self.stats['misses'] += 1
            return None

    def set(self, prompt: str, model: str, response: Any, ttl: Optional[int] = None):
        """
        Cache a response.

        Args:
            prompt: The prompt that generated this response
            model: Model used
            response: Response to cache
            ttl: Time to live in seconds (default: 1 hour)
        """
        cache_key = self._generate_cache_key(prompt, model)
        ttl = ttl or self.default_ttl

        self.redis.setex(
            cache_key,
            ttl,
            json.dumps(response)
        )

    def invalidate(self, prompt: str, model: str):
        """Invalidate cached response."""
        cache_key = self._generate_cache_key(prompt, model)
        self.redis.delete(cache_key)

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.stats['hits'] + self.stats['misses']
        if total == 0:
            return 0.0
        return self.stats['hits'] / total

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': self.get_hit_rate(),
            'total_requests': self.stats['hits'] + self.stats['misses']
        }


class CachedAnthropicClient:
    """Anthropic client with caching."""

    def __init__(self, api_key: str, cache: CacheLayer):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.cache = cache

    def create_message(self, model: str, max_tokens: int, messages: list, **kwargs):
        """
        Create message with caching.

        If identical request was made recently, return cached response.
        """
        # Generate cache key from messages
        prompt = json.dumps(messages)

        # Check cache
        cached_response = self.cache.get(prompt, model)

        if cached_response:
            print(f"[CACHE HIT] Returning cached response")
            return cached_response

        # Cache miss - call API
        print(f"[CACHE MISS] Calling API")
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs
        )

        # Cache the response (convert to dict for serialization)
        cached_data = {
            'content': [{'text': response.content[0].text}],
            'usage': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            },
            'cached': True
        }

        self.cache.set(prompt, model, cached_data)

        return response


# Example Usage
if __name__ == "__main__":
    import os

    cache = CacheLayer()
    client = CachedAnthropicClient(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        cache=cache
    )

    # First request (cache miss)
    print("\nRequest 1:")
    start = time.time()
    response1 = client.create_message(
        model='claude-sonnet-4-20250514',
        max_tokens=500,
        messages=[{"role": "user", "content": "What is BGP?"}]
    )
    print(f"Duration: {time.time() - start:.3f}s")

    # Second identical request (cache hit)
    print("\nRequest 2 (identical):")
    start = time.time()
    response2 = client.create_message(
        model='claude-sonnet-4-20250514',
        max_tokens=500,
        messages=[{"role": "user", "content": "What is BGP?"}]
    )
    print(f"Duration: {time.time() - start:.3f}s")

    # Show cache statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']*100:.1f}%")
```

### Performance Impact

**Without Caching** (1,000 requests, 50% duplicates):
```
API Calls: 1,000
Average latency: 2.5s
Total time: 2,500s (42 minutes)
Cost: $15 (1,000 requests × $0.015)
```

**With Caching** (1,000 requests, 50% duplicates):
```
API Calls: 500 (50% cache hits)
Average latency: 1.25s (50% at 2.5s, 50% at <0.01s)
Total time: 625s (10 minutes)
Cost: $7.50 (500 requests × $0.015)

Speedup: 4x
Cost savings: 50%
```

---

## Database Design for Scale

SQLite works for 1,000 records. At 1M+ records, you need PostgreSQL with proper indexing.

### Implementation: Scalable Database

```python
"""
Scalable Database Design
File: scaling/database.py
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()


class ConfigRequest(Base):
    """Store config generation requests."""

    __tablename__ = 'config_requests'

    id = Column(Integer, primary_key=True)
    job_id = Column(String(64), unique=True, nullable=False, index=True)
    device_name = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    application = Column(String(255), nullable=False, index=True)
    status = Column(String(50), nullable=False, index=True)  # pending, processing, complete, error
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    cost_dollars = Column(Float, nullable=True)
    requirements = Column(Text, nullable=True)
    config = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_user_created', 'user_id', 'created_at'),
        Index('idx_status_created', 'status', 'created_at'),
        Index('idx_device_status', 'device_name', 'status'),
    )


class ScalableDatabase:
    """Database interface optimized for scale."""

    def __init__(self, connection_string: str):
        """
        Args:
            connection_string: SQLAlchemy connection string
                SQLite: 'sqlite:///configs.db'
                PostgreSQL: 'postgresql://user:pass@localhost/dbname'
        """
        self.engine = create_engine(connection_string, pool_size=20, max_overflow=40)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def create_request(self, job_id: str, device_name: str, user_id: str,
                      application: str, requirements: str) -> int:
        """Create a new config request."""
        session = self.Session()

        try:
            request = ConfigRequest(
                job_id=job_id,
                device_name=device_name,
                user_id=user_id,
                application=application,
                status='pending',
                requirements=requirements
            )

            session.add(request)
            session.commit()

            return request.id

        finally:
            session.close()

    def update_request(self, job_id: str, status: str, config: str = None,
                      error: str = None, duration: float = None,
                      tokens: dict = None, cost: float = None):
        """Update request status and results."""
        session = self.Session()

        try:
            request = session.query(ConfigRequest).filter_by(job_id=job_id).first()

            if request:
                request.status = status

                if status in ['complete', 'error']:
                    request.completed_at = datetime.utcnow()

                if config:
                    request.config = config

                if error:
                    request.error = error

                if duration:
                    request.duration_seconds = duration

                if tokens:
                    request.input_tokens = tokens.get('input', 0)
                    request.output_tokens = tokens.get('output', 0)

                if cost:
                    request.cost_dollars = cost

                session.commit()

        finally:
            session.close()

    def get_request(self, job_id: str) -> dict:
        """Get request by job ID."""
        session = self.Session()

        try:
            request = session.query(ConfigRequest).filter_by(job_id=job_id).first()

            if not request:
                return None

            return {
                'job_id': request.job_id,
                'device_name': request.device_name,
                'status': request.status,
                'config': request.config,
                'error': request.error,
                'created_at': request.created_at.isoformat(),
                'completed_at': request.completed_at.isoformat() if request.completed_at else None,
                'duration_seconds': request.duration_seconds,
                'tokens': {
                    'input': request.input_tokens,
                    'output': request.output_tokens
                },
                'cost_dollars': request.cost_dollars
            }

        finally:
            session.close()

    def get_requests_by_user(self, user_id: str, limit: int = 100) -> list:
        """Get recent requests for a user (uses index)."""
        session = self.Session()

        try:
            requests = session.query(ConfigRequest).filter_by(
                user_id=user_id
            ).order_by(
                ConfigRequest.created_at.desc()
            ).limit(limit).all()

            return [
                {
                    'job_id': r.job_id,
                    'device_name': r.device_name,
                    'status': r.status,
                    'created_at': r.created_at.isoformat()
                }
                for r in requests
            ]

        finally:
            session.close()

    def get_pending_count(self) -> int:
        """Get count of pending requests (for queue monitoring)."""
        session = self.Session()

        try:
            return session.query(ConfigRequest).filter_by(status='pending').count()

        finally:
            session.close()


# Example Usage
if __name__ == "__main__":
    # Use PostgreSQL for production scale
    db = ScalableDatabase('postgresql://user:password@localhost/ai_configs')

    # Create a request
    job_id = 'test-job-123'
    request_id = db.create_request(
        job_id=job_id,
        device_name='switch-01',
        user_id='john.doe',
        application='config-generator',
        requirements='Standard access switch config'
    )

    print(f"Created request: {request_id}")

    # Update when complete
    db.update_request(
        job_id=job_id,
        status='complete',
        config='interface Gi0/1\n switchport mode access\n...',
        duration=2.3,
        tokens={'input': 1000, 'output': 500},
        cost=0.012
    )

    # Retrieve
    request = db.get_request(job_id)
    print(f"\nRequest: {json.dumps(request, indent=2)}")
```

### Performance: SQLite vs PostgreSQL

**Query Performance (1M records)**:

| Operation | SQLite | PostgreSQL (indexed) | Speedup |
|-----------|--------|---------------------|---------|
| Get by job_id | 0.8s | 0.002s | 400x |
| Get by user (recent 100) | 12s | 0.015s | 800x |
| Count pending | 5s | 0.001s | 5000x |

**Why PostgreSQL Wins**:
- Better indexing (B-tree indexes optimized for large datasets)
- Connection pooling (handles concurrent requests)
- Advanced query optimizer
- Parallel query execution

---

## Rate Limiting and Backpressure

Prevent overwhelming the API with too many concurrent requests.

### Implementation: Rate Limiter

```python
"""
Rate Limiting for API Calls
File: scaling/rate_limiter.py
"""
import time
from threading import Lock
from collections import deque

class RateLimiter:
    """
    Token bucket rate limiter.

    Prevents exceeding API rate limits.
    """

    def __init__(self, requests_per_minute: int = 100):
        """
        Args:
            requests_per_minute: Max requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.requests = deque()
        self.lock = Lock()

    def acquire(self):
        """
        Acquire permission to make a request.

        Blocks if rate limit would be exceeded.
        """
        with self.lock:
            now = time.time()

            # Remove requests older than 1 minute
            cutoff = now - 60
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()

            # Check if we're at the limit
            if len(self.requests) >= self.requests_per_minute:
                # Calculate wait time
                oldest = self.requests[0]
                wait_time = 60 - (now - oldest)

                if wait_time > 0:
                    print(f"[RATE LIMIT] Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)

                    # Retry
                    return self.acquire()

            # Record this request
            self.requests.append(now)


# Example: Wrap API client with rate limiting
class RateLimitedClient:
    """API client with rate limiting."""

    def __init__(self, client, rate_limiter: RateLimiter):
        self.client = client
        self.rate_limiter = rate_limiter

    def create_message(self, *args, **kwargs):
        """Create message with rate limiting."""
        # Acquire rate limit token (blocks if necessary)
        self.rate_limiter.acquire()

        # Make API call
        return self.client.create_message(*args, **kwargs)
```

---

## Complete Scalable Architecture

Putting it all together: production-ready scalable system.

### Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│                      Load Balancer                        │
│                     (nginx/HAProxy)                       │
└───────────────────────────────────────────────────────────┘
         │                   │                  │
         ▼                   ▼                  ▼
┌─────────────┐      ┌─────────────┐    ┌─────────────┐
│ API Server  │      │ API Server  │    │ API Server  │
│   (Flask)   │      │   (Flask)   │    │   (Flask)   │
└─────────────┘      └─────────────┘    └─────────────┘
         │                   │                  │
         └───────────────────┼──────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  Redis Queue    │
                    └─────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  ┌──────────┐        ┌──────────┐       ┌──────────┐
  │ Worker 1 │        │ Worker 2 │  ...  │Worker 50 │
  └──────────┘        └──────────┘       └──────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   + Read Replicas│
                    └─────────────────┘
```

**Capacity**:
- 3 API servers: Handle 3,000 requests/second
- 50 workers: Process 500 requests/minute (parallel)
- Redis queue: Buffer 10,000 pending requests
- PostgreSQL: Store 10M+ records efficiently
- Cache: 80% hit rate = 5x effective throughput

**Handles**:
- 10,000 devices
- 50,000 requests/day
- <1s latency for 95% of requests
- $500/month API costs (with caching)

---

## Summary

You now know how to scale AI systems from 10 to 10,000+ devices:

1. **Queue-Based Architecture**: Celery + Redis for async processing
2. **Parallel Processing**: Worker pools (10-50 workers) for 10-50x speedup
3. **Batch Processing**: Efficient bulk operations (1,000 configs in 5 minutes)
4. **Caching**: 80% cache hit rate = 5x effective throughput
5. **Database Design**: PostgreSQL with indexes for 1M+ records
6. **Rate Limiting**: Prevent overwhelming APIs

**Key Results**:
- **Throughput**: From 20 requests/min → 400 requests/min (20x)
- **Latency**: From 50s wait → <1s response (50x)
- **Cost**: 50% reduction with caching
- **Scale**: Handle 10,000 devices reliably

**Production Deployment**:
```
10 devices     → Single-threaded Python script (OK)
100 devices    → Add Redis queue + 5 workers
1,000 devices  → 20 workers + caching + PostgreSQL
10,000 devices → Load balancer + 50 workers + read replicas
```

**Next Chapter**: Complete case study tying everything together—building a real NetOps AI system with architecture, code, deployment, and 6 months of lessons learned.

---

## What Can Go Wrong?

**1. Workers get stuck processing (queue backs up)**
- **Cause**: Worker hangs on API call, never releases
- **Fix**: Set task timeouts (5 minutes max), restart workers regularly

**2. Cache returns stale data**
- **Cause**: Config changed but cache wasn't invalidated
- **Fix**: Set appropriate TTL (1 hour for stable data), invalidate on updates

**3. Database connection pool exhausted**
- **Cause**: Too many workers trying to connect simultaneously
- **Fix**: Increase pool size, use connection pooling (PgBouncer)

**4. Queue memory overflow (too many pending requests)**
- **Cause**: Workers can't keep up with request rate
- **Fix**: Add more workers, implement backpressure (reject requests when queue > 10K)

**5. Parallel processing causes API rate limits**
- **Cause**: 50 workers × 1 request/s = 50 requests/s (exceeds limit)
- **Fix**: Implement rate limiting at worker level, add delays between requests

**6. Cache causes consistency issues**
- **Cause**: Multiple workers updating same data with cached reads
- **Fix**: Use cache-aside pattern, invalidate on writes, shorter TTL for mutable data

**7. PostgreSQL becomes bottleneck**
- **Cause**: All workers write to single DB instance
- **Fix**: Add read replicas, partition data, optimize queries

**Code for this chapter**: `github.com/vexpertai/ai-networking-book/chapter-51/`
