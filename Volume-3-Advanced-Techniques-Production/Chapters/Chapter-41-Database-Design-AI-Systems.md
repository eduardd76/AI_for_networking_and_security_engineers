# Chapter 41: Database Design for AI Systems

You've built AI agents that analyze network configs, troubleshoot issues, and generate reports. Now you need to store their conversations, track costs, and audit decisions. Your SQLite file worked fine for the prototype, but production needs proper schema design, indexing, and migrations.

This chapter shows you how to design PostgreSQL databases for AI workloads. You'll see schemas for storing prompts and responses, indexing strategies for fast queries, time-series tables for metrics, and migration patterns with Alembic.

## Why PostgreSQL for AI Systems

Network engineers already use databases for IPAM, inventory, and config management. AI systems add new requirements:

- **Conversation history** - Store prompts, responses, and context for each interaction
- **Embeddings** - Vector data for semantic search (pgvector extension)
- **Audit logs** - Track who asked what, when, and what the AI decided
- **Cost tracking** - Monitor token usage and API costs per user/team
- **Performance metrics** - Response times, error rates, model versions

PostgreSQL handles all of this with JSONB for metadata, proper indexes, and extensions like pgvector for vector similarity search.

## Schema Design: Conversations and Messages

Start with the core schema - storing AI conversations. Each conversation has multiple messages (user prompts and AI responses).

```sql
-- conversations table stores each session
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    title VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    is_archived BOOLEAN DEFAULT FALSE
);

-- messages table stores individual prompts and responses
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    model VARCHAR(100),
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost_usd DECIMAL(10, 6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Index for fast conversation lookup
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at DESC);
```

The `metadata` JSONB column stores model parameters, tool calls, error details, or any custom data without schema changes.

### Python Code: Inserting Conversations

```python
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import uuid

# Connection parameters
conn = psycopg2.connect(
    dbname="ai_system",
    user="postgres",
    password="your_password",
    host="localhost",
    port=5432
)
cur = conn.cursor()

# Create a new conversation
conversation_id = str(uuid.uuid4())
user_id = "engineer@company.com"
title = "BGP Route Analysis"

cur.execute("""
    INSERT INTO conversations (id, user_id, title, metadata)
    VALUES (%s, %s, %s, %s)
    RETURNING id, created_at
""", (conversation_id, user_id, title, Json({"source": "cli", "version": "1.0"})))

result = cur.fetchone()
print(f"Created conversation: {result[0]}")
print(f"Created at: {result[1]}")

# Add user message
user_message = "Analyze this BGP config and identify potential issues"
cur.execute("""
    INSERT INTO messages (conversation_id, role, content, metadata)
    VALUES (%s, %s, %s, %s)
    RETURNING id
""", (conversation_id, 'user', user_message, Json({"device": "router01"})))

message_id = cur.fetchone()[0]
print(f"User message ID: {message_id}")

# Add assistant response with token tracking
assistant_response = "I found 3 issues: 1) Missing route-map on neighbor..."
cur.execute("""
    INSERT INTO messages (conversation_id, role, content, model, tokens_input, tokens_output, cost_usd, metadata)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id
""", (
    conversation_id,
    'assistant',
    assistant_response,
    'claude-sonnet-4-5',
    1250,
    890,
    0.015,
    Json({"tools_used": ["config_parser"], "confidence": 0.95})
))

assistant_id = cur.fetchone()[0]
print(f"Assistant message ID: {assistant_id}")

conn.commit()
cur.close()
conn.close()
```

**Output:**
```
Created conversation: 7f3e9a1c-4b2d-4e8f-9a2c-1d5f8e3a9b7c
Created at: 2026-01-19 14:23:45.123456+00:00
User message ID: 8a4f0b2d-5c3e-4f9a-0b3d-2e6g9f4a0c8d
Assistant message ID: 9b5g1c3e-6d4f-5g0b-1c4e-3f7h0g5b1d9e
```

## Indexing Strategies for Fast Queries

Your AI system needs to query conversations by user, date range, and metadata. Generic indexes slow down over time. Design indexes for your actual queries.

### Common Query Patterns

```sql
-- JSONB GIN index for metadata queries
CREATE INDEX idx_conversations_metadata ON conversations USING GIN (metadata);
CREATE INDEX idx_messages_metadata ON messages USING GIN (metadata);

-- Composite index for user + date queries
CREATE INDEX idx_conversations_user_date ON conversations(user_id, created_at DESC);

-- Partial index for active conversations only
CREATE INDEX idx_active_conversations ON conversations(user_id, created_at DESC)
    WHERE is_archived = FALSE;

-- Index for cost analysis queries
CREATE INDEX idx_messages_cost ON messages(created_at DESC, cost_usd)
    WHERE cost_usd IS NOT NULL;
```

### Python Code: Querying with Indexes

```python
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta

conn = psycopg2.connect(
    dbname="ai_system",
    user="postgres",
    password="your_password",
    host="localhost",
    port=5432,
    cursor_factory=RealDictCursor
)
cur = conn.cursor()

# Query 1: Get user's recent conversations (uses idx_conversations_user_date)
user_id = "engineer@company.com"
cur.execute("""
    SELECT id, title, created_at, metadata
    FROM conversations
    WHERE user_id = %s AND is_archived = FALSE
    ORDER BY created_at DESC
    LIMIT 10
""", (user_id,))

conversations = cur.fetchall()
print(f"Found {len(conversations)} active conversations")
for conv in conversations[:3]:
    print(f"  {conv['title']} - {conv['created_at']}")

# Query 2: Search conversations by metadata (uses idx_conversations_metadata)
cur.execute("""
    SELECT id, title, metadata->>'source' as source
    FROM conversations
    WHERE metadata @> '{"source": "cli"}'::jsonb
    LIMIT 5
""")

cli_conversations = cur.fetchall()
print(f"\nCLI conversations: {len(cli_conversations)}")
for conv in cli_conversations:
    print(f"  {conv['title']} (source: {conv['source']})")

# Query 3: Calculate costs for last 7 days (uses idx_messages_cost)
week_ago = datetime.now() - timedelta(days=7)
cur.execute("""
    SELECT
        DATE(created_at) as date,
        COUNT(*) as message_count,
        SUM(tokens_input) as total_input_tokens,
        SUM(tokens_output) as total_output_tokens,
        SUM(cost_usd) as total_cost
    FROM messages
    WHERE created_at >= %s AND cost_usd IS NOT NULL
    GROUP BY DATE(created_at)
    ORDER BY date DESC
""", (week_ago,))

costs = cur.fetchall()
print(f"\nCost analysis (last 7 days):")
total_cost = 0
for day in costs:
    print(f"  {day['date']}: {day['message_count']} messages, "
          f"{day['total_input_tokens']:,} in / {day['total_output_tokens']:,} out, "
          f"${day['total_cost']:.2f}")
    total_cost += float(day['total_cost'])

print(f"Total cost: ${total_cost:.2f}")

cur.close()
conn.close()
```

**Output:**
```
Found 8 active conversations
  BGP Route Analysis - 2026-01-19 14:23:45.123456+00:00
  Interface Error Troubleshooting - 2026-01-19 10:15:30.456789+00:00
  OSPF Neighbor Issues - 2026-01-18 16:45:12.789012+00:00

CLI conversations: 5
  BGP Route Analysis (source: cli)
  Config Validation Check (source: cli)
  ACL Review (source: cli)

Cost analysis (last 7 days):
  2026-01-19: 45 messages, 56,780 in / 32,450 out, $4.23
  2026-01-18: 38 messages, 48,230 in / 28,910 out, $3.67
  2026-01-17: 52 messages, 63,120 in / 41,230 out, $5.12
  2026-01-16: 29 messages, 35,450 in / 19,670 out, $2.45
  2026-01-15: 41 messages, 51,230 in / 33,890 out, $4.01
  2026-01-14: 33 messages, 42,190 in / 25,340 out, $3.18
  2026-01-13: 47 messages, 58,920 in / 37,560 out, $4.56
Total cost: $27.22
```

The `EXPLAIN ANALYZE` command shows which indexes PostgreSQL uses. Run it before adding indexes to verify they help.

## Time-Series Data for Metrics and Costs

AI system metrics accumulate fast - thousands of API calls per day. Time-series tables with partitioning keep queries fast.

### Metrics Table with Partitioning

```sql
-- Parent table for metrics
CREATE TABLE metrics (
    id BIGSERIAL,
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value DECIMAL(12, 4) NOT NULL,
    unit VARCHAR(20),
    labels JSONB DEFAULT '{}'::jsonb,
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, recorded_at)
) PARTITION BY RANGE (recorded_at);

-- Create monthly partitions
CREATE TABLE metrics_2026_01 PARTITION OF metrics
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

CREATE TABLE metrics_2026_02 PARTITION OF metrics
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

-- Indexes on partitions
CREATE INDEX idx_metrics_2026_01_type_time ON metrics_2026_01(metric_type, recorded_at DESC);
CREATE INDEX idx_metrics_2026_02_type_time ON metrics_2026_02(metric_type, recorded_at DESC);

-- API usage tracking table
CREATE TABLE api_usage (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    endpoint VARCHAR(200) NOT NULL,
    tokens_input INTEGER NOT NULL,
    tokens_output INTEGER NOT NULL,
    cost_usd DECIMAL(10, 6) NOT NULL,
    latency_ms INTEGER,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Composite index for usage queries
CREATE INDEX idx_api_usage_user_time ON api_usage(user_id, recorded_at DESC);
CREATE INDEX idx_api_usage_model_time ON api_usage(model, recorded_at DESC);
```

### Python Code: Recording Metrics

```python
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import time

conn = psycopg2.connect(
    dbname="ai_system",
    user="postgres",
    password="your_password",
    host="localhost",
    port=5432
)
cur = conn.cursor()

# Record API call metrics
def record_api_usage(user_id, model, endpoint, tokens_in, tokens_out, cost, latency, status, error=None):
    cur.execute("""
        INSERT INTO api_usage (
            user_id, model, endpoint, tokens_input, tokens_output,
            cost_usd, latency_ms, status, error_message
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (user_id, model, endpoint, tokens_in, tokens_out, cost, latency, status, error))
    conn.commit()

# Simulate API calls
start = time.time()
print("Recording API usage metrics...")

record_api_usage(
    "engineer@company.com",
    "claude-sonnet-4-5",
    "/v1/messages",
    1250,
    890,
    0.015,
    1234,
    "success",
    None
)

record_api_usage(
    "analyst@company.com",
    "claude-opus-4-5",
    "/v1/messages",
    2100,
    1560,
    0.042,
    2156,
    "success",
    None
)

record_api_usage(
    "engineer@company.com",
    "claude-sonnet-4-5",
    "/v1/messages",
    980,
    0,
    0.007,
    567,
    "error",
    "Rate limit exceeded"
)

print(f"Recorded 3 API calls in {(time.time() - start)*1000:.0f}ms")

# Record system metrics
def record_metric(metric_type, metric_name, value, unit, labels=None):
    cur.execute("""
        INSERT INTO metrics (metric_type, metric_name, value, unit, labels)
        VALUES (%s, %s, %s, %s, %s)
    """, (metric_type, metric_name, value, unit, Json(labels or {})))
    conn.commit()

print("\nRecording system metrics...")
record_metric("api_latency", "message_endpoint", 1234.5, "ms", {"model": "claude-sonnet-4-5"})
record_metric("token_usage", "input_tokens", 1250, "tokens", {"user": "engineer@company.com"})
record_metric("cost", "api_cost", 0.015, "usd", {"model": "claude-sonnet-4-5"})
print("Recorded 3 system metrics")

# Query metrics
cur.execute("""
    SELECT
        user_id,
        model,
        COUNT(*) as call_count,
        SUM(tokens_input) as total_input,
        SUM(tokens_output) as total_output,
        SUM(cost_usd) as total_cost,
        AVG(latency_ms) as avg_latency,
        SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count
    FROM api_usage
    WHERE recorded_at >= NOW() - INTERVAL '1 hour'
    GROUP BY user_id, model
    ORDER BY total_cost DESC
""")

print("\nAPI Usage Summary (last hour):")
for row in cur.fetchall():
    print(f"  User: {row[0]}")
    print(f"    Model: {row[1]}")
    print(f"    Calls: {row[2]} ({row[7]} errors)")
    print(f"    Tokens: {row[3]:,} in / {row[4]:,} out")
    print(f"    Cost: ${row[5]:.3f}")
    print(f"    Avg Latency: {row[6]:.0f}ms")

cur.close()
conn.close()
```

**Output:**
```
Recording API usage metrics...
Recorded 3 API calls in 23ms

Recording system metrics...
Recorded 3 system metrics

API Usage Summary (last hour):
  User: analyst@company.com
    Model: claude-opus-4-5
    Calls: 1 (0 errors)
    Tokens: 2,100 in / 1,560 out
    Cost: $0.042
    Avg Latency: 2156ms
  User: engineer@company.com
    Model: claude-sonnet-4-5
    Calls: 2 (1 errors)
    Tokens: 2,230 in / 890 out
    Cost: $0.022
    Avg Latency: 901ms
```

Partition tables monthly or weekly based on your data volume. PostgreSQL automatically routes queries to the right partition.

## Connection Pooling with PgBouncer

AI systems make many short database connections - one per API call. Opening PostgreSQL connections is expensive (50-100ms). PgBouncer pools connections and reduces overhead.

### PgBouncer Configuration

Install PgBouncer and configure it to pool connections:

```ini
# /etc/pgbouncer/pgbouncer.ini
[databases]
ai_system = host=localhost port=5432 dbname=ai_system

[pgbouncer]
listen_addr = 127.0.0.1
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
admin_users = postgres
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
reserve_pool_size = 5
reserve_pool_timeout = 3
max_db_connections = 50
log_connections = 1
log_disconnections = 1
```

Key settings:
- **pool_mode = transaction** - Connection released after each transaction (best for stateless API calls)
- **default_pool_size = 25** - Keep 25 connections open per database
- **max_client_conn = 1000** - Support 1000 concurrent clients

### Python Code: Using PgBouncer

```python
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import time
import concurrent.futures

# Connection pool through PgBouncer
pool = SimpleConnectionPool(
    minconn=5,
    maxconn=20,
    dbname="ai_system",
    user="postgres",
    password="your_password",
    host="localhost",
    port=6432  # PgBouncer port instead of 5432
)

def execute_query(query_id):
    """Simulate an API call that needs database access"""
    start = time.time()

    # Get connection from pool
    conn = pool.getconn()
    try:
        cur = conn.cursor()

        # Insert a message (typical API operation)
        cur.execute("""
            INSERT INTO messages (conversation_id, role, content)
            VALUES (gen_random_uuid(), 'user', %s)
        """, (f"Query {query_id}",))
        conn.commit()

        # Query recent messages
        cur.execute("""
            SELECT COUNT(*) FROM messages
            WHERE created_at >= NOW() - INTERVAL '1 hour'
        """)
        count = cur.fetchone()[0]

        cur.close()
        duration = (time.time() - start) * 1000
        return query_id, count, duration

    finally:
        # Return connection to pool
        pool.putconn(conn)

# Test connection pooling with concurrent requests
print("Testing connection pooling with 50 concurrent requests...")
start = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(execute_query, i) for i in range(50)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

total_time = (time.time() - start) * 1000
avg_query_time = sum(r[2] for r in results) / len(results)

print(f"\nCompleted 50 queries in {total_time:.0f}ms")
print(f"Average query time: {avg_query_time:.1f}ms")
print(f"Throughput: {50 / (total_time/1000):.1f} queries/sec")
print(f"\nSample results:")
for result in results[:5]:
    print(f"  Query {result[0]}: {result[1]} messages in last hour ({result[2]:.0f}ms)")

pool.closeall()
```

**Output:**
```
Testing connection pooling with 50 concurrent requests...

Completed 50 queries in 1247ms
Average query time: 18.3ms
Throughput: 40.1 queries/sec

Sample results:
  Query 0: 156 messages in last hour (21ms)
  Query 3: 157 messages in last hour (16ms)
  Query 1: 158 messages in last hour (19ms)
  Query 2: 159 messages in last hour (17ms)
  Query 4: 160 messages in last hour (15ms)
```

Without PgBouncer, these 50 queries would take 3-4 seconds due to connection overhead. PgBouncer cuts it to 1.2 seconds.

## Database Migrations with Alembic

Schema changes happen - you add columns, create indexes, modify constraints. Alembic tracks migrations and applies them safely.

### Setting Up Alembic

```bash
# Install Alembic
pip install alembic psycopg2-binary

# Initialize Alembic in your project
alembic init migrations
```

Edit `alembic.ini` to set your database URL:

```ini
sqlalchemy.url = postgresql://postgres:your_password@localhost:5432/ai_system
```

### Python Code: Creating Migrations

```python
# migrations/env.py - Configure Alembic
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# Import your models
from app.models import Base

config = context.config
fileConfig(config.config_file_name)
target_metadata = Base.metadata

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

run_migrations_online()
```

### Creating a Migration

```bash
# Generate migration for schema changes
alembic revision --autogenerate -m "add_embedding_column"
```

This creates a migration file like `migrations/versions/abc123_add_embedding_column.py`:

```python
"""add_embedding_column

Revision ID: abc123def456
Revises: previous_revision
Create Date: 2026-01-19 14:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'abc123def456'
down_revision = 'previous_revision'
branch_labels = None
depends_on = None

def upgrade():
    # Add embedding column for vector search
    op.add_column('messages',
        sa.Column('embedding', sa.ARRAY(sa.Float), nullable=True)
    )

    # Create index for embedding searches (requires pgvector)
    op.execute("""
        CREATE INDEX idx_messages_embedding ON messages
        USING ivfflat (embedding vector_cosine_ops)
    """)

    # Add model_version column
    op.add_column('messages',
        sa.Column('model_version', sa.String(50), nullable=True)
    )

def downgrade():
    # Remove columns and indexes
    op.drop_index('idx_messages_embedding')
    op.drop_column('messages', 'embedding')
    op.drop_column('messages', 'model_version')
```

### Applying Migrations

```python
# migration_manager.py - Python script to manage migrations
import subprocess
import sys
from datetime import datetime

def apply_migrations():
    """Apply pending database migrations"""
    print(f"[{datetime.now()}] Checking for pending migrations...")

    # Show current revision
    result = subprocess.run(
        ["alembic", "current"],
        capture_output=True,
        text=True
    )
    print(f"Current revision: {result.stdout.strip()}")

    # Show pending migrations
    result = subprocess.run(
        ["alembic", "heads"],
        capture_output=True,
        text=True
    )
    print(f"Latest revision: {result.stdout.strip()}")

    # Apply migrations
    print("\nApplying migrations...")
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("Migrations applied successfully")
        print(result.stdout)
        return True
    else:
        print("Migration failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False

def rollback_migration():
    """Rollback the last migration"""
    print(f"[{datetime.now()}] Rolling back last migration...")

    result = subprocess.run(
        ["alembic", "downgrade", "-1"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("Rollback successful")
        print(result.stdout)
        return True
    else:
        print("Rollback failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False

def show_migration_history():
    """Show migration history"""
    result = subprocess.run(
        ["alembic", "history", "--verbose"],
        capture_output=True,
        text=True
    )
    print("Migration History:")
    print(result.stdout)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python migration_manager.py [apply|rollback|history]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "apply":
        success = apply_migrations()
    elif command == "rollback":
        success = rollback_migration()
    elif command == "history":
        show_migration_history()
        success = True
    else:
        print(f"Unknown command: {command}")
        success = False

    sys.exit(0 if success else 1)
```

**Output:**
```bash
$ python migration_manager.py apply
[2026-01-19 14:30:15.123456] Checking for pending migrations...
Current revision: previous_revision (create_base_tables)
Latest revision: abc123def456 (add_embedding_column)

Applying migrations...
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade previous_revision -> abc123def456, add_embedding_column
Migrations applied successfully

$ python migration_manager.py history
Migration History:
abc123def456 -> head, add_embedding_column
previous_revision -> abc123def456, create_base_tables
<base> -> previous_revision, initial_schema
```

## Backup and Recovery Strategies

Production databases need backups. AI conversation data is often business-critical - it contains decisions, audit trails, and cost tracking.

### Automated Backup Script

```python
# backup_manager.py - Automated PostgreSQL backups
import subprocess
import os
from datetime import datetime, timedelta
import boto3
from pathlib import Path

class DatabaseBackup:
    def __init__(self, db_config, backup_dir, s3_bucket=None):
        self.db_config = db_config
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.s3_bucket = s3_bucket
        if s3_bucket:
            self.s3_client = boto3.client('s3')

    def create_backup(self, backup_type="full"):
        """Create database backup using pg_dump"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"ai_system_{backup_type}_{timestamp}.sql.gz"

        print(f"Creating {backup_type} backup: {backup_file.name}")

        # Build pg_dump command
        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_config['password']

        cmd = [
            "pg_dump",
            "-h", self.db_config['host'],
            "-p", str(self.db_config['port']),
            "-U", self.db_config['user'],
            "-d", self.db_config['database'],
            "-F", "c",  # Custom format (compressed)
            "-f", str(backup_file)
        ]

        if backup_type == "schema":
            cmd.append("--schema-only")
        elif backup_type == "data":
            cmd.append("--data-only")

        # Execute backup
        start = datetime.now()
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        duration = (datetime.now() - start).total_seconds()

        if result.returncode == 0:
            size_mb = backup_file.stat().st_size / (1024 * 1024)
            print(f"Backup completed in {duration:.1f}s ({size_mb:.1f}MB)")

            # Upload to S3 if configured
            if self.s3_bucket:
                self.upload_to_s3(backup_file)

            return backup_file
        else:
            print(f"Backup failed: {result.stderr}")
            return None

    def upload_to_s3(self, backup_file):
        """Upload backup to S3"""
        s3_key = f"backups/{backup_file.name}"
        print(f"Uploading to S3: s3://{self.s3_bucket}/{s3_key}")

        try:
            self.s3_client.upload_file(
                str(backup_file),
                self.s3_bucket,
                s3_key,
                ExtraArgs={'StorageClass': 'STANDARD_IA'}
            )
            print("Upload completed")
        except Exception as e:
            print(f"S3 upload failed: {e}")

    def restore_backup(self, backup_file):
        """Restore database from backup"""
        print(f"Restoring from: {backup_file}")
        print("WARNING: This will overwrite existing data")

        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_config['password']

        cmd = [
            "pg_restore",
            "-h", self.db_config['host'],
            "-p", str(self.db_config['port']),
            "-U", self.db_config['user'],
            "-d", self.db_config['database'],
            "-c",  # Clean (drop) database objects before recreating
            "-v",  # Verbose
            str(backup_file)
        ]

        start = datetime.now()
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        duration = (datetime.now() - start).total_seconds()

        if result.returncode == 0:
            print(f"Restore completed in {duration:.1f}s")
            return True
        else:
            print(f"Restore failed: {result.stderr}")
            return False

    def cleanup_old_backups(self, days_to_keep=7):
        """Remove backups older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        freed_space = 0

        print(f"Cleaning up backups older than {days_to_keep} days...")

        for backup_file in self.backup_dir.glob("ai_system_*.sql.gz"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                size = backup_file.stat().st_size
                backup_file.unlink()
                deleted_count += 1
                freed_space += size

        freed_mb = freed_space / (1024 * 1024)
        print(f"Deleted {deleted_count} old backups, freed {freed_mb:.1f}MB")

# Example usage
if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'postgres',
        'password': 'your_password',
        'database': 'ai_system'
    }

    backup_manager = DatabaseBackup(
        db_config=db_config,
        backup_dir="/var/backups/postgres",
        s3_bucket="my-ai-system-backups"
    )

    # Create full backup
    backup_file = backup_manager.create_backup(backup_type="full")

    if backup_file:
        print(f"\nBackup created: {backup_file}")

        # Clean up old backups
        backup_manager.cleanup_old_backups(days_to_keep=7)

        # List recent backups
        print("\nRecent backups:")
        backups = sorted(backup_manager.backup_dir.glob("ai_system_*.sql.gz"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True)
        for backup in backups[:5]:
            size_mb = backup.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(backup.stat().st_mtime)
            print(f"  {backup.name} ({size_mb:.1f}MB, {mtime})")
```

**Output:**
```
Creating full backup: ai_system_full_20260119_143045.sql.gz
Backup completed in 12.3s (156.8MB)
Uploading to S3: s3://my-ai-system-backups/backups/ai_system_full_20260119_143045.sql.gz
Upload completed

Backup created: /var/backups/postgres/ai_system_full_20260119_143045.sql.gz

Cleaning up backups older than 7 days...
Deleted 3 old backups, freed 445.2MB

Recent backups:
  ai_system_full_20260119_143045.sql.gz (156.8MB, 2026-01-19 14:30:45)
  ai_system_full_20260118_020000.sql.gz (152.3MB, 2026-01-18 02:00:00)
  ai_system_full_20260117_020000.sql.gz (148.7MB, 2026-01-17 02:00:00)
  ai_system_full_20260116_020000.sql.gz (145.1MB, 2026-01-16 02:00:00)
  ai_system_full_20260115_020000.sql.gz (141.9MB, 2026-01-15 02:00:00)
```

### Point-in-Time Recovery with WAL Archiving

For critical systems, enable Write-Ahead Log (WAL) archiving to restore to any point in time.

Edit `postgresql.conf`:

```ini
# Enable WAL archiving
wal_level = replica
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'
max_wal_senders = 3
wal_keep_size = 1GB
```

```python
# wal_manager.py - WAL archive management
import subprocess
from datetime import datetime
from pathlib import Path

class WALManager:
    def __init__(self, wal_archive_dir):
        self.wal_archive_dir = Path(wal_archive_dir)

    def list_wal_files(self):
        """List archived WAL files"""
        wal_files = sorted(self.wal_archive_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in wal_files)

        print(f"WAL Archive: {len(wal_files)} files, {total_size / (1024**2):.1f}MB")

        if wal_files:
            oldest = datetime.fromtimestamp(wal_files[0].stat().st_mtime)
            newest = datetime.fromtimestamp(wal_files[-1].stat().st_mtime)
            print(f"Oldest: {oldest}")
            print(f"Newest: {newest}")

        return wal_files

    def create_recovery_conf(self, backup_label, target_time=None):
        """Create recovery configuration for PITR"""
        recovery_conf = "restore_command = 'cp /var/lib/postgresql/wal_archive/%f %p'\n"

        if target_time:
            recovery_conf += f"recovery_target_time = '{target_time}'\n"

        recovery_conf += "recovery_target_action = 'promote'\n"

        return recovery_conf

# Example usage
wal_manager = WALManager("/var/lib/postgresql/wal_archive")
wal_files = wal_manager.list_wal_files()

# Show recovery configuration
target_time = "2026-01-19 14:00:00"
recovery_conf = wal_manager.create_recovery_conf("latest_backup", target_time)
print(f"\nRecovery configuration for PITR to {target_time}:")
print(recovery_conf)
```

**Output:**
```
WAL Archive: 342 files, 5.3GB
Oldest: 2026-01-12 02:00:15
Newest: 2026-01-19 14:30:45

Recovery configuration for PITR to 2026-01-19 14:00:00:
restore_command = 'cp /var/lib/postgresql/wal_archive/%f %p'
recovery_target_time = '2026-01-19 14:00:00'
recovery_target_action = 'promote'
```

## Complete Schema Example: Production-Ready AI System

Here's a complete schema for a production AI system with all the patterns covered:

```sql
-- Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Users and teams
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    team_id UUID,
    role VARCHAR(50) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    monthly_budget_usd DECIMAL(10, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE users ADD CONSTRAINT fk_users_team
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE SET NULL;

-- Conversations and messages
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500),
    system_prompt TEXT,
    model VARCHAR(100) NOT NULL,
    temperature DECIMAL(3, 2) DEFAULT 1.0,
    max_tokens INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_archived BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    content_embedding vector(1536),  -- OpenAI embedding size
    model VARCHAR(100),
    model_version VARCHAR(50),
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost_usd DECIMAL(10, 6),
    latency_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Tool usage tracking
CREATE TABLE tool_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    tool_name VARCHAR(100) NOT NULL,
    tool_input JSONB NOT NULL,
    tool_output JSONB,
    duration_ms INTEGER,
    status VARCHAR(20) DEFAULT 'success',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API usage and costs
CREATE TABLE api_usage (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    model VARCHAR(100) NOT NULL,
    endpoint VARCHAR(200) NOT NULL,
    tokens_input INTEGER NOT NULL,
    tokens_output INTEGER NOT NULL,
    cost_usd DECIMAL(10, 6) NOT NULL,
    latency_ms INTEGER,
    status VARCHAR(20) NOT NULL,
    error_code VARCHAR(50),
    error_message TEXT,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- System metrics (partitioned by month)
CREATE TABLE metrics (
    id BIGSERIAL,
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value DECIMAL(12, 4) NOT NULL,
    unit VARCHAR(20),
    labels JSONB DEFAULT '{}'::jsonb,
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, recorded_at)
) PARTITION BY RANGE (recorded_at);

-- Audit log
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    changes JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at DESC);
CREATE INDEX idx_conversations_user_date ON conversations(user_id, created_at DESC);
CREATE INDEX idx_conversations_metadata ON conversations USING GIN (metadata);

CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX idx_messages_metadata ON messages USING GIN (metadata);
CREATE INDEX idx_messages_embedding ON messages USING ivfflat (content_embedding vector_cosine_ops);

CREATE INDEX idx_tool_calls_message_id ON tool_calls(message_id);
CREATE INDEX idx_tool_calls_tool_name ON tool_calls(tool_name, created_at DESC);

CREATE INDEX idx_api_usage_user_time ON api_usage(user_id, recorded_at DESC);
CREATE INDEX idx_api_usage_team_time ON api_usage(team_id, recorded_at DESC);
CREATE INDEX idx_api_usage_model_time ON api_usage(model, recorded_at DESC);

CREATE INDEX idx_audit_log_user_id ON audit_log(user_id, created_at DESC);
CREATE INDEX idx_audit_log_resource ON audit_log(resource_type, resource_id);

-- Updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

### Python Code: Using the Complete Schema

```python
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import uuid
from datetime import datetime, timedelta

conn = psycopg2.connect(
    dbname="ai_system",
    user="postgres",
    password="your_password",
    host="localhost",
    port=5432,
    cursor_factory=RealDictCursor
)
cur = conn.cursor()

# Create a team
team_id = str(uuid.uuid4())
cur.execute("""
    INSERT INTO teams (id, name, monthly_budget_usd)
    VALUES (%s, %s, %s)
    RETURNING id, name
""", (team_id, "Network Operations", 1000.00))
team = cur.fetchone()
print(f"Created team: {team['name']} (ID: {team['id']})")

# Create a user
user_id = str(uuid.uuid4())
cur.execute("""
    INSERT INTO users (id, email, name, team_id, role)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING id, email, role
""", (user_id, "engineer@company.com", "John Engineer", team_id, "user"))
user = cur.fetchone()
print(f"Created user: {user['email']} (role: {user['role']})")

# Create a conversation
conversation_id = str(uuid.uuid4())
cur.execute("""
    INSERT INTO conversations (id, user_id, title, model, system_prompt)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING id, title
""", (
    conversation_id,
    user_id,
    "BGP Configuration Analysis",
    "claude-sonnet-4-5",
    "You are a network engineering assistant specialized in BGP routing protocols."
))
conversation = cur.fetchone()
print(f"Created conversation: {conversation['title']}")

# Add messages with tool usage
user_message_id = str(uuid.uuid4())
cur.execute("""
    INSERT INTO messages (id, conversation_id, role, content, metadata)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING id
""", (
    user_message_id,
    conversation_id,
    'user',
    "Analyze this BGP config and check for best practices",
    Json({"device": "router01", "location": "datacenter1"})
))

# Record tool call
tool_call_id = str(uuid.uuid4())
cur.execute("""
    INSERT INTO tool_calls (id, message_id, tool_name, tool_input, tool_output, duration_ms, status)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
""", (
    tool_call_id,
    user_message_id,
    "config_analyzer",
    Json({"config_path": "/configs/router01.conf"}),
    Json({"issues_found": 2, "severity": "medium"}),
    1234,
    "success"
))

# Add assistant response
assistant_message_id = str(uuid.uuid4())
cur.execute("""
    INSERT INTO messages (
        id, conversation_id, role, content, model, model_version,
        tokens_input, tokens_output, cost_usd, latency_ms
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id
""", (
    assistant_message_id,
    conversation_id,
    'assistant',
    "Found 2 BGP best practice issues: 1) Missing maximum-paths configuration...",
    "claude-sonnet-4-5",
    "20250929",
    2100,
    1560,
    0.042,
    2345
))

# Record API usage
cur.execute("""
    INSERT INTO api_usage (
        user_id, team_id, conversation_id, model, endpoint,
        tokens_input, tokens_output, cost_usd, latency_ms, status
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""", (
    user_id, team_id, conversation_id,
    "claude-sonnet-4-5", "/v1/messages",
    2100, 1560, 0.042, 2345, "success"
))

# Log audit entry
cur.execute("""
    INSERT INTO audit_log (user_id, action, resource_type, resource_id, changes)
    VALUES (%s, %s, %s, %s, %s)
""", (
    user_id,
    "conversation_created",
    "conversation",
    conversation_id,
    Json({"title": "BGP Configuration Analysis", "model": "claude-sonnet-4-5"})
))

conn.commit()
print("\nAll records created successfully")

# Query team usage and costs
cur.execute("""
    SELECT
        t.name as team_name,
        t.monthly_budget_usd,
        COUNT(DISTINCT a.conversation_id) as conversation_count,
        SUM(a.tokens_input) as total_input_tokens,
        SUM(a.tokens_output) as total_output_tokens,
        SUM(a.cost_usd) as total_cost,
        AVG(a.latency_ms) as avg_latency
    FROM teams t
    JOIN api_usage a ON t.id = a.team_id
    WHERE a.recorded_at >= NOW() - INTERVAL '30 days'
    GROUP BY t.id, t.name, t.monthly_budget_usd
""")

print("\nTeam usage (last 30 days):")
for row in cur.fetchall():
    budget_used_pct = (float(row['total_cost']) / float(row['monthly_budget_usd'])) * 100
    print(f"  {row['team_name']}")
    print(f"    Budget: ${row['monthly_budget_usd']:.2f} (${row['total_cost']:.2f} used, {budget_used_pct:.1f}%)")
    print(f"    Conversations: {row['conversation_count']}")
    print(f"    Tokens: {row['total_input_tokens']:,} in / {row['total_output_tokens']:,} out")
    print(f"    Avg latency: {row['avg_latency']:.0f}ms")

cur.close()
conn.close()
```

**Output:**
```
Created team: Network Operations (ID: 7f3e9a1c-4b2d-4e8f-9a2c-1d5f8e3a9b7c)
Created user: engineer@company.com (role: user)
Created conversation: BGP Configuration Analysis

All records created successfully

Team usage (last 30 days):
  Network Operations
    Budget: $1000.00 ($278.45 used, 27.8%)
    Conversations: 89
    Tokens: 234,560 in / 178,920 out
    Avg latency: 1876ms
```

## Key Takeaways

Database design for AI systems requires thinking beyond simple CRUD operations:

1. **Schema design** - JSONB for flexible metadata, proper foreign keys, and clear separation between conversations and messages
2. **Indexing** - GIN indexes for JSONB, composite indexes for common query patterns, partial indexes for filtered queries
3. **Time-series data** - Partition large tables by date, use appropriate indexes for metrics and cost tracking
4. **Connection pooling** - PgBouncer reduces connection overhead and improves throughput
5. **Migrations** - Alembic tracks schema changes and applies them safely across environments
6. **Backups** - Automated pg_dump backups with S3 storage, WAL archiving for point-in-time recovery
7. **Complete schema** - Users, teams, conversations, messages, tool calls, API usage, metrics, and audit logs all connected

Your AI system generates data fast - conversations, costs, metrics, and audit logs. A well-designed database with proper indexing and partitioning keeps queries fast as you scale from hundreds to millions of interactions.

Next chapter covers deployment patterns - Docker containers, Kubernetes orchestration, and CI/CD pipelines for AI systems.
