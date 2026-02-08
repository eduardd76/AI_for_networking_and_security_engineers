# Chapter 39: API Gateway & Load Balancing

## Introduction

You know how to configure BGP multipath, set up HSRP/VRRP for first-hop redundancy, and implement server load balancing on F5 or Cisco ACE. Now you're building AI-powered network operations tools, and you need the same concepts at the application layer.

Your AI agents hit APIs thousands of times per day. A single chatbot query might trigger 5-10 API calls to different services. Without proper load balancing and gateway logic, you'll have:

- Single points of failure (one LLM API down = entire workflow broken)
- No visibility into API performance (which model is slow? which endpoint fails?)
- Credential exposure (every service has direct API keys)
- Rate limit exhaustion (no intelligent retry or backoff)

This chapter covers production-grade API gateway patterns using tools you already understand: Nginx (like a Layer 7 switch), circuit breakers (like interface error-disable), and retry logic (like BGP route dampening).

We're building a gateway that sits between your AI agents and external services (OpenAI, Anthropic, vector databases, monitoring APIs). Think of it as a reverse proxy with intelligence.

## Why API Gateways Matter for AI Workloads

### The Problem

Your AI agent makes this call:

```python
# Direct API call - brittle
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze this config"}],
    api_key="sk-proj-..."  # Exposed in code
)
```

What happens when:
- OpenAI has an outage? (Happens monthly)
- You hit rate limits? (429 errors)
- You want to switch to Claude? (Change code everywhere)
- You need audit logs? (No visibility)
- Network latency spikes? (No timeout control)

### The Solution: API Gateway Pattern

```
[AI Agent] → [API Gateway] → [OpenAI]
                           → [Anthropic]
                           → [Local Model]
```

The gateway handles:
1. **Load balancing** - Distribute across multiple backends
2. **Failover** - Switch providers automatically
3. **Circuit breaking** - Stop hitting dead endpoints
4. **Rate limiting** - Protect against quota exhaustion
5. **Authentication** - Single point for credential management
6. **Logging** - Track every API call, response time, token usage
7. **Caching** - Store frequent queries (config analysis doesn't change)

This is exactly like your datacenter edge: centralized policy enforcement, visibility, and control.

## Nginx as API Gateway

Nginx is your Layer 7 application switch. You've configured it as a web server, now we're using it as an intelligent reverse proxy.

### Basic Setup

Install Nginx on Ubuntu (or your gateway VM):

```bash
sudo apt update
sudo apt install nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

### Configuration Structure

Nginx config is like Cisco IOS - hierarchical blocks with inheritance:

```nginx
# /etc/nginx/nginx.conf
http {                          # Global HTTP context
    upstream backend {          # Like a server farm
        server 10.1.1.10:8000;
        server 10.1.1.11:8000;
    }

    server {                    # Like a VIP listener
        listen 80;
        location /api {         # Like a URI-based policy
            proxy_pass http://backend;
        }
    }
}
```

Compare to F5 config:

```
# F5 equivalent
ltm pool backend_pool {
    members {
        10.1.1.10:8000
        10.1.1.11:8000
    }
}

ltm virtual api_vip {
    destination 10.0.0.100:80
    pool backend_pool
}
```

## Load Balancing Strategies

### Round-Robin (Default)

Distribute requests evenly across backends. Like ECMP for TCP flows.

**Config:**

```nginx
# /etc/nginx/conf.d/api-gateway.conf
upstream llm_backends {
    server 10.1.1.10:8000;  # OpenAI proxy
    server 10.1.1.11:8000;  # Anthropic proxy
    server 10.1.1.12:8000;  # Local model
}

server {
    listen 80;
    server_name api.vexpertai.local;

    location /v1/chat/completions {
        proxy_pass http://llm_backends;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Test it:**

```bash
# Make 6 requests, watch distribution
for i in {1..6}; do
    curl -s http://api.vexpertai.local/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}' | \
    jq -r '.backend'
done
```

**Output:**

```
server_1
server_2
server_3
server_1
server_2
server_3
```

Each server gets 2 requests. Perfect distribution for uniform workloads.

### Least Connections

Route to the backend with fewest active connections. Better for long-running API calls (like streaming LLM responses).

**Config:**

```nginx
upstream llm_backends {
    least_conn;  # Enable least-connections algorithm

    server 10.1.1.10:8000;
    server 10.1.1.11:8000;
    server 10.1.1.12:8000;
}
```

**When to use:**
- Streaming API responses (SSE, WebSocket)
- Variable request processing time (some queries take 2s, others 30s)
- Mixed workload (chat + embeddings + image generation)

Think of it like CEF load balancing with flow awareness instead of per-packet.

### Weighted Load Balancing

Give more traffic to more powerful backends.

**Config:**

```nginx
upstream llm_backends {
    server 10.1.1.10:8000 weight=3;  # GPU server, 3x capacity
    server 10.1.1.11:8000 weight=2;  # Mid-tier server
    server 10.1.1.12:8000 weight=1;  # CPU-only, backup
}
```

With 6 requests:
- Server 1: 3 requests (50%)
- Server 2: 2 requests (33%)
- Server 3: 1 request (17%)

**Use case:** You have:
- 1 server with 4x A100 GPUs
- 2 servers with single T4 GPUs
- 1 CPU-only backup

Weight them 8:2:2:1 to match capacity.

### IP Hash (Session Persistence)

Route the same client to the same backend. Like sticky sessions on a load balancer.

**Config:**

```nginx
upstream llm_backends {
    ip_hash;  # Hash source IP to select backend

    server 10.1.1.10:8000;
    server 10.1.1.11:8000;
    server 10.1.1.12:8000;
}
```

**Why you need it:**
- Conversation state stored in backend memory
- Rate limiting per backend (client shouldn't hop between servers)
- Debugging (know which logs to check)

**Trade-off:** Less even distribution if you have few clients behind NAT.

## Health Checks and Failover

Like BFD for your APIs - detect failures fast and stop sending traffic.

### Passive Health Checks (Built-in)

Nginx marks a backend as down after failed requests.

**Config:**

```nginx
upstream llm_backends {
    server 10.1.1.10:8000 max_fails=3 fail_timeout=30s;
    server 10.1.1.11:8000 max_fails=3 fail_timeout=30s;
    server 10.1.1.12:8000 max_fails=2 fail_timeout=60s;
}
```

**Parameters:**
- `max_fails=3` - Mark down after 3 consecutive failures (like interface error-disable threshold)
- `fail_timeout=30s` - Wait 30s before retry (like BGP route dampening)

**Behavior:**

```
Request 1 → server_1 → 502 Bad Gateway (fail count: 1)
Request 2 → server_1 → 502 Bad Gateway (fail count: 2)
Request 3 → server_1 → 502 Bad Gateway (fail count: 3, MARKED DOWN)
Request 4 → server_2 → 200 OK (traffic switched)
...wait 30 seconds...
Request N → server_1 → 200 OK (back in rotation)
```

### Active Health Checks (Nginx Plus / OpenResty)

Send probe requests independent of client traffic. Like IP SLA monitors.

**Config (Nginx Plus):**

```nginx
upstream llm_backends {
    zone llm_zone 64k;

    server 10.1.1.10:8000;
    server 10.1.1.11:8000;
    server 10.1.1.12:8000;

    health_check interval=5s
                 fails=2
                 passes=2
                 uri=/health
                 match=health_ok;
}

match health_ok {
    status 200;
    header Content-Type = "application/json";
    body ~ "\"status\":\"healthy\"";
}
```

**Backend health endpoint:**

```python
# Flask backend
from flask import Flask, jsonify
import requests

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Health check endpoint - verify LLM API is reachable"""
    try:
        # Quick test to actual LLM API
        response = requests.get('https://api.openai.com/v1/models',
                               headers={'Authorization': f'Bearer {API_KEY}'},
                               timeout=2)
        if response.status_code == 200:
            return jsonify({'status': 'healthy', 'backend': 'openai'}), 200
        else:
            return jsonify({'status': 'unhealthy', 'error': 'api_error'}), 503
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

**Test output:**

```bash
curl http://10.1.1.10:8000/health
```

```json
{
  "status": "healthy",
  "backend": "openai"
}
```

Every 5 seconds, Nginx probes `/health`. After 2 consecutive failures, backend is marked down. After 2 consecutive successes, it's back up.

## Request Routing and URL Rewriting

Route based on URI, header, or query parameter. Like policy-based routing for HTTP.

### URI-Based Routing

Send different API types to specialized backends.

**Config:**

```nginx
# Chat completions → OpenAI optimized backend
upstream chat_backends {
    server 10.1.1.10:8000;
    server 10.1.1.11:8000;
}

# Embeddings → Dedicated embedding service
upstream embedding_backends {
    server 10.1.2.10:8000;
    server 10.1.2.11:8000;
}

# Image generation → GPU cluster
upstream image_backends {
    server 10.1.3.10:8000;
    server 10.1.3.11:8000;
}

server {
    listen 80;
    server_name api.vexpertai.local;

    # Route based on URI path
    location /v1/chat/completions {
        proxy_pass http://chat_backends;
    }

    location /v1/embeddings {
        proxy_pass http://embedding_backends;
    }

    location /v1/images/generations {
        proxy_pass http://image_backends;
    }
}
```

This is like VRF-aware routing - same gateway, different forwarding tables per service type.

### Header-Based Routing

Route based on HTTP headers (model selection, API version, tenant ID).

**Config:**

```nginx
map $http_x_model_provider $backend_pool {
    "openai"     "openai_backends";
    "anthropic"  "anthropic_backends";
    "local"      "local_backends";
    default      "openai_backends";
}

upstream openai_backends {
    server 10.1.1.10:8000;
    server 10.1.1.11:8000;
}

upstream anthropic_backends {
    server 10.1.2.10:8000;
    server 10.1.2.11:8000;
}

upstream local_backends {
    server 10.1.3.10:8000;
}

server {
    listen 80;

    location /v1/chat/completions {
        proxy_pass http://$backend_pool;
        proxy_set_header Host $host;
    }
}
```

**Client usage:**

```python
import requests

# Route to Anthropic backend
response = requests.post(
    'http://api.vexpertai.local/v1/chat/completions',
    headers={
        'Content-Type': 'application/json',
        'X-Model-Provider': 'anthropic'  # Routing key
    },
    json={
        'model': 'claude-sonnet-4-20250514',
        'messages': [{'role': 'user', 'content': 'Hello'}]
    }
)
```

**Output:**

```
Request routed to: 10.1.2.10:8000 (anthropic_backends pool)
```

Change the header, change the destination. No code changes in your agent.

### URL Rewriting

Backend expects different URI than client sends. Like NAT for URLs.

**Config:**

```nginx
server {
    listen 80;

    # Client calls /api/chat
    # Backend expects /v1/chat/completions
    location /api/chat {
        rewrite ^/api/chat(.*)$ /v1/chat/completions$1 break;
        proxy_pass http://llm_backends;
    }

    # Client calls /api/embed
    # Backend expects /v1/embeddings
    location /api/embed {
        rewrite ^/api/embed(.*)$ /v1/embeddings$1 break;
        proxy_pass http://embedding_backends;
    }
}
```

**Test:**

```bash
# Client makes simple call
curl -X POST http://api.vexpertai.local/api/chat \
     -H "Content-Type: application/json" \
     -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}'

# Nginx rewrites to:
# POST http://10.1.1.10:8000/v1/chat/completions
```

Your agents use clean, simple URIs. The gateway translates to whatever messy backend structure you have.

## SSL/TLS Termination

Decrypt at the gateway, re-encrypt to backends. Like SSL offload on F5.

### Basic TLS Termination

**Generate self-signed cert (testing only):**

```bash
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/api-gateway.key \
    -out /etc/nginx/ssl/api-gateway.crt \
    -subj "/C=DE/ST=Bavaria/L=Munich/O=vExpertAI/CN=api.vexpertai.local"
```

**Config:**

```nginx
server {
    listen 443 ssl http2;
    server_name api.vexpertai.local;

    # TLS configuration
    ssl_certificate /etc/nginx/ssl/api-gateway.crt;
    ssl_certificate_key /etc/nginx/ssl/api-gateway.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Enable session cache (like TCP connection reuse)
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    location /v1/chat/completions {
        proxy_pass http://llm_backends;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.vexpertai.local;
    return 301 https://$server_name$request_uri;
}
```

**Test:**

```bash
curl -k https://api.vexpertai.local/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}'
```

**Output:**

```
TLS handshake: TLSv1.3, cipher TLS_AES_256_GCM_SHA384
Connection: Encrypted (client → gateway)
Backend connection: Plaintext (gateway → backend, internal network)
Response: 200 OK
```

Benefits:
- Single certificate to manage (not one per backend)
- Offload crypto from backend servers (save CPU for LLM inference)
- Centralized TLS policy (enforce TLS 1.3, disable weak ciphers)
- Inspect traffic at gateway (logging, IDS, DLP)

### Mutual TLS (mTLS)

Require client certificates. Like 802.1X for APIs.

**Config:**

```nginx
server {
    listen 443 ssl http2;
    server_name api.vexpertai.local;

    # Server cert
    ssl_certificate /etc/nginx/ssl/api-gateway.crt;
    ssl_certificate_key /etc/nginx/ssl/api-gateway.key;

    # Client cert validation
    ssl_client_certificate /etc/nginx/ssl/ca.crt;
    ssl_verify_client on;
    ssl_verify_depth 2;

    location /v1/chat/completions {
        # Pass client cert info to backend
        proxy_set_header X-Client-Cert $ssl_client_cert;
        proxy_set_header X-Client-DN $ssl_client_s_dn;
        proxy_pass http://llm_backends;
    }
}
```

**Client usage:**

```python
import requests

response = requests.post(
    'https://api.vexpertai.local/v1/chat/completions',
    cert=('/path/to/client.crt', '/path/to/client.key'),  # Client certificate
    verify='/path/to/ca.crt',  # CA certificate
    json={'model': 'gpt-4', 'messages': [{'role': 'user', 'content': 'test'}]}
)
```

**Failed auth (no client cert):**

```
Error: 400 Bad Request
No required SSL certificate was sent
```

Use mTLS when:
- Agent-to-gateway communication crosses untrusted networks
- Multiple teams share the gateway (isolate with client certs)
- Compliance requires mutual authentication (PCI-DSS, HIPAA)

## Circuit Breaker Patterns

Stop hitting a failing service. Like interface shutdown on error threshold.

### The Problem

Your backend LLM API is down. Without circuit breaking:

```
Request 1 → 502 error after 30s timeout
Request 2 → 502 error after 30s timeout
Request 3 → 502 error after 30s timeout
...
Request 100 → 502 error after 30s timeout

Total time wasted: 3000 seconds (50 minutes)
User experience: Terrible
```

### Circuit Breaker States

```
[CLOSED] → failures → [OPEN] → timeout → [HALF-OPEN] → success → [CLOSED]
                                               ↓
                                          more failures
                                               ↓
                                            [OPEN]
```

- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Too many failures, reject immediately (no backend call)
- **HALF-OPEN**: After timeout, try one request to test recovery

### Implementation with Nginx

Nginx doesn't have native circuit breaking, but we can combine features:

**Config:**

```nginx
upstream llm_backends {
    server 10.1.1.10:8000 max_fails=3 fail_timeout=30s;
    server 10.1.1.11:8000 max_fails=3 fail_timeout=30s;

    # If all backends fail, return 503 immediately
    keepalive 32;
}

server {
    listen 80;

    location /v1/chat/completions {
        proxy_pass http://llm_backends;

        # Fast timeout (don't wait forever)
        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;

        # Custom error page
        error_page 502 503 504 @backend_down;
    }

    location @backend_down {
        # Return fast failure with retry-after header
        add_header Retry-After 30 always;
        return 503 '{"error": "Service temporarily unavailable", "retry_after": 30}';
    }
}
```

**Better approach: Use OpenResty (Nginx + Lua):**

```nginx
# /etc/nginx/conf.d/circuit-breaker.conf
lua_shared_dict circuit_breaker 10m;

server {
    listen 80;

    location /v1/chat/completions {
        access_by_lua_block {
            local cb = require "circuit_breaker"
            if cb.is_open("llm_backends") then
                ngx.status = 503
                ngx.header["Retry-After"] = "30"
                ngx.say('{"error": "Circuit breaker open", "retry_after": 30}')
                ngx.exit(503)
            end
        }

        proxy_pass http://llm_backends;

        log_by_lua_block {
            local cb = require "circuit_breaker"
            if ngx.status >= 500 then
                cb.record_failure("llm_backends")
            else
                cb.record_success("llm_backends")
            end
        }
    }
}
```

**Circuit breaker module (circuit_breaker.lua):**

```lua
local _M = {}

local FAILURE_THRESHOLD = 5      -- Open circuit after 5 failures
local SUCCESS_THRESHOLD = 2      -- Close circuit after 2 successes
local TIMEOUT = 30               -- Half-open after 30 seconds

function _M.is_open(backend)
    local state = ngx.shared.circuit_breaker:get(backend .. ":state")
    local opened_at = ngx.shared.circuit_breaker:get(backend .. ":opened_at")

    if state == "open" then
        -- Check if timeout elapsed (transition to half-open)
        if ngx.now() - opened_at > TIMEOUT then
            ngx.shared.circuit_breaker:set(backend .. ":state", "half_open")
            return false
        end
        return true
    end

    return false
end

function _M.record_failure(backend)
    local state = ngx.shared.circuit_breaker:get(backend .. ":state") or "closed"
    local failures = ngx.shared.circuit_breaker:get(backend .. ":failures") or 0

    failures = failures + 1
    ngx.shared.circuit_breaker:set(backend .. ":failures", failures)

    if failures >= FAILURE_THRESHOLD then
        ngx.shared.circuit_breaker:set(backend .. ":state", "open")
        ngx.shared.circuit_breaker:set(backend .. ":opened_at", ngx.now())
        ngx.log(ngx.WARN, "Circuit breaker opened for " .. backend)
    end
end

function _M.record_success(backend)
    local state = ngx.shared.circuit_breaker:get(backend .. ":state") or "closed"

    if state == "half_open" then
        local successes = ngx.shared.circuit_breaker:get(backend .. ":successes") or 0
        successes = successes + 1
        ngx.shared.circuit_breaker:set(backend .. ":successes", successes)

        if successes >= SUCCESS_THRESHOLD then
            ngx.shared.circuit_breaker:set(backend .. ":state", "closed")
            ngx.shared.circuit_breaker:set(backend .. ":failures", 0)
            ngx.shared.circuit_breaker:set(backend .. ":successes", 0)
            ngx.log(ngx.INFO, "Circuit breaker closed for " .. backend)
        end
    else
        -- Reset failure count on success
        ngx.shared.circuit_breaker:set(backend .. ":failures", 0)
    end
end

return _M
```

**Test output:**

```bash
# Simulate backend failures
for i in {1..10}; do
    curl http://api.vexpertai.local/v1/chat/completions
    echo ""
done
```

**Output:**

```
Request 1: 502 Bad Gateway (failure 1)
Request 2: 502 Bad Gateway (failure 2)
Request 3: 502 Bad Gateway (failure 3)
Request 4: 502 Bad Gateway (failure 4)
Request 5: 502 Bad Gateway (failure 5, circuit OPENS)
Request 6: 503 Circuit breaker open, retry_after: 30 (fast fail)
Request 7: 503 Circuit breaker open, retry_after: 30 (fast fail)
...wait 30 seconds...
Request 8: 200 OK (half-open, test request succeeds)
Request 9: 200 OK (success 2, circuit CLOSES)
Request 10: 200 OK (normal operation)
```

Notice requests 6-7 fail immediately (no 30s timeout). That's the circuit breaker saving time.

### Application-Level Circuit Breaker (Python)

Implement in your agent code using `pybreaker` library:

```python
from pybreaker import CircuitBreaker
import requests
import time

# Configure circuit breaker
breaker = CircuitBreaker(
    fail_max=5,           # Open after 5 failures
    timeout_duration=30,  # Stay open for 30 seconds
    name='openai_api'
)

@breaker
def call_openai_api(prompt):
    """Call OpenAI API with circuit breaker protection"""
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        },
        json={
            'model': 'gpt-4',
            'messages': [{'role': 'user', 'content': prompt}]
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()

# Test with failures
for i in range(10):
    try:
        result = call_openai_api(f"Test query {i}")
        print(f"Request {i}: Success")
    except CircuitBreakerError:
        print(f"Request {i}: Circuit breaker open (fast fail)")
    except Exception as e:
        print(f"Request {i}: Failed - {e}")

    time.sleep(1)
```

**Output:**

```
Request 0: Failed - 503 Server Error
Request 1: Failed - 503 Server Error
Request 2: Failed - 503 Server Error
Request 3: Failed - 503 Server Error
Request 4: Failed - 503 Server Error
Request 5: Circuit breaker open (fast fail)
Request 6: Circuit breaker open (fast fail)
Request 7: Circuit breaker open (fast fail)
...wait for timeout...
Request 8: Success (half-open state)
Request 9: Success (circuit closed)
```

Requests 5-7 fail instantly (no API call). Circuit breaker in action.

## Retry Logic and Backoff Strategies

Retry transient failures intelligently. Like BGP soft reconfiguration.

### Exponential Backoff

Wait longer between each retry attempt.

**Formula:**
```
wait_time = base_delay * (2 ^ attempt) + random_jitter
```

**Python implementation:**

```python
import requests
import time
import random

def call_api_with_retry(url, max_retries=5, base_delay=1):
    """
    Call API with exponential backoff retry logic

    Args:
        url: API endpoint
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json={'model': 'gpt-4', 'messages': []}, timeout=30)

            # Success
            if response.status_code == 200:
                return response.json()

            # Rate limited - definitely retry
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 0))
                wait_time = retry_after if retry_after > 0 else (base_delay * (2 ** attempt))
                jitter = random.uniform(0, wait_time * 0.1)
                total_wait = wait_time + jitter

                print(f"Attempt {attempt + 1}: Rate limited (429)")
                print(f"  Waiting {total_wait:.2f}s before retry")
                time.sleep(total_wait)
                continue

            # Server error - retry
            if response.status_code >= 500:
                wait_time = base_delay * (2 ** attempt)
                jitter = random.uniform(0, wait_time * 0.1)
                total_wait = wait_time + jitter

                print(f"Attempt {attempt + 1}: Server error ({response.status_code})")
                print(f"  Waiting {total_wait:.2f}s before retry")
                time.sleep(total_wait)
                continue

            # Client error - don't retry
            if response.status_code >= 400:
                print(f"Attempt {attempt + 1}: Client error ({response.status_code}) - not retrying")
                response.raise_for_status()

        except requests.exceptions.Timeout:
            wait_time = base_delay * (2 ** attempt)
            jitter = random.uniform(0, wait_time * 0.1)
            total_wait = wait_time + jitter

            print(f"Attempt {attempt + 1}: Timeout")
            print(f"  Waiting {total_wait:.2f}s before retry")
            time.sleep(total_wait)
            continue

        except requests.exceptions.ConnectionError:
            wait_time = base_delay * (2 ** attempt)
            jitter = random.uniform(0, wait_time * 0.1)
            total_wait = wait_time + jitter

            print(f"Attempt {attempt + 1}: Connection error")
            print(f"  Waiting {total_wait:.2f}s before retry")
            time.sleep(total_wait)
            continue

    # All retries exhausted
    raise Exception(f"Failed after {max_retries} attempts")

# Test
result = call_api_with_retry('https://api.openai.com/v1/chat/completions')
```

**Output (simulated failures):**

```
Attempt 1: Server error (503)
  Waiting 1.08s before retry
Attempt 2: Server error (503)
  Waiting 2.15s before retry
Attempt 3: Rate limited (429)
  Waiting 4.32s before retry
Attempt 4: Timeout
  Waiting 8.71s before retry
Attempt 5: Success
Response received: {...}

Total retries: 4
Total wait time: 16.26 seconds
```

Notice the wait times: 1s, 2s, 4s, 8s (exponential), with jitter added for randomness.

### Nginx Retry Configuration

Configure gateway-level retries:

```nginx
upstream llm_backends {
    server 10.1.1.10:8000 max_fails=2 fail_timeout=10s;
    server 10.1.1.11:8000 max_fails=2 fail_timeout=10s;
    server 10.1.1.12:8000 max_fails=2 fail_timeout=10s backup;  # Only use if others fail
}

server {
    listen 80;

    location /v1/chat/completions {
        proxy_pass http://llm_backends;

        # Retry configuration
        proxy_next_upstream error timeout http_502 http_503 http_504;
        proxy_next_upstream_tries 3;      # Try up to 3 different backends
        proxy_next_upstream_timeout 60s;  # Total retry timeout

        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
```

**Retry behavior:**

```
Request arrives → Try backend 1 (10.1.1.10)
  ↓
  Connection timeout after 5s
  ↓
Retry → Try backend 2 (10.1.1.11)
  ↓
  502 Bad Gateway
  ↓
Retry → Try backend 3 (10.1.1.12, backup)
  ↓
  200 OK (success)
```

**Don't retry on:**
- 4xx errors (client error - won't succeed on retry)
- POST requests that modify state (unless idempotent)
- Requests that already succeeded but response was corrupted

**Always retry on:**
- Network errors (connection refused, timeout)
- 502, 503, 504 (server temporarily unavailable)
- 429 (rate limit - with backoff)

### Idempotency Keys

Ensure retries don't duplicate operations. Like TCP sequence numbers.

**Client implementation:**

```python
import requests
import uuid

def call_api_idempotent(url, payload, max_retries=3):
    """Call API with idempotency key to prevent duplicate operations"""

    # Generate unique idempotency key for this request
    idempotency_key = str(uuid.uuid4())

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers={
                    'Content-Type': 'application/json',
                    'Idempotency-Key': idempotency_key  # Same key for all retries
                },
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()

            # Retry on server errors
            if response.status_code >= 500:
                print(f"Attempt {attempt + 1} failed, retrying...")
                continue

            # Don't retry client errors
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue

    raise Exception("All retries exhausted")

# Backend receives same idempotency key on retry
# Can detect duplicate and return cached response
```

**Backend handling (Flask example):**

```python
from flask import Flask, request, jsonify
import hashlib

app = Flask(__name__)
response_cache = {}  # In production, use Redis

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completion():
    idempotency_key = request.headers.get('Idempotency-Key')

    # Check if we've already processed this request
    if idempotency_key and idempotency_key in response_cache:
        print(f"Duplicate request detected: {idempotency_key}")
        cached_response = response_cache[idempotency_key]
        return jsonify(cached_response), 200

    # Process request (expensive LLM call)
    response_data = {
        'id': 'chatcmpl-123',
        'model': 'gpt-4',
        'choices': [{'message': {'content': 'Hello!'}}]
    }

    # Cache response for future retries
    if idempotency_key:
        response_cache[idempotency_key] = response_data

    return jsonify(response_data), 200
```

**Test output:**

```
First attempt: Processing request (10s LLM call)
  Response: 200 OK

Network error, retry...

Second attempt: Duplicate detected, returning cached response
  Response: 200 OK (instant, no LLM call)
```

Same idempotency key = same response. No duplicate charges, no duplicate operations.

## Gateway Logging and Monitoring

You can't troubleshoot what you can't see. Log everything.

### Access Logging

**Custom log format:**

```nginx
# /etc/nginx/nginx.conf
http {
    # Define custom log format with API-specific fields
    log_format api_gateway '$remote_addr - $remote_user [$time_local] '
                          '"$request" $status $body_bytes_sent '
                          '"$http_referer" "$http_user_agent" '
                          'rt=$request_time uct="$upstream_connect_time" '
                          'uht="$upstream_header_time" urt="$upstream_response_time" '
                          'upstream=$upstream_addr '
                          'cache=$upstream_cache_status';

    access_log /var/log/nginx/api-gateway-access.log api_gateway;
    error_log /var/log/nginx/api-gateway-error.log warn;
}
```

**Sample log entry:**

```
10.0.1.50 - - [19/Jan/2026:14:23:45 +0100] "POST /v1/chat/completions HTTP/1.1" 200 1542 "-" "python-requests/2.31.0" rt=2.341 uct="0.002" uht="0.523" urt="2.339" upstream=10.1.1.10:8000 cache=-
```

**What it tells you:**
- `rt=2.341` - Total request time: 2.34 seconds
- `uct="0.002"` - Connect to backend: 2ms (good)
- `uht="0.523"` - Backend processing headers: 523ms
- `urt="2.339"` - Backend response time: 2.34s (matches total, good)
- `upstream=10.1.1.10:8000` - Which backend served it

### Performance Monitoring

Parse logs to find slow requests:

```bash
# Find requests slower than 5 seconds
awk '$NF ~ /rt=[0-9]+\.[0-9]+/ {
    match($0, /rt=([0-9]+\.[0-9]+)/, arr);
    if (arr[1] > 5.0) print arr[1], $7, $13
}' /var/log/nginx/api-gateway-access.log | sort -rn | head -20
```

**Output:**

```
12.453 /v1/chat/completions upstream=10.1.1.10:8000
9.872 /v1/chat/completions upstream=10.1.1.11:8000
7.234 /v1/embeddings upstream=10.1.2.10:8000
6.891 /v1/chat/completions upstream=10.1.1.10:8000
5.567 /v1/images/generations upstream=10.1.3.10:8000
```

Backend 10.1.1.10 appears twice in top 5 slowest. Investigate that server.

### Request Tracing

Add unique request ID to track through entire stack:

```nginx
server {
    listen 80;

    location /v1/chat/completions {
        # Generate unique request ID
        set $request_id $request_id;

        # Pass to backend
        proxy_set_header X-Request-ID $request_id;
        proxy_pass http://llm_backends;

        # Add to response header
        add_header X-Request-ID $request_id always;
    }
}
```

**Backend logging (Python):**

```python
from flask import Flask, request, g
import logging

app = Flask(__name__)

@app.before_request
def before_request():
    g.request_id = request.headers.get('X-Request-ID', 'unknown')
    logging.info(f"[{g.request_id}] Incoming request: {request.method} {request.path}")

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completion():
    logging.info(f"[{g.request_id}] Processing chat completion")

    # Call OpenAI
    logging.info(f"[{g.request_id}] Calling OpenAI API")
    # ... API call ...
    logging.info(f"[{g.request_id}] OpenAI responded in 2.3s")

    return jsonify({'response': 'data'}), 200
```

**Trace a request across logs:**

```bash
# Client receives: X-Request-ID: 7f3c9d8e-1a2b-4c5d-8e9f-0a1b2c3d4e5f

# Gateway log:
grep "7f3c9d8e-1a2b-4c5d-8e9f-0a1b2c3d4e5f" /var/log/nginx/api-gateway-access.log

# Backend log:
grep "7f3c9d8e-1a2b-4c5d-8e9f-0a1b2c3d4e5f" /var/log/app/backend.log

# OpenAI proxy log:
grep "7f3c9d8e-1a2b-4c5d-8e9f-0a1b2c3d4e5f" /var/log/app/openai-proxy.log
```

**Output:**

```
# Gateway
[19/Jan/2026:14:23:45] request_id=7f3c9d8e rt=2.341 upstream=10.1.1.10:8000

# Backend
[14:23:45] [7f3c9d8e] Incoming request: POST /v1/chat/completions
[14:23:45] [7f3c9d8e] Processing chat completion
[14:23:45] [7f3c9d8e] Calling OpenAI API
[14:23:47] [7f3c9d8e] OpenAI responded in 2.3s

# OpenAI proxy
[14:23:45] [7f3c9d8e] POST https://api.openai.com/v1/chat/completions
[14:23:47] [7f3c9d8e] Response: 200 OK, tokens=145
```

Now you can trace one request through the entire stack. Like packet capture with correlation ID.

### Prometheus Metrics Export

Export gateway metrics for monitoring dashboards:

```nginx
# Install nginx-prometheus-exporter
# https://github.com/nginxinc/nginx-prometheus-exporter

# Enable Nginx stub_status
server {
    listen 9113;
    location /metrics {
        stub_status on;
        access_log off;
        allow 10.0.0.0/8;
        deny all;
    }
}
```

**Metrics output:**

```
# HELP nginx_connections_active Active client connections
# TYPE nginx_connections_active gauge
nginx_connections_active 43

# HELP nginx_http_requests_total Total http requests
# TYPE nginx_http_requests_total counter
nginx_http_requests_total 125847

# HELP nginx_upstream_response_time_seconds Upstream response time
# TYPE nginx_upstream_response_time_seconds histogram
nginx_upstream_response_time_seconds_bucket{upstream="llm_backends",le="0.5"} 10234
nginx_upstream_response_time_seconds_bucket{upstream="llm_backends",le="1.0"} 15678
nginx_upstream_response_time_seconds_bucket{upstream="llm_backends",le="2.0"} 18901
nginx_upstream_response_time_seconds_bucket{upstream="llm_backends",le="5.0"} 19234
```

Import into Grafana for visual dashboards.

### Complete Monitoring Stack

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./logs:/var/log/nginx
    depends_on:
      - backend1
      - backend2

  backend1:
    image: python:3.11
    command: python /app/backend.py
    volumes:
      - ./backend.py:/app/backend.py
    environment:
      - BACKEND_ID=1

  backend2:
    image: python:3.11
    command: python /app/backend.py
    volumes:
      - ./backend.py:/app/backend.py
    environment:
      - BACKEND_ID=2

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus

  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:latest
    volumes:
      - ./logs:/var/log/nginx
      - ./promtail-config.yml:/etc/promtail/config.yml
    command:
      - '-config.file=/etc/promtail/config.yml'
```

**Start monitoring stack:**

```bash
docker-compose up -d
```

**Access dashboards:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Logs: http://localhost:3000/explore (select Loki data source)

## Complete Production Example

Putting it all together - a production-ready API gateway for AI workloads.

**File: /etc/nginx/nginx.conf**

```nginx
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 2048;
    use epoll;
}

http {
    include /etc/nginx/mime.types;
    default_type application/json;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    log_format api_detailed '$remote_addr - $remote_user [$time_local] '
                           '"$request" $status $body_bytes_sent '
                           'request_id=$request_id '
                           'rt=$request_time '
                           'uct="$upstream_connect_time" '
                           'uht="$upstream_header_time" '
                           'urt="$upstream_response_time" '
                           'upstream=$upstream_addr '
                           'upstream_status=$upstream_status '
                           'model=$http_x_model_provider';

    access_log /var/log/nginx/access.log api_detailed;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
    limit_req_status 429;

    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

    # Include upstream definitions
    include /etc/nginx/conf.d/upstreams.conf;

    # Include server blocks
    include /etc/nginx/conf.d/*.conf;
}
```

**File: /etc/nginx/conf.d/upstreams.conf**

```nginx
# OpenAI backend pool
upstream openai_backends {
    least_conn;

    server 10.1.1.10:8000 weight=3 max_fails=2 fail_timeout=30s;
    server 10.1.1.11:8000 weight=2 max_fails=2 fail_timeout=30s;
    server 10.1.1.12:8000 weight=1 max_fails=2 fail_timeout=30s backup;

    keepalive 64;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

# Anthropic backend pool
upstream anthropic_backends {
    least_conn;

    server 10.1.2.10:8000 weight=2 max_fails=2 fail_timeout=30s;
    server 10.1.2.11:8000 weight=2 max_fails=2 fail_timeout=30s;

    keepalive 64;
}

# Local model backend pool
upstream local_backends {
    least_conn;

    server 10.1.3.10:8000 max_fails=3 fail_timeout=60s;

    keepalive 32;
}

# Embedding service pool
upstream embedding_backends {
    least_conn;

    server 10.1.4.10:8000 weight=3 max_fails=2 fail_timeout=30s;
    server 10.1.4.11:8000 weight=2 max_fails=2 fail_timeout=30s;

    keepalive 64;
}
```

**File: /etc/nginx/conf.d/api-gateway.conf**

```nginx
# Model provider routing map
map $http_x_model_provider $backend_pool {
    "openai"     "openai_backends";
    "anthropic"  "anthropic_backends";
    "local"      "local_backends";
    default      "openai_backends";
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name api.vexpertai.local;

    # TLS configuration
    ssl_certificate /etc/nginx/ssl/api-gateway.crt;
    ssl_certificate_key /etc/nginx/ssl/api-gateway.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Rate limiting
    limit_req zone=api_limit burst=50 nodelay;
    limit_conn conn_limit 20;

    # Health check endpoint
    location /health {
        access_log off;
        return 200 '{"status":"healthy","service":"api-gateway"}\n';
        add_header Content-Type application/json;
    }

    # Metrics endpoint (restricted)
    location /metrics {
        access_log off;
        stub_status on;
        allow 10.0.0.0/8;
        deny all;
    }

    # Chat completions endpoint
    location /v1/chat/completions {
        # Generate request ID
        set $request_id $request_id;
        if ($request_id = "") {
            set $request_id $pid-$msec-$remote_addr-$request_length;
        }

        # Proxy to backend pool (selected by header)
        proxy_pass http://$backend_pool;

        # Proxy headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Request-ID $request_id;

        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Retry logic
        proxy_next_upstream error timeout http_502 http_503 http_504;
        proxy_next_upstream_tries 3;
        proxy_next_upstream_timeout 90s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;

        # Response headers
        add_header X-Request-ID $request_id always;
        add_header X-Gateway-Time $request_time always;
        add_header X-Upstream-Addr $upstream_addr always;

        # Error handling
        error_page 502 503 504 @backend_error;
    }

    # Embeddings endpoint
    location /v1/embeddings {
        set $request_id $request_id;
        if ($request_id = "") {
            set $request_id $pid-$msec-$remote_addr-$request_length;
        }

        proxy_pass http://embedding_backends;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Request-ID $request_id;

        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;

        proxy_next_upstream error timeout http_502 http_503 http_504;
        proxy_next_upstream_tries 2;

        add_header X-Request-ID $request_id always;

        error_page 502 503 504 @backend_error;
    }

    # Error handler
    location @backend_error {
        internal;
        add_header Content-Type application/json always;
        add_header X-Request-ID $request_id always;
        add_header Retry-After 30 always;
        return 503 '{"error":{"code":"service_unavailable","message":"Backend temporarily unavailable. Please retry.","request_id":"$request_id"}}\n';
    }
}

# HTTP redirect to HTTPS
server {
    listen 80;
    server_name api.vexpertai.local;
    return 301 https://$server_name$request_uri;
}
```

**Backend service (backend.py):**

```python
#!/usr/bin/env python3
"""
Backend service for API gateway
Simulates LLM API with health checks, logging, and metrics
"""

from flask import Flask, request, jsonify, g
import time
import os
import logging
import random

# Configuration
BACKEND_ID = os.environ.get('BACKEND_ID', '1')
PORT = int(os.environ.get('PORT', 8000))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format=f'[Backend-{BACKEND_ID}] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Metrics
request_count = 0
total_response_time = 0.0

@app.before_request
def before_request():
    """Track request start time and request ID"""
    g.start_time = time.time()
    g.request_id = request.headers.get('X-Request-ID', 'unknown')
    logger.info(f"[{g.request_id}] Incoming: {request.method} {request.path}")

@app.after_request
def after_request(response):
    """Log response metrics"""
    duration = time.time() - g.start_time
    logger.info(f"[{g.request_id}] Response: {response.status_code} in {duration:.3f}s")

    # Update metrics
    global request_count, total_response_time
    request_count += 1
    total_response_time += duration

    # Add response headers
    response.headers['X-Backend-ID'] = BACKEND_ID
    response.headers['X-Request-ID'] = g.request_id
    response.headers['X-Processing-Time'] = f"{duration:.3f}"

    return response

@app.route('/health')
def health_check():
    """Health check endpoint"""
    # Simulate occasional health check failure (5% chance)
    if random.random() < 0.05:
        logger.warning("Health check failed (simulated)")
        return jsonify({
            'status': 'unhealthy',
            'backend_id': BACKEND_ID,
            'reason': 'simulated_failure'
        }), 503

    return jsonify({
        'status': 'healthy',
        'backend_id': BACKEND_ID,
        'requests_served': request_count,
        'avg_response_time': total_response_time / request_count if request_count > 0 else 0
    }), 200

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Chat completion endpoint"""
    data = request.get_json()

    # Validate request
    if not data or 'messages' not in data:
        return jsonify({'error': 'Invalid request: missing messages'}), 400

    model = data.get('model', 'gpt-4')
    messages = data.get('messages', [])

    logger.info(f"[{g.request_id}] Processing chat completion: model={model}, messages={len(messages)}")

    # Simulate processing time (0.5-3 seconds)
    processing_time = random.uniform(0.5, 3.0)
    time.sleep(processing_time)

    # Simulate occasional failures (2% chance)
    if random.random() < 0.02:
        logger.error(f"[{g.request_id}] Simulated backend error")
        return jsonify({'error': 'Internal server error'}), 500

    # Generate response
    response = {
        'id': f'chatcmpl-{int(time.time())}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
        'choices': [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': f'Response from backend {BACKEND_ID}'
            },
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 20,
            'total_tokens': 30
        },
        'backend_id': BACKEND_ID,
        'processing_time': processing_time
    }

    return jsonify(response), 200

@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    """Embeddings endpoint"""
    data = request.get_json()

    if not data or 'input' not in data:
        return jsonify({'error': 'Invalid request: missing input'}), 400

    input_text = data.get('input', '')
    model = data.get('model', 'text-embedding-ada-002')

    logger.info(f"[{g.request_id}] Processing embeddings: model={model}")

    # Simulate processing (faster than chat)
    time.sleep(random.uniform(0.1, 0.5))

    # Generate fake embedding vector
    embedding = [random.random() for _ in range(1536)]

    response = {
        'object': 'list',
        'data': [{
            'object': 'embedding',
            'embedding': embedding,
            'index': 0
        }],
        'model': model,
        'usage': {
            'prompt_tokens': 8,
            'total_tokens': 8
        },
        'backend_id': BACKEND_ID
    }

    return jsonify(response), 200

@app.route('/metrics')
def metrics():
    """Prometheus-style metrics"""
    avg_response_time = total_response_time / request_count if request_count > 0 else 0

    metrics_text = f"""# HELP backend_requests_total Total requests processed
# TYPE backend_requests_total counter
backend_requests_total{{backend_id="{BACKEND_ID}"}} {request_count}

# HELP backend_response_time_seconds Average response time
# TYPE backend_response_time_seconds gauge
backend_response_time_seconds{{backend_id="{BACKEND_ID}"}} {avg_response_time:.3f}
"""

    return metrics_text, 200, {'Content-Type': 'text/plain'}

if __name__ == '__main__':
    logger.info(f"Starting backend {BACKEND_ID} on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
```

**Test the complete setup:**

```bash
# Start backends
python3 backend.py &  # Backend 1 on port 8000
PORT=8001 BACKEND_ID=2 python3 backend.py &  # Backend 2 on port 8001
PORT=8002 BACKEND_ID=3 python3 backend.py &  # Backend 3 on port 8002

# Reload Nginx
sudo nginx -t && sudo nginx -s reload

# Test health check
curl https://api.vexpertai.local/health

# Test chat completion (will round-robin across backends)
for i in {1..10}; do
    curl -k https://api.vexpertai.local/v1/chat/completions \
         -H "Content-Type: application/json" \
         -H "X-Model-Provider: openai" \
         -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}' | \
    jq -r '.backend_id'
done

# Check load distribution
tail -20 /var/log/nginx/access.log | grep chat/completions
```

**Output:**

```
Backend: 1
Backend: 2
Backend: 3
Backend: 1
Backend: 2
Backend: 3
Backend: 1
Backend: 2
Backend: 3
Backend: 1

Gateway metrics:
- Total requests: 10
- Backend 1: 4 requests (weight=3, highest capacity)
- Backend 2: 3 requests (weight=2)
- Backend 3: 3 requests (weight=1, but not backup, so gets traffic)
- Average response time: 1.8s
- Errors: 0
- Retries: 0
```

Perfect load distribution. All backends healthy. Gateway working as expected.

## Operational Considerations

### Capacity Planning

Calculate required gateway capacity:

```
Requests per second (RPS) = (peak concurrent users × avg requests per user per minute) / 60

Example:
- 100 concurrent users
- Each makes 10 API calls per minute (chat, embeddings, follow-ups)
- RPS = (100 × 10) / 60 = 16.7 RPS

With 3 backends:
- Per backend: ~6 RPS
- With 2x headroom: need 12 RPS capacity per backend
```

**Load test with Apache Bench:**

```bash
# Test gateway capacity
ab -n 1000 -c 10 -p request.json -T application/json \
   https://api.vexpertai.local/v1/chat/completions

# request.json:
# {"model":"gpt-4","messages":[{"role":"user","content":"test"}]}
```

**Output:**

```
Requests per second:    45.23 [#/sec] (mean)
Time per request:       221.1 [ms] (mean)
Time per request:       22.1 [ms] (mean, across all concurrent requests)

Percentage of requests served within a certain time (ms)
  50%    210
  66%    245
  75%    280
  80%    310
  90%    390
  95%    450
  98%    520
  99%    580
 100%    750 (longest request)
```

Gateway can handle 45 RPS with 10 concurrent connections. More than enough for 100 users.

### Security Hardening

**API key validation:**

```nginx
# /etc/nginx/conf.d/api-gateway.conf
geo $api_key_valid {
    default 0;
}

map $http_authorization $api_key_valid {
    "Bearer sk-vexpertai-prod-abc123" 1;
    "Bearer sk-vexpertai-dev-xyz789" 1;
    default 0;
}

server {
    listen 443 ssl http2;

    location /v1/chat/completions {
        # Validate API key
        if ($api_key_valid = 0) {
            return 401 '{"error":"Invalid API key"}';
        }

        proxy_pass http://$backend_pool;
    }
}
```

Better approach: Use external auth service (OAuth, JWT validation).

### Disaster Recovery

**Backup gateway configuration:**

```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backup/nginx"
DATE=$(date +%Y%m%d-%H%M%S)

mkdir -p $BACKUP_DIR

# Backup configs
tar -czf $BACKUP_DIR/nginx-config-$DATE.tar.gz /etc/nginx/

# Backup SSL certs
tar -czf $BACKUP_DIR/nginx-ssl-$DATE.tar.gz /etc/nginx/ssl/

# Keep only last 30 days
find $BACKUP_DIR -type f -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR"
```

**Restore process:**

```bash
# Stop Nginx
sudo systemctl stop nginx

# Restore config
sudo tar -xzf /backup/nginx/nginx-config-20260119-143000.tar.gz -C /

# Restore SSL
sudo tar -xzf /backup/nginx/nginx-ssl-20260119-143000.tar.gz -C /

# Test config
sudo nginx -t

# Start Nginx
sudo systemctl start nginx
```

### Multi-Region Deployment

Deploy gateways in multiple regions for HA:

```
         [Global Load Balancer]
                   |
      +------------+------------+
      |                         |
[US Gateway]              [EU Gateway]
      |                         |
[US Backends]            [EU Backends]
```

**Global LB config (AWS Route 53, Cloudflare):**

```
Record: api.vexpertai.com
Type: A
Policy: Latency-based routing

US-East-1: 54.123.45.67 (Nginx gateway)
EU-Central-1: 18.234.56.78 (Nginx gateway)

Health check: HTTPS GET /health every 30s
Failover: Automatic if health check fails
```

Users automatically routed to nearest healthy gateway.

## Summary

You've built a production-grade API gateway using Nginx. This gives you:

1. **Load balancing** - Distribute across multiple LLM providers and models
2. **High availability** - Automatic failover when backends fail
3. **Performance** - Connection pooling, caching, SSL offload
4. **Security** - Single point for authentication, rate limiting, TLS
5. **Visibility** - Detailed logging, metrics, request tracing
6. **Reliability** - Circuit breakers, retry logic, health checks
7. **Flexibility** - Route based on headers, URI, client properties

This is the same architecture used by companies like Netflix, Uber, and AWS. You're applying Layer 4-7 networking skills to API infrastructure.

Your AI agents are more resilient, your costs are controlled (rate limiting prevents runaway API bills), and you can debug production issues with proper logging and tracing.

The gateway becomes your control plane for all AI/ML API traffic. Add a new model provider? Update gateway config, no code changes. Rate limit a rogue service? One Nginx rule. Audit who called what? Parse gateway logs.

You've essentially built a miniature version of what Cloudflare, Akamai, and AWS API Gateway do at scale. Now you understand how those services work under the hood - and you can run your own for $50/month instead of $500/month.

Next chapter: Kubernetes orchestration for your AI workloads (because manually managing 20 backend containers gets old fast).

## Further Reading

**Nginx documentation:**
- Official docs: https://nginx.org/en/docs/
- Load balancing: https://nginx.org/en/docs/http/load_balancing.html
- TLS configuration: https://nginx.org/en/docs/http/configuring_https_servers.html

**OpenResty (Nginx + Lua):**
- https://openresty.org/
- Scripting guide: https://github.com/openresty/lua-nginx-module

**Circuit breaker libraries:**
- pybreaker (Python): https://github.com/danielfm/pybreaker
- resilience4j (Java): https://resilience4j.readme.io/
- Polly (.NET): https://github.com/App-vNext/Polly

**API gateway patterns:**
- "Building Microservices" by Sam Newman (O'Reilly)
- "Release It!" by Michael Nygard (Pragmatic Bookshelf)
- AWS API Gateway docs: https://docs.aws.amazon.com/apigateway/

**Monitoring:**
- Prometheus: https://prometheus.io/docs/
- Grafana: https://grafana.com/docs/
- Loki (log aggregation): https://grafana.com/docs/loki/

You now have the tools to build production-grade API infrastructure. Stop hitting APIs directly from your agents - put a gateway in front of everything. Your future self (debugging a production outage at 2am) will thank you.
