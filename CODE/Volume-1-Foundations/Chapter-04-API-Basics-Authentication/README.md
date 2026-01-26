# Chapter 4: API Basics and Authentication

> Build production-ready API clients that handle the real world—rate limits, errors, retries, and cost tracking.

## What You'll Build

A **ResilientAPIClient** that:
- Automatically retries on transient failures
- Handles rate limits with exponential backoff
- Tracks usage and costs in real-time
- Provides comprehensive error handling

This is the foundation for all AI networking tools.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Check with `python --version` |
| anthropic | `pip install anthropic` |
| API key | Set `ANTHROPIC_API_KEY` environment variable |

## Quick Start

```bash
# Navigate to Chapter 4
cd CODE/Volume-1-Foundations/Chapter-04-API-Basics-Authentication

# Install dependencies
pip install anthropic python-dotenv

# Set API key
export ANTHROPIC_API_KEY=sk-ant-api03-your-key

# Run the demo
python resilient_api_client.py
```

## Sample Output

```
🔧 ResilientAPIClient Demo
   Chapter 4: API Basics and Authentication
============================================================

📤 Prompt: What is a VLAN?
2024-01-26 23:30:15 - INFO - Success: 15 in, 42 out, $0.0007, 2.3s
📥 Response: A VLAN (Virtual Local Area Network) is a logical grouping...
   📊 15 → 42 tokens
   💰 $0.0007 | ⏱️ 2.3s

📤 Prompt: Explain BGP in 20 words.
2024-01-26 23:30:18 - INFO - Success: 18 in, 28 out, $0.0005, 1.8s
📥 Response: BGP (Border Gateway Protocol) exchanges routing information...
   📊 18 → 28 tokens
   💰 $0.0005 | ⏱️ 1.8s

============================================================
📊 Session Metrics
============================================================
   total_requests      : 3
   successful          : 3
   failed              : 0
   success_rate        : 100.0%
   rate_limit_hits     : 0
   total_tokens        : 170
   total_cost          : $0.0019
   avg_latency         : 2.05s
```

## Using the Client in Your Code

```python
from resilient_api_client import ResilientAPIClient

# Initialize
client = ResilientAPIClient(
    max_retries=3,           # Retry up to 3 times
    initial_delay=1.0,       # Start with 1s backoff
    timeout=120              # 2 minute timeout
)

# Make a call
result = client.call(
    prompt="Analyze this config for security issues: ...",
    model="claude-sonnet-4-20250514",
    max_tokens=2000,
    temperature=0
)

if result:
    print(result["text"])        # The response
    print(result["cost_usd"])    # How much it cost
    print(result["latency_seconds"])  # How long it took
else:
    print("Request failed after retries")

# Check session metrics
metrics = client.get_metrics()
print(f"Total cost: {metrics['total_cost']}")
```

## Key Features

### Exponential Backoff
```
Attempt 1 fails → wait 1s
Attempt 2 fails → wait 2s
Attempt 3 fails → wait 4s
Attempt 4 fails → give up
```

### Automatic Error Classification
- **Retry**: 429 (rate limit), 5xx (server errors), timeouts
- **Don't retry**: 400 (bad request), 401 (auth), 403 (forbidden)

### Built-in Cost Tracking
Tracks tokens and calculates costs in real-time. No surprise bills.

## Files

| File | Purpose |
|------|---------|
| `resilient_api_client.py` | Main client class + demo |
| `README.md` | This file |

## API Key Security

### ❌ Never do this
```python
client = Anthropic(api_key="sk-ant-api03-...")  # Hardcoded!
```

### ✅ Use environment variables
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-your-key
```

### ✅ Or use a .env file
```bash
# .env (add to .gitignore!)
ANTHROPIC_API_KEY=sk-ant-api03-your-key
```

## Troubleshooting

### "API key required"
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-your-key
```

### "anthropic package not installed"
```bash
pip install anthropic
```

### "Rate limit exceeded" repeatedly
- Free tier: only 5 requests/minute
- Upgrade tier or add longer delays
- Use client-side rate limiting

### Timeout errors
```python
# Increase timeout
client = ResilientAPIClient(timeout=180)  # 3 minutes
```

## Lab Exercises

1. **Test retry logic**: Simulate rate limits and verify backoff works
2. **Add OpenAI support**: Extend client for multiple providers
3. **Budget enforcer**: Add daily spending limit
4. **Prometheus metrics**: Export metrics for monitoring

## Next Steps

→ **Chapter 5:** Prompt Engineering Fundamentals

---

**Full chapter:** See `Volume-1-Foundations/Chapters/Chapter-04-API-Basics-Authentication.md`
