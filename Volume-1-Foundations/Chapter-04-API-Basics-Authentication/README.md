# Chapter 4: API Basics and Authentication

Production-ready API client infrastructure with authentication, retry logic, rate limiting, and usage tracking.

## Files in This Chapter

| File | Description | Lines | Key Features |
|------|-------------|-------|--------------|
| `resilient_api_client.py` | Production API client | ~450 | Retry logic, metrics, cost tracking |
| `rate_limiter.py` | Client-side rate limiting | ~350 | Token bucket, decorator support |
| `usage_tracker.py` | Usage and cost tracking | ~380 | Reports, forecasting, budgets |
| `secret_manager.py` | Secrets management | (TBD) | AWS/Azure/GCP integration |

## Quick Start

```bash
# Setup
cd Chapter-04-API-Basics-Authentication
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install anthropic python-dotenv

# Configure
cp ../.env.example ../.env
# Edit ../.env and add your ANTHROPIC_API_KEY

# Run examples
python resilient_api_client.py
python rate_limiter.py
python usage_tracker.py
```

## resilient_api_client.py

Production-grade API client that handles all failure modes.

**Features**:
- ✅ Exponential backoff retry (up to 3 attempts)
- ✅ Rate limit handling (429 errors)
- ✅ Connection error recovery
- ✅ Usage metrics tracking
- ✅ Cost calculation per request
- ✅ Latency monitoring
- ✅ JSON logs for analysis

**Usage**:
```python
from resilient_api_client import ResilientAPIClient

# Initialize client
client = ResilientAPIClient(
    max_retries=3,
    initial_retry_delay=1.0,
    timeout=60
)

# Make API call
result = client.call(
    prompt="Explain BGP in one sentence",
    max_tokens=200,
    temperature=0.0
)

if result:
    print(result['text'])
    print(f"Cost: ${result['cost']:.6f}")
    print(f"Latency: {result['latency']:.2f}s")

# Get session metrics
metrics = client.get_metrics()
print(f"Success rate: {metrics['success_rate']}")
print(f"Total cost: {metrics['total_cost']}")
```

**Error Handling**:
```python
# Handles automatically:
- RateLimitError (429) → Retry with backoff
- APIConnectionError → Retry
- 5xx Server Errors → Retry
- 4xx Client Errors → Don't retry (permanent)
```

**Output Example**:
```
INFO:__main__:API call attempt 1/4 (model=claude-3-5-sonnet-20241022, temp=0.0)
INFO:__main__:API call succeeded in 2.34s (tokens: 24 in, 38 out, cost: $0.000642)
```

## rate_limiter.py

Client-side rate limiting to prevent hitting API provider limits.

**Features**:
- ✅ Token bucket algorithm
- ✅ Configurable requests per time window
- ✅ Decorator support
- ✅ Thread-safe
- ✅ Non-blocking mode
- ✅ Token + Request limiting

**Usage**:

### Simple Rate Limiter
```python
from rate_limiter import RateLimiter

# Allow 50 requests per minute
limiter = RateLimiter(
    max_requests=50,
    time_window=60.0
)

# Use as decorator
@limiter
def make_api_call():
    # Your API call here
    pass

# Or call explicitly
limiter.acquire()  # Blocks until allowed
result = api_call()
```

### Token Bucket Limiter
```python
from rate_limiter import TokenBucketRateLimiter

# Limit both requests (RPM) and tokens (TPM)
limiter = TokenBucketRateLimiter(
    max_requests_per_minute=50,
    max_tokens_per_minute=40000
)

# Acquire with estimated tokens
if limiter.acquire(estimated_tokens=1500):
    result = api_call()

    # Update with actual usage
    limiter.record_actual_tokens(actual_tokens=1523)
```

**Non-Blocking Mode**:
```python
# Don't wait if limit reached
if limiter.acquire(block=False):
    make_request()
else:
    print("Rate limit reached, skipping request")
```

## usage_tracker.py

Track API usage, costs, and generate budget reports.

**Features**:
- ✅ JSONL logging (one request per line)
- ✅ Time-based summaries
- ✅ Cost breakdowns by model and task
- ✅ Hourly usage reports
- ✅ Monthly cost estimates
- ✅ Budget enforcement
- ✅ CSV/JSON export

**Usage**:
```python
from usage_tracker import UsageTracker

tracker = UsageTracker(log_file="api_usage.jsonl")

# Log a request
tracker.log_request(
    model="claude-3-5-sonnet",
    input_tokens=100,
    output_tokens=200,
    cost=0.003,
    latency=2.5,
    success=True,
    task="config_analysis"
)

# Get summary
summary = tracker.get_summary(hours=24)
print(f"Total cost: {summary['total_cost']}")
print(f"Total requests: {summary['total_requests']}")

# Monthly estimate
estimate = tracker.estimate_monthly_cost(days_sample=7)
print(f"Estimated monthly: {estimate['estimated_monthly']}")

# Budget check
status = tracker.check_budget(daily_limit=50.0)
print(f"Budget status: {status['status']}")
print(f"Remaining: {status['remaining']}")
```

**Reports Available**:
1. **Time-based summary** - Last N hours/days
2. **Hourly breakdown** - Usage by hour
3. **Cost by model** - Which models cost most
4. **Cost by task** - Which tasks cost most
5. **Monthly forecast** - Projected costs
6. **Budget alerts** - Warn at 80% usage

## Integration Example

Combine all three components:

```python
from resilient_api_client import ResilientAPIClient
from rate_limiter import TokenBucketRateLimiter
from usage_tracker import UsageTracker

# Setup
limiter = TokenBucketRateLimiter(
    max_requests_per_minute=50,
    max_tokens_per_minute=40000
)
client = ResilientAPIClient()
tracker = UsageTracker()

# Make rate-limited API calls
def analyze_config(config_text: str):
    # Wait for rate limit
    estimated_tokens = len(config_text) // 4  # Rough estimate
    limiter.acquire(estimated_tokens=estimated_tokens)

    # Make API call
    result = client.call(
        prompt=f"Analyze this config:\n{config_text}",
        model="claude-3-5-sonnet-20241022"
    )

    if result:
        # Track usage
        tracker.log_request(
            model=result['model'],
            input_tokens=result['input_tokens'],
            output_tokens=result['output_tokens'],
            cost=result['cost'],
            latency=result['latency'],
            success=True,
            task="config_analysis"
        )

        # Update rate limiter with actual usage
        actual_tokens = result['input_tokens'] + result['output_tokens']
        limiter.record_actual_tokens(actual_tokens)

        return result['text']

    return None
```

## Cost Reference

**Claude 3.5 Sonnet** (January 2026):
- Input: $3 / million tokens
- Output: $15 / million tokens

**Claude 3.5 Haiku** (January 2026):
- Input: $0.80 / million tokens
- Output: $4 / million tokens

**Example Calculations**:
```python
# Config analysis (200 input + 400 output)
# Sonnet: (200/1M * $3) + (400/1M * $15) = $0.0066
# Haiku:  (200/1M * $0.80) + (400/1M * $4) = $0.0018
# Savings with Haiku: 73%
```

## Testing

```bash
# Test API client
python resilient_api_client.py

# Test rate limiter
python rate_limiter.py

# Test usage tracker
python usage_tracker.py

# Run unit tests
pytest test_chapter04.py -v
```

## Common Issues

### "Invalid API Key"
```
anthropic.AuthenticationError: Invalid API key
```
**Fix**: Check `.env` file has valid `ANTHROPIC_API_KEY`

### "Rate Limit Exceeded"
```
anthropic.RateLimitError: rate_limit_error
```
**Fix**: Use `RateLimiter` to stay under provider limits

### "Context Length Exceeded"
```
anthropic.BadRequestError: messages: total length exceeds maximum
```
**Fix**: Reduce `max_tokens` or chunk input (see Chapter 7)

## Best Practices

✅ **Always use retry logic** - Networks fail, APIs have outages
✅ **Track costs** - API usage can get expensive quickly
✅ **Rate limit client-side** - Don't rely on provider limits
✅ **Log everything** - Debugging requires data
✅ **Set timeouts** - Don't wait forever
✅ **Monitor metrics** - Success rate, latency, costs

## Production Checklist

- [ ] API keys stored in secrets manager (not .env)
- [ ] Retry logic configured (max 3 attempts)
- [ ] Rate limiting enabled
- [ ] Usage tracking to database (not local file)
- [ ] Cost alerts configured (e.g., SNS, email)
- [ ] Logs shipped to central logging (e.g., CloudWatch)
- [ ] Metrics exported to monitoring (e.g., Prometheus)
- [ ] Timeouts set appropriately
- [ ] Error handling tested
- [ ] Budget limits enforced

## Next Steps

- **Chapter 5**: Prompt Engineering - Write effective prompts
- **Chapter 6**: Structured Outputs - Get JSON responses
- **Chapter 7**: Context Management - Handle large configs
- **Chapter 8**: Cost Optimization - Reduce costs 50-70%

## Resources

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Rate Limiting Best Practices](https://cloud.google.com/architecture/rate-limiting-strategies-techniques)
- [Exponential Backoff RFC](https://datatracker.ietf.org/doc/html/rfc2616#section-8.1.4)

---

**Chapter 4 Complete** | Production-ready API infrastructure ✅
