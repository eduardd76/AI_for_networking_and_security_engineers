# Chapter 4: API Basics and Authentication

## Learning Objectives

By the end of this chapter, you will:
- Set up API keys securely (never hardcode secrets)
- Make authenticated API calls to Claude, OpenAI, and Gemini
- Implement proper error handling and retries
- Understand rate limits and how to handle them
- Build a robust API client with exponential backoff
- Monitor API usage and costs in real-time

**Prerequisites**: Chapters 1-3 completed, basic Python knowledge, API keys from providers.

**What You'll Build**: A production-ready API client wrapper that handles authentication, retries, rate limiting, and error handling automatically—the foundation for all your AI networking tools.

---

## The 3 AM Wake-Up Call

It was supposed to be a quiet deployment weekend.

My team had built an AI-powered config compliance checker. It worked flawlessly in testing—scanning hundreds of router configs against our security baseline, flagging violations, generating remediation commands. Management loved the demo. We got approval to run it against production.

Saturday at 11 PM, I kicked off the batch job: 2,847 routers across three data centers. I watched the first hundred complete successfully, felt satisfied, and went to bed.

At 3:17 AM, my phone exploded.

Not the compliance checker failing—that would have been manageable. No, the *entire* AI-assisted operations pipeline was down. The chatbot that helped NOC handle tickets? Dead. The log analyzer catching anomalies? Dead. The documentation generator? You guessed it.

The culprit: my batch job had burned through our API rate limit. Every subsequent request from every tool got a 429 "Too Many Requests" response. And none of our code handled that error gracefully—they all just crashed.

It took four hours to restore service. The postmortem was embarrassing. We'd built clever AI tools without understanding the fundamental reliability engineering that APIs require.

This chapter exists so you don't learn these lessons at 3 AM.

---

## APIs Fail: Accept This Truth

Here's what nobody tells you when you start building with AI APIs:

**They are not as reliable as you think.**

We're used to calling internal APIs that we control. Database queries that either work or throw a sensible exception. Network devices that respond consistently to CLI commands.

Cloud AI APIs are different:

**Rate limits are real and aggressive.** Free tier Anthropic allows 5 requests per minute. That's one request every 12 seconds. Miss that window and you get a 429. GPT-4's free tier is even stingier—3 requests per minute.

**Latency varies wildly.** The same Claude Sonnet request might take 3 seconds at 2 PM and 15 seconds at 2 AM when batch jobs worldwide are running.

**Services have outages.** Anthropic, OpenAI, Google—they all have had multi-hour outages. Your code needs to handle "the API is simply gone" gracefully.

**Costs can spiral.** A bug that retries infinitely can run up hundreds of dollars before you notice. A prompt that accidentally includes duplicate data can 10x your token costs.

The difference between a demo that works and a production system that stays up is all in how you handle these realities.

---

## API Authentication: A Networking Perspective

### The Mental Model: API Keys Are Like Pre-Shared Keys

If you've configured IPsec VPNs, you understand pre-shared keys (PSK):

```cisco
crypto isakmp key MySecretKey123 address 203.0.113.5
```

API keys work exactly the same way:

1. **You generate a secret** (the API key) during registration
2. **Every request includes this secret** (typically in a header)
3. **The server validates** before processing
4. **If invalid → rejected** (HTTP 401 Unauthorized)

And just like PSKs, the security rules are identical:

| PSK Best Practice | API Key Equivalent |
|-------------------|-------------------|
| Never put in running-config that gets backed up unencrypted | Never commit to Git |
| Rotate every 90 days | Rotate every 90 days |
| Different keys per peer/site | Different keys per environment (dev/staging/prod) |
| Store in secure vault | Use secrets manager |
| Monitor for unauthorized use | Set up usage alerts |

If you'd never hardcode a VPN PSK directly in a script, don't hardcode an API key either.

---

## The Three Levels of API Key Security

### Level 1: The Dangerous Way (Don't Do This)

```python
# ⛔ DANGER: Hardcoded secret
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-api03-abcdef123456")
```

I've seen this in production code. I've seen it committed to public GitHub repos. I've seen companies get their API keys scraped by bots within minutes of pushing to GitHub.

**What happens**: Your code goes into version control. Someone (or some bot) finds the key. They run up your bill mining crypto prompts or testing exploits. You find out when you get a $10,000 invoice or when your account gets suspended.

### Level 2: Environment Variables (Minimum Acceptable)

```python
# ✅ Acceptable: Environment variables
import os
from anthropic import Anthropic

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not set")

client = Anthropic(api_key=api_key)
```

**Setting the variable**:

```bash
# Linux/macOS - add to ~/.bashrc or ~/.zshrc for persistence
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"

# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-your-key-here"

# Windows CMD (persistent via System Properties → Environment Variables)
setx ANTHROPIC_API_KEY "sk-ant-api03-your-key-here"
```

**Better: Use a `.env` file**

Create `.env` in your project root:
```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
OPENAI_API_KEY=sk-proj-your-key-here
GOOGLE_API_KEY=AIzaSy-your-key-here
```

**Critical: Add to `.gitignore` immediately**:
```bash
echo ".env" >> .gitignore
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file

anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
```

This is secure enough for development and small-scale production. But for enterprise...

### Level 3: Secrets Manager (Production Standard)

In production systems—especially anything handling customer data or running in cloud infrastructure—use a proper secrets manager:

- **AWS Secrets Manager**
- **HashiCorp Vault**
- **Azure Key Vault**
- **Google Secret Manager**

Example with AWS Secrets Manager:

```python
import boto3
import json

def get_api_keys():
    """Retrieve API keys from AWS Secrets Manager."""
    client = boto3.client('secretsmanager', region_name='us-east-1')
    
    try:
        response = client.get_secret_value(SecretId='prod/ai-tools/api-keys')
        secrets = json.loads(response['SecretString'])
        return {
            'anthropic': secrets['anthropic_api_key'],
            'openai': secrets['openai_api_key'],
        }
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve secrets: {e}")

# Usage
keys = get_api_keys()
client = Anthropic(api_key=keys['anthropic'])
```

**Why this matters**:
- Keys are never on disk
- Rotation doesn't require code deploys
- Access is audited
- Permissions are granular (IAM policies)
- Works across multiple services automatically

---

## Your First API Calls

Let's make real API calls to the three major providers. Same task—explain a networking concept—to see how each API works.

### Claude (Anthropic)

```python
#!/usr/bin/env python3
"""
Basic Claude API call.
The foundation for everything in this book.
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

def ask_claude(question: str) -> str:
    """
    Ask Claude a question and get a response.
    
    Args:
        question: What you want to know
        
    Returns:
        Claude's response as a string
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0,  # Deterministic output
        messages=[
            {"role": "user", "content": question}
        ]
    )
    
    return response.content[0].text


if __name__ == "__main__":
    answer = ask_claude("Explain BGP in one sentence for a network engineer.")
    print(answer)
    # BGP (Border Gateway Protocol) is the path-vector routing protocol that 
    # exchanges routing information between autonomous systems to determine 
    # the best paths for traffic across the internet.
```

### GPT-4 (OpenAI)

```python
#!/usr/bin/env python3
"""
Basic OpenAI API call.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def ask_gpt(question: str) -> str:
    """
    Ask GPT-4 a question and get a response.
    
    Args:
        question: What you want to know
        
    Returns:
        GPT's response as a string
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=0,
        max_tokens=1024
    )
    
    return response.choices[0].message.content


if __name__ == "__main__":
    answer = ask_gpt("Explain OSPF in one sentence for a network engineer.")
    print(answer)
```

### Gemini (Google)

```python
#!/usr/bin/env python3
"""
Basic Gemini API call.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def ask_gemini(question: str) -> str:
    """
    Ask Gemini a question and get a response.
    
    Args:
        question: What you want to know
        
    Returns:
        Gemini's response as a string
    """
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(question)
    
    return response.text


if __name__ == "__main__":
    answer = ask_gemini("Explain EIGRP in one sentence for a network engineer.")
    print(answer)
```

Each provider has slightly different API structures, but the pattern is the same:
1. Initialize client with API key
2. Create a message/request with your prompt
3. Extract the text from the response

---

## Understanding Rate Limits

### The QoS Analogy

In networking, we use QoS to prevent any single flow from consuming all bandwidth:

```cisco
policy-map RATE-LIMIT
 class VOICE
  police 128000 conform-action transmit exceed-action drop
```

API rate limits work identically:

| QoS Concept | API Equivalent |
|-------------|----------------|
| Committed Information Rate (CIR) | Requests Per Minute (RPM) |
| Burst size | Token bucket |
| Exceed action: drop | 429 Too Many Requests |
| Police per-flow | Limit per API key |

When you exceed your rate limit, you don't get "slow responses"—you get rejected. Just like traffic policing drops packets that exceed the rate.

### Real Rate Limit Numbers

| Provider | Tier | Requests/Min | Tokens/Min |
|----------|------|--------------|------------|
| **Anthropic** | Free | 5 | 10,000 |
| | Tier 1 ($5 spend) | 50 | 40,000 |
| | Tier 2 ($40 spend) | 1,000 | 400,000 |
| **OpenAI** | Free | 3 | 40,000 |
| | Tier 1 | 500 | 200,000 |
| | Tier 2 | 5,000 | 2,000,000 |

**The trap**: You build and test on paid tier limits (1,000 RPM). You give the tool to a colleague who's on free tier (5 RPM). Their requests fail constantly and they think your tool is broken.

Always test with the lowest tier your users might have.

---

## Error Handling Done Right

### The Two Types of Errors

**Transient errors** (retry these):
- 429 Rate Limit Exceeded
- 500 Internal Server Error
- 502 Bad Gateway
- 503 Service Unavailable
- Network timeouts
- Connection resets

**Permanent errors** (don't retry):
- 400 Bad Request (your prompt is malformed)
- 401 Unauthorized (bad API key)
- 403 Forbidden (no permission for this model)
- 404 Not Found (model doesn't exist)

The critical insight: **retrying a permanent error is pointless and expensive**. Retrying a transient error is essential for reliability.

### Exponential Backoff: The TCP Analogy

You know how TCP congestion control works:
1. Packet loss detected (congestion signal)
2. Reduce sending rate
3. Gradually probe for more capacity
4. Repeat

API retry logic works the same way:

```
Attempt 1: Request fails (rate limit)
→ Wait 1 second

Attempt 2: Request fails (still rate limited)
→ Wait 2 seconds (1 × 2¹)

Attempt 3: Request fails (still rate limited)
→ Wait 4 seconds (1 × 2²)

Attempt 4: Request succeeds!
```

This is **exponential backoff**. The delay doubles after each failure, giving the system time to recover without hammering it with retry attempts.

### Production-Grade Error Handler

```python
#!/usr/bin/env python3
"""
Production API client with comprehensive error handling.
"""

import os
import time
import logging
from typing import Optional
from anthropic import Anthropic, APIError, RateLimitError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def call_with_retry(
    prompt: str,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0
) -> Optional[str]:
    """
    Make API call with exponential backoff retry logic.
    
    Args:
        prompt: The prompt to send
        max_retries: Maximum number of retry attempts
        initial_delay: Starting delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
    
    Returns:
        Response text, or None if all retries exhausted
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1}")
            
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            logger.info("Request succeeded")
            return response.content[0].text
            
        except RateLimitError:
            # Rate limit - definitely retry with backoff
            delay = min(initial_delay * (2 ** attempt), max_delay)
            logger.warning(f"Rate limited. Waiting {delay:.1f}s before retry...")
            
            if attempt < max_retries:
                time.sleep(delay)
            else:
                logger.error("Max retries exceeded for rate limit")
                return None
                
        except APIConnectionError as e:
            # Network error - retry with backoff
            delay = min(initial_delay * (2 ** attempt), max_delay)
            logger.warning(f"Connection error: {e}. Waiting {delay:.1f}s...")
            
            if attempt < max_retries:
                time.sleep(delay)
            else:
                logger.error("Max retries exceeded for connection error")
                return None
                
        except APIError as e:
            # Check if it's a retryable server error (5xx)
            if hasattr(e, 'status_code') and 500 <= e.status_code < 600:
                delay = min(initial_delay * (2 ** attempt), max_delay)
                logger.warning(f"Server error ({e.status_code}). Waiting {delay:.1f}s...")
                
                if attempt < max_retries:
                    time.sleep(delay)
                else:
                    logger.error("Max retries exceeded for server error")
                    return None
            else:
                # Permanent error - don't retry
                logger.error(f"Permanent API error: {e}")
                return None
                
        except Exception as e:
            # Unexpected error - log and fail
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            return None
    
    return None
```

---

## Building a Production API Client

Let's assemble everything into a reusable client class:

```python
#!/usr/bin/env python3
"""
ResilientAPIClient - Production-grade API client for network automation.

Features:
- Automatic retry with exponential backoff
- Rate limit handling
- Usage tracking and cost monitoring
- Comprehensive error handling
- Multi-provider support ready
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from anthropic import Anthropic, APIError, RateLimitError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class UsageMetrics:
    """Track API usage statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limit_hits: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_seconds: float = 0.0


class ResilientAPIClient:
    """
    Production-ready API client with built-in resilience.
    
    This is the foundation for all AI networking tools.
    """
    
    # Pricing per million tokens (update as needed)
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-20250514": {"input": 0.25, "output": 1.25},
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: int = 120
    ):
        """
        Initialize the resilient API client.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            max_retries: Maximum retry attempts for transient failures
            initial_delay: Starting delay for exponential backoff
            max_delay: Maximum delay between retries
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = Anthropic(api_key=self.api_key, timeout=timeout)
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.metrics = UsageMetrics()
    
    def call(
        self,
        prompt: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make an API call with automatic retry and monitoring.
        
        Args:
            prompt: User message/prompt
            model: Model to use
            max_tokens: Maximum response tokens
            temperature: Randomness (0 = deterministic)
            system: Optional system prompt
            
        Returns:
            Dict with response and metadata, or None on failure
        """
        # Build request
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        if system:
            kwargs["system"] = system
        
        # Attempt with retries
        for attempt in range(self.max_retries + 1):
            self.metrics.total_requests += 1
            
            try:
                start_time = time.time()
                response = self.client.messages.create(**kwargs)
                latency = time.time() - start_time
                
                # Extract data
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                text = response.content[0].text
                
                # Calculate cost
                pricing = self.PRICING.get(model, {"input": 3.0, "output": 15.0})
                cost = (input_tokens / 1_000_000) * pricing["input"]
                cost += (output_tokens / 1_000_000) * pricing["output"]
                
                # Update metrics
                self.metrics.successful_requests += 1
                self.metrics.total_input_tokens += input_tokens
                self.metrics.total_output_tokens += output_tokens
                self.metrics.total_cost_usd += cost
                self.metrics.total_latency_seconds += latency
                
                logger.info(
                    f"Success: {input_tokens} in, {output_tokens} out, "
                    f"${cost:.4f}, {latency:.1f}s"
                )
                
                return {
                    "text": text,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost,
                    "latency_seconds": latency,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except RateLimitError:
                self.metrics.rate_limit_hits += 1
                if not self._handle_retry(attempt, "Rate limit exceeded"):
                    self.metrics.failed_requests += 1
                    return None
                    
            except APIConnectionError as e:
                if not self._handle_retry(attempt, f"Connection error: {e}"):
                    self.metrics.failed_requests += 1
                    return None
                    
            except APIError as e:
                if hasattr(e, 'status_code') and 500 <= e.status_code < 600:
                    if not self._handle_retry(attempt, f"Server error: {e.status_code}"):
                        self.metrics.failed_requests += 1
                        return None
                else:
                    logger.error(f"Permanent error: {e}")
                    self.metrics.failed_requests += 1
                    return None
                    
            except Exception as e:
                logger.error(f"Unexpected error: {type(e).__name__}: {e}")
                self.metrics.failed_requests += 1
                return None
        
        return None
    
    def _handle_retry(self, attempt: int, reason: str) -> bool:
        """Handle retry logic with exponential backoff."""
        if attempt >= self.max_retries:
            logger.error(f"{reason} - max retries exceeded")
            return False
        
        delay = min(self.initial_delay * (2 ** attempt), self.max_delay)
        logger.warning(f"{reason} - retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
        time.sleep(delay)
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get usage metrics summary."""
        m = self.metrics
        success_rate = (m.successful_requests / max(m.total_requests, 1)) * 100
        avg_latency = m.total_latency_seconds / max(m.successful_requests, 1)
        
        return {
            "total_requests": m.total_requests,
            "successful": m.successful_requests,
            "failed": m.failed_requests,
            "success_rate": f"{success_rate:.1f}%",
            "rate_limit_hits": m.rate_limit_hits,
            "total_tokens": m.total_input_tokens + m.total_output_tokens,
            "total_cost": f"${m.total_cost_usd:.4f}",
            "avg_latency": f"{avg_latency:.2f}s"
        }
    
    def reset_metrics(self):
        """Reset usage metrics."""
        self.metrics = UsageMetrics()


# Example usage and testing
if __name__ == "__main__":
    print("🔧 Testing ResilientAPIClient")
    print("=" * 60)
    
    # Initialize
    client = ResilientAPIClient(max_retries=3)
    
    # Test prompts
    tests = [
        "What is a VLAN? One sentence.",
        "Explain BGP in 20 words or less.",
        "What does OSPF stand for?"
    ]
    
    for prompt in tests:
        print(f"\n📤 Prompt: {prompt}")
        result = client.call(prompt, max_tokens=100)
        
        if result:
            print(f"📥 Response: {result['text']}")
            print(f"   Tokens: {result['input_tokens']} → {result['output_tokens']}")
            print(f"   Cost: ${result['cost_usd']:.4f} | Latency: {result['latency_seconds']:.1f}s")
        else:
            print("❌ Request failed")
    
    # Print session summary
    print("\n" + "=" * 60)
    print("📊 Session Metrics")
    print("=" * 60)
    for key, value in client.get_metrics().items():
        print(f"   {key}: {value}")
```

---

## Client-Side Rate Limiting

Sometimes you want to throttle yourself to avoid hitting provider limits:

```python
import time
from collections import deque
from functools import wraps


class RateLimiter:
    """
    Token bucket rate limiter.
    Prevents exceeding API rate limits by throttling requests client-side.
    """
    
    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.max_requests = requests_per_minute
        self.window_seconds = 60.0
        self.request_times = deque()
    
    def acquire(self):
        """
        Acquire permission to make a request.
        Blocks if rate limit would be exceeded.
        """
        now = time.time()
        
        # Remove requests outside the time window
        while self.request_times and self.request_times[0] < now - self.window_seconds:
            self.request_times.popleft()
        
        # If at limit, wait
        if len(self.request_times) >= self.max_requests:
            wait_time = self.window_seconds - (now - self.request_times[0])
            if wait_time > 0:
                print(f"⏳ Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                return self.acquire()
        
        # Record this request
        self.request_times.append(time.time())
    
    def __call__(self, func):
        """Use as decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)
        return wrapper


# Usage example
limiter = RateLimiter(requests_per_minute=50)

@limiter
def make_api_call(prompt):
    # Your API call here
    pass
```

---

## Monitoring and Cost Tracking

Track every API call for debugging and cost management:

```python
import json
from pathlib import Path
from datetime import datetime


class APIUsageLogger:
    """Log all API calls for monitoring and debugging."""
    
    def __init__(self, log_file: str = "api_usage.jsonl"):
        self.log_path = Path(log_file)
    
    def log(self, **kwargs):
        """Log an API call."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def get_daily_summary(self) -> dict:
        """Get today's usage summary."""
        today = datetime.utcnow().date().isoformat()
        total_cost = 0.0
        total_requests = 0
        
        if not self.log_path.exists():
            return {"date": today, "requests": 0, "cost": "$0.00"}
        
        with open(self.log_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry["timestamp"].startswith(today):
                    total_requests += 1
                    total_cost += entry.get("cost_usd", 0)
        
        return {
            "date": today,
            "requests": total_requests,
            "cost": f"${total_cost:.2f}"
        }
```

---

## Common Errors and Fixes

### "Invalid API Key"
```python
anthropic.AuthenticationError: Invalid API key
```

**Diagnosis checklist**:
1. Is the key set? `echo $ANTHROPIC_API_KEY`
2. Does it start with `sk-ant-api03-`?
3. Was it copy-pasted correctly (no extra spaces)?
4. Has it been revoked?

### "Rate Limit Exceeded"
```python
anthropic.RateLimitError: rate_limit_error
```

**Solutions**:
- Implement exponential backoff (see above)
- Add client-side rate limiting
- Upgrade your tier
- Spread requests over time

### "Context Length Exceeded"
```python
anthropic.BadRequestError: prompt is too long
```

**Solutions**:
- Count tokens before sending (Chapter 2)
- Chunk large inputs (Chapter 7)
- Use a model with larger context

### "Request Timeout"
```python
anthropic.APIConnectionError: Connection timeout
```

**Solutions**:
- Increase timeout: `Anthropic(timeout=120)`
- Check network connectivity
- Retry with backoff

---

## Lab Exercises

### Lab 1: Error Simulation (30 minutes)
Create a test harness that simulates various API errors. Verify your retry logic handles each correctly.

### Lab 2: Cost Dashboard (45 minutes)
Build a simple dashboard that reads from `api_usage.jsonl` and displays:
- Requests per hour
- Cost per day
- Most expensive prompts
- Error rate

### Lab 3: Multi-Provider Client (60 minutes)
Extend `ResilientAPIClient` to support OpenAI and Gemini. Use a factory pattern:
```python
client = APIClient.create(provider="openai")
```

### Lab 4: Budget Enforcer (45 minutes)
Add a daily budget limit to the client. When 80% of budget is used, log a warning. When 100% is reached, reject new requests.

### Lab 5: Metrics to Prometheus (90 minutes)
Export the usage metrics in Prometheus format so they can be scraped and displayed in Grafana.

---

## Key Takeaways

### 1. Security First
- Never hardcode API keys
- Use environment variables minimum, secrets manager for production
- Rotate keys regularly
- Monitor for unauthorized usage

### 2. APIs Fail—Plan for It
- Implement retry with exponential backoff
- Distinguish transient vs permanent errors
- Set reasonable timeouts
- Have fallback behavior

### 3. Monitor Everything
- Log every request
- Track costs in real-time
- Set budget alerts
- Measure latency and success rate

### 4. Rate Limits Are Real
- Know your tier's limits
- Implement client-side throttling
- Use backoff on 429 errors
- Test with lower-tier limits

### 5. Build Reusable Infrastructure
- The `ResilientAPIClient` pattern works everywhere
- Consistent error handling across tools
- Centralized metrics and logging
- Easy to swap providers

---

## What's Next

You now have production-grade API client code that handles the real-world challenges of working with AI APIs. This is the foundation everything else builds on.

But making API calls is only half the battle. The other half is crafting prompts that actually work. A bad prompt to a great API client still gives bad results.

Chapter 5 covers prompt engineering—the art and science of getting AI models to do exactly what you need for networking tasks.

**Ready?** → Chapter 5: Prompt Engineering Fundamentals

---

## Quick Reference

### API Key Setup
```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=AIzaSy...
```

### Basic Call Pattern
```python
from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}]
)
text = response.content[0].text
```

### Retry with Backoff
```python
for attempt in range(max_retries):
    try:
        return make_request()
    except RateLimitError:
        delay = initial_delay * (2 ** attempt)
        time.sleep(delay)
```

### Rate Limit Tiers
| Provider | Free | Paid |
|----------|------|------|
| Anthropic | 5 RPM | 50-1000 RPM |
| OpenAI | 3 RPM | 500-5000 RPM |

---

**Chapter Status**: Complete  
**Word Count**: ~5,200  
**Code**: Production-ready `ResilientAPIClient`  
**Estimated Reading Time**: 30 minutes
