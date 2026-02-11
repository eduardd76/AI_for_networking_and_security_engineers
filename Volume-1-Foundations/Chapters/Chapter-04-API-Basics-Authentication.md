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
# WARNING: DANGER - Hardcoded secret
# This will be detected by GitHub secret scanning bots within minutes if committed
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-api03-abcdef123456")
```

I've seen this in production code. I've seen it committed to public GitHub repos. I've seen companies get their API keys scraped by bots within minutes of pushing to GitHub.

**What happens**: Your code goes into version control. Someone (or some bot) finds the key. They run up your bill mining crypto prompts or testing exploits. You find out when you get a $10,000 invoice or when your account gets suspended.

### Level 2: Environment Variables (Minimum Acceptable)

```python
# ACCEPTABLE: Environment variables
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
        model="claude-sonnet-4.5",  # Using standard model alias
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

    # Output:
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

    # Output:
    # OSPF (Open Shortest Path First) is a link-state routing protocol that
    # uses Dijkstra's algorithm to calculate the shortest path and maintains
    # a complete topology database of the network.

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

    # Output:
    # EIGRP (Enhanced Interior Gateway Routing Protocol) is a Cisco proprietary
    # advanced distance-vector routing protocol that uses DUAL algorithm for
    # fast convergence and supports unequal-cost load balancing.
```

Each provider has slightly different API structures, but the pattern is the same:
1. Initialize client with API key
2. Create a message/request with your prompt
3. Extract the text from the response

---

## Check Your Understanding: API Basics

Before moving to rate limits, verify you understand the fundamentals:

1. **What are the three levels of API key security?**
   <details>
   <summary>Show answer</summary>
   Level 1: Hardcoded (dangerous), Level 2: Environment variables (acceptable), Level 3: Secrets manager (production standard)
   </details>

2. **Why is hardcoding API keys dangerous?**
   <details>
   <summary>Show answer</summary>
   Gets committed to version control, bots scan GitHub for keys within minutes, leads to unauthorized usage and unexpected bills
   </details>

3. **What networking concept is analogous to API keys?**
   <details>
   <summary>Show answer</summary>
   IPsec Pre-Shared Keys (PSK) - both are secrets used for authentication, both need rotation, both should never be in configs
   </details>

4. **What's the pattern for all three providers (Claude, GPT, Gemini)?**
   <details>
   <summary>Show answer</summary>
   1. Initialize client with API key, 2. Create message/request with prompt, 3. Extract text from response
   </details>

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

### Error Handling Decision Tree

Here's the complete logic flow for handling API errors:

```
┌─────────────────┐
│  Make API Call  │
└────────┬────────┘
         │
         ▼
   ┌──────────┐
   │ Success? │
   └────┬─────┘
        │
    ┌───┴───┐
    │       │
   YES     NO
    │       │
    │       ▼
    │  ┌───────────────┐
    │  │ What error?   │
    │  └───┬───────────┘
    │      │
    │  ┌───┴────────────────────────┐
    │  │                            │
    │  ▼                            ▼
    │ 429                      400/401/403
    │ Rate Limit               Invalid Request
    │  │                            │
    │  ▼                            ▼
    │ Retry?         ┌───────────────────────┐
    │ < max          │ DON'T RETRY           │
    │  │             │ (Permanent error)     │
    │  ▼             └───────────────────────┘
    │ Wait                         │
    │ 2^attempt                    │
    │ seconds                      │
    │  │                           │
    │  └──────┐    ┌───────────────┘
    │         │    │
    │         ▼    ▼
    │  ┌──────────────┐
    │  │ Return None  │
    │  │ (Failed)     │
    │  └──────────────┘
    │
    ▼
┌────────────────┐
│ Return Result  │
└────────────────┘
```

**Key Decision Points:**
1. **Success**: Return immediately
2. **429/500/502/503**: Transient errors → Retry with backoff
3. **400/401/403/404**: Permanent errors → Don't retry, return error
4. **Network timeout**: Transient → Retry with backoff

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
                model="claude-sonnet-4.5",
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
    
    # Pricing per million tokens (January 2026, update as needed)
    PRICING = {
        "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
        "claude-haiku-4.5": {"input": 1.00, "output": 5.00},
        "claude-opus-4.5": {"input": 5.00, "output": 25.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
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
        model: str = "claude-sonnet-4.5",
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
    print("Testing ResilientAPIClient")
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
        print(f"\nPrompt: {prompt}")
        result = client.call(prompt, max_tokens=100)

        if result:
            print(f"Response: {result['text']}")
            print(f"  Tokens: {result['input_tokens']} → {result['output_tokens']}")
            print(f"  Cost: ${result['cost_usd']:.4f} | Latency: {result['latency_seconds']:.1f}s")
        else:
            print("[FAILED] Request failed")

    # Print session summary
    print("\n" + "=" * 60)
    print("Session Metrics")
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
                print(f"[WAITING] Rate limit: waiting {wait_time:.1f}s...")
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

## Check Your Understanding: Error Handling

Test your grasp of resilient API patterns:

1. **What's the difference between transient and permanent errors?**
   <details>
   <summary>Show answer</summary>
   Transient errors (429, 500, 503, timeouts) are temporary and should be retried. Permanent errors (400, 401, 403, 404) indicate a problem with the request itself and retrying won't help.
   </details>

2. **Why use exponential backoff instead of fixed delays?**
   <details>
   <summary>Show answer</summary>
   Exponential backoff (1s, 2s, 4s, 8s...) gives the system progressively more time to recover while reducing load. Fixed delays might be too short (keep failing) or too long (waste time). Similar to TCP congestion control.
   </details>

3. **Should you retry a 401 Unauthorized error?**
   <details>
   <summary>Show answer</summary>
   No. 401 means invalid API key. Retrying with the same invalid key will never succeed. Fix the API key configuration instead.
   </details>

4. **What's the maximum delay you should use for backoff?**
   <details>
   <summary>Show answer</summary>
   60 seconds is a common maximum. Without a cap, delays could grow to minutes (64s, 128s, 256s...), which makes the system feel broken to users.
   </details>

5. **Why track metrics like cost and latency?**
   <details>
   <summary>Show answer</summary>
   Cost tracking prevents bill surprises and enables budget enforcement. Latency tracking helps identify performance issues and informs model selection (fast vs slow models).
   </details>

---

## Building ResilientAPIClient Progressively

The `ResilientAPIClient` class shown earlier is production-ready, but it's complex. Let's see how to build it step-by-step so you understand each piece.

### Version 1: Basic Retry (30 lines)

Start simple—just retry on rate limits:

```python
class SimpleRetryClient:
    """Basic client with fixed-delay retry."""

    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)

    def call(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-haiku-4.5",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(2)  # Fixed delay
                else:
                    return None
        return None
```

**What it does**: Retries rate limit errors with fixed 2-second delay.

**What's missing**: Exponential backoff, other errors, metrics.

---

### Version 2: Add Exponential Backoff (40 lines)

Improve the retry logic:

```python
class ExponentialBackoffClient:
    """Client with exponential backoff retry."""

    def __init__(self, api_key, max_retries=3, initial_delay=1.0):
        self.client = Anthropic(api_key=api_key)
        self.max_retries = max_retries
        self.initial_delay = initial_delay

    def call(self, prompt):
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-haiku-4.5",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            except RateLimitError:
                if attempt < self.max_retries - 1:
                    # Exponential: 1s, 2s, 4s, 8s...
                    delay = self.initial_delay * (2 ** attempt)
                    delay = min(delay, 60)  # Cap at 60s
                    print(f"Rate limited, waiting {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    return None

            except APIConnectionError:
                # Network error - also retry
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay * (2 ** attempt)
                    time.sleep(min(delay, 60))
                else:
                    return None

        return None
```

**What it adds**: Exponential backoff (faster recovery), connection error handling.

**What's missing**: Metrics, cost tracking, different error types.

---

### Version 3: Add Metrics Tracking (80 lines)

Track usage and costs:

```python
from dataclasses import dataclass

@dataclass
class Metrics:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0

class MetricsTrackingClient:
    """Client with metrics tracking."""

    PRICING = {
        "claude-haiku-4.5": {"input": 1.00, "output": 5.00},
        "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    }

    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        self.metrics = Metrics()

    def call(self, prompt, model="claude-haiku-4.5"):
        self.metrics.requests += 1

        try:
            # (Retry logic from Version 2 here)
            response = self.client.messages.create(...)

            # Track metrics
            self.metrics.successes += 1
            self.metrics.tokens_in += response.usage.input_tokens
            self.metrics.tokens_out += response.usage.output_tokens

            # Calculate cost
            pricing = self.PRICING[model]
            cost = (response.usage.input_tokens / 1_000_000) * pricing["input"]
            cost += (response.usage.output_tokens / 1_000_000) * pricing["output"]
            self.metrics.cost += cost

            return response.content[0].text

        except Exception as e:
            self.metrics.failures += 1
            return None

    def get_metrics(self):
        """Return summary of metrics."""
        return {
            "total": self.metrics.requests,
            "successes": self.metrics.successes,
            "success_rate": f"{(self.metrics.successes/max(self.metrics.requests, 1))*100:.1f}%",
            "total_cost": f"${self.metrics.cost:.4f}",
            "total_tokens": self.metrics.tokens_in + self.metrics.tokens_out
        }
```

**What it adds**: Request counting, token tracking, cost calculation.

**What's missing**: Better error classification, latency tracking, logging.

---

### Version 4: Complete Production Client (220 lines)

Add everything else—proper logging, all error types, latency tracking, and comprehensive configuration:

```python
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResilientAPIClient:
    """Production-ready API client. This is the final version."""

    # All pricing and configuration
    PRICING = {
        "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
        "claude-haiku-4.5": {"input": 1.00, "output": 5.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    }

    def __init__(self, api_key, max_retries=3, initial_delay=1.0, timeout=120):
        self.client = Anthropic(api_key=api_key, timeout=timeout)
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.metrics = Metrics()

    def call(self, prompt, model="claude-sonnet-4.5", **kwargs):
        """Make resilient API call with full error handling."""
        self.metrics.requests += 1

        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self.client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                latency = time.time() - start

                # Update all metrics
                self._update_metrics_success(response, model, latency)

                return {
                    "text": response.content[0].text,
                    "cost_usd": self._calculate_cost(response, model),
                    "latency_seconds": latency,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }

            except RateLimitError:
                if not self._handle_retry(attempt, "Rate limit"):
                    self.metrics.failures += 1
                    return None

            except APIConnectionError:
                if not self._handle_retry(attempt, "Connection error"):
                    self.metrics.failures += 1
                    return None

            except APIError as e:
                # Check if server error (5xx) - retry
                if hasattr(e, 'status_code') and 500 <= e.status_code < 600:
                    if not self._handle_retry(attempt, f"Server error {e.status_code}"):
                        self.metrics.failures += 1
                        return None
                else:
                    # Permanent error (4xx) - don't retry
                    logger.error(f"Permanent error: {e}")
                    self.metrics.failures += 1
                    return None

        return None

    def _handle_retry(self, attempt, reason):
        """Handle retry logic with exponential backoff."""
        if attempt >= self.max_retries - 1:
            logger.error(f"{reason} - max retries exceeded")
            return False

        delay = min(self.initial_delay * (2 ** attempt), 60)
        logger.warning(f"{reason} - retrying in {delay:.1f}s")
        time.sleep(delay)
        return True

    def _update_metrics_success(self, response, model, latency):
        """Update metrics after successful call."""
        self.metrics.successes += 1
        self.metrics.tokens_in += response.usage.input_tokens
        self.metrics.tokens_out += response.usage.output_tokens
        self.metrics.cost += self._calculate_cost(response, model)
        # Add latency tracking if desired

    def _calculate_cost(self, response, model):
        """Calculate cost for this API call."""
        pricing = self.PRICING.get(model, {"input": 3.0, "output": 15.0})
        return (response.usage.input_tokens / 1_000_000) * pricing["input"] + \
               (response.usage.output_tokens / 1_000_000) * pricing["output"]

    def get_metrics(self):
        """Get usage summary."""
        return self.metrics.get_summary()  # Implementation in Metrics class
```

**What it adds**: Comprehensive error handling, logging, proper abstractions, configuration, full metrics.

**This is production-ready.** All four error categories handled, proper retry logic, cost tracking, logging.

---

### The Progression

| Version | Lines | What It Does | Use Case |
|---------|-------|--------------|----------|
| **V1: Simple Retry** | 30 | Fixed delay, rate limits only | Learning/demos |
| **V2: Exponential Backoff** | 40 | Better retry, connection errors | Small scripts |
| **V3: Metrics** | 80 | Cost tracking | Cost-conscious tools |
| **V4: Production** | 220 | Everything | Production systems |

**Lab exercises guide you through building V1 → V2 → V3 → V4**. By the end, you'll have built every piece yourself and understand the full client.

---

## Lab Exercises

### Lab 0: Verify Setup and Build Simple Retry (30 minutes)

Before diving into complex error handling, verify your environment works and build a basic retry handler from scratch.

**Part A: Environment Verification (15 minutes)**

Create a test script to verify your API setup:

```python
#!/usr/bin/env python3
"""Lab 0 Part A: Verify your API setup works"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("API Setup Verification")
print("=" * 60)

# Test 1: API key is set
print("\n[Test 1] Checking API key...")
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("[FAILED] ANTHROPIC_API_KEY not set")
    print("Fix: Set it in .env file or environment")
    exit(1)
print("[OK] API key found")

# Test 2: Can create client
print("\n[Test 2] Creating API client...")
try:
    client = Anthropic(api_key=api_key)
    print("[OK] Client created successfully")
except Exception as e:
    print(f"[FAILED] Client creation failed: {e}")
    exit(1)

# Test 3: Can make a simple call
print("\n[Test 3] Making test API call...")
try:
    response = client.messages.create(
        model="claude-sonnet-4.5",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'API test successful'"}]
    )
    print(f"[OK] API call successful")
    print(f"Response: {response.content[0].text}")
    print(f"Tokens used: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
except Exception as e:
    print(f"[FAILED] API call failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("All tests passed! Environment ready.")
print("=" * 60)
```

**Part B: Build Simple Retry Handler (15 minutes)**

Now build a basic retry handler from scratch. Start simple—just fixed delays:

```python
#!/usr/bin/env python3
"""Lab 0 Part B: Build your first retry handler"""

import os
import time
from anthropic import Anthropic, RateLimitError
from dotenv import load_dotenv

load_dotenv()

def call_with_simple_retry(prompt: str, max_retries: int = 3) -> str:
    """
    Make API call with simple retry logic.
    Uses FIXED 2-second delay between retries.

    Args:
        prompt: What to ask the model
        max_retries: How many times to retry

    Returns:
        Model response text, or None if all retries fail
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")

            response = client.messages.create(
                model="claude-haiku-4.5",  # Fast model
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )

            print("[OK] Request succeeded")
            return response.content[0].text

        except RateLimitError:
            print(f"[RATE LIMITED] Hit rate limit on attempt {attempt + 1}")

            if attempt < max_retries - 1:
                print("Waiting 2 seconds before retry...")
                time.sleep(2)  # Fixed delay
            else:
                print("[FAILED] Max retries exceeded")
                return None

        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            return None

    return None


# Test it
if __name__ == "__main__":
    print("Testing Simple Retry Handler")
    print("=" * 60)

    result = call_with_simple_retry("What is BGP? One sentence.")

    if result:
        print(f"\nFinal result: {result}")
    else:
        print("\nNo result - all retries failed")
```

**Success Criteria:**
- [ ] Part A: All 3 verification tests pass
- [ ] Part A: You can see the API response and token count
- [ ] Part B: Retry handler runs and catches rate limit errors
- [ ] Part B: You see "Attempt 1/3", "Attempt 2/3" messages
- [ ] Part B: You understand how the for loop + try/except creates retry logic
- [ ] Can modify max_retries and see behavior change

**Expected Outcome:**
- Part A should pass all tests in under 30 seconds
- Part B will succeed on first try if you're not rate-limited
- If rate-limited, you'll see retry attempts with 2-second delays

**If You Finish Early:**
1. Modify Part B to use a 1-second delay instead of 2 seconds
2. Add a counter that tracks total retries across multiple calls
3. Make the delay configurable as a function parameter
4. Add timing to measure how long retries take

---

### Lab 1: Add Exponential Backoff (60 minutes)

Enhance your simple retry handler from Lab 0 with exponential backoff and better error classification.

**Steps:**
1. Start with your Lab 0 Part B code
2. Change the fixed 2-second delay to exponential: `delay = 2 ** attempt`
3. Add a maximum delay cap (60 seconds)
4. Add handling for different error types (connection errors, server errors)
5. Test with multiple rapid calls to trigger rate limits

**Starter Code Enhancement:**
```python
def call_with_exponential_backoff(prompt: str, max_retries: int = 3) -> str:
    """Enhanced retry with exponential backoff."""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    for attempt in range(max_retries):
        try:
            response = client.messages.create(...)
            return response.content[0].text

        except RateLimitError:
            # Calculate exponential delay: 1s, 2s, 4s, 8s...
            delay = min(2 ** attempt, 60)  # Cap at 60 seconds
            print(f"Rate limited. Waiting {delay}s before retry {attempt + 2}/{max_retries}...")

            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return None

        except APIConnectionError as e:
            # TODO: Add connection error handling
            pass
```

**Success Criteria:**
- [ ] Converted fixed delay to exponential (1s → 2s → 4s → 8s)
- [ ] Added max delay cap (60 seconds)
- [ ] Handles at least 3 error types (RateLimitError, APIConnectionError, generic APIError)
- [ ] Distinguishes transient errors (retry) from permanent errors (don't retry)
- [ ] Tested with rapid calls and observed exponential backoff
- [ ] Can explain why exponential is better than fixed delay

**Expected Outcome:**
If you trigger rate limits, you should see delays increasing: "Waiting 1s", "Waiting 2s", "Waiting 4s". This reduces server load while recovering from rate limits.

**If You Finish Early:**
- Add jitter (random variation) to prevent thundering herd
- Log each retry attempt to a file
- Add a circuit breaker (after N failures, stop trying for X seconds)

---

### Lab 2: Add Usage Metrics Tracking (90 minutes)

Build metrics tracking into your retry handler so you can monitor costs and performance.

**Steps:**
1. Create a `Metrics` dataclass to track: requests, successes, failures, tokens, cost, latency
2. Modify your retry handler to update metrics after each call
3. Add a `get_summary()` function that returns formatted metrics
4. Test with 10 API calls and print the summary

**Starter Code:**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CallMetrics:
    """Track API call statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_seconds: float = 0.0

# Global metrics instance
metrics = CallMetrics()

def call_with_metrics(prompt: str) -> Optional[str]:
    """Make API call and track metrics."""
    import time

    metrics.total_requests += 1
    start = time.time()

    try:
        response = client.messages.create(...)
        latency = time.time() - start

        # Update metrics
        metrics.successful_requests += 1
        metrics.total_input_tokens += response.usage.input_tokens
        metrics.total_output_tokens += response.usage.output_tokens

        # Calculate cost (Haiku: $1/$5 per million tokens)
        cost = (response.usage.input_tokens * 1.00 / 1_000_000 +
                response.usage.output_tokens * 5.00 / 1_000_000)
        metrics.total_cost_usd += cost
        metrics.total_latency_seconds += latency

        return response.content[0].text
    except Exception as e:
        metrics.failed_requests += 1
        return None

def print_metrics_summary():
    """Display metrics summary."""
    success_rate = (metrics.successful_requests /
                   max(metrics.total_requests, 1)) * 100
    avg_latency = (metrics.total_latency_seconds /
                  max(metrics.successful_requests, 1))

    print("\n" + "=" * 60)
    print("Session Metrics")
    print("=" * 60)
    print(f"Total requests: {metrics.total_requests}")
    print(f"Successful: {metrics.successful_requests}")
    print(f"Failed: {metrics.failed_requests}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total tokens: {metrics.total_input_tokens + metrics.total_output_tokens}")
    print(f"Total cost: ${metrics.total_cost_usd:.4f}")
    print(f"Avg latency: {avg_latency:.2f}s")
```

**Success Criteria:**
- [ ] Created dataclass with all 7 metrics fields
- [ ] Modified retry handler to update metrics on success and failure
- [ ] Correctly calculate cost using model pricing
- [ ] Track latency using time.time()
- [ ] Summary function shows all metrics formatted nicely
- [ ] Ran 10+ test calls and verified metrics are accurate
- [ ] Can explain what each metric tells you about API usage

**Expected Outcome:**
After 10 calls, you should see something like:
```
Total requests: 10
Successful: 10
Success rate: 100%
Total tokens: 1,234
Total cost: $0.0123
Avg latency: 3.45s
```

**If You Finish Early:**
- Add per-model cost tracking (separate Haiku vs Sonnet costs)
- Add metrics for rate limit hits
- Export metrics to JSON file
- Create a simple bar chart of cost per prompt

---

### Lab 3: Integrate into ResilientAPIClient (90 minutes)

Combine your retry logic and metrics into a reusable client class matching the chapter's `ResilientAPIClient`.

**Steps:**
1. Copy the `ResilientAPIClient` class structure from the chapter (lines 573-750)
2. Replace the retry logic with YOUR exponential backoff code from Lab 1
3. Replace the metrics with YOUR tracking code from Lab 2
4. Add proper initialization, error handling, and helper methods
5. Test with the provided test cases

**Success Criteria:**
- [ ] Class has `__init__`, `call`, `get_metrics`, and `_handle_retry` methods
- [ ] Uses your Lab 1 exponential backoff logic
- [ ] Uses your Lab 2 metrics tracking
- [ ] Handles at least 4 exception types
- [ ] Returns dict with text, cost, latency, tokens
- [ ] Test cases run successfully
- [ ] Can explain every part of the class you built

**Expected Outcome:**
You should have a 150-200 line class that handles all production concerns. Running the test cases should show successful calls with metrics.

**If You Finish Early:**
- Add support for system prompts
- Add a method to reset metrics
- Add configuration for different model defaults
- Write unit tests for the retry logic

---

### Lab 4: Multi-Provider Support (120 minutes)

Extend your `ResilientAPIClient` to support OpenAI and Gemini alongside Anthropic.

**Steps:**
1. Create an abstract base class `APIProvider` with methods: `call()`, `parse_response()`
2. Implement three concrete providers: `AnthropicProvider`, `OpenAIProvider`, `GeminiProvider`
3. Modify your client to accept a provider instance
4. Test with all three providers

**Starter Pattern:**
```python
from abc import ABC, abstractmethod

class APIProvider(ABC):
    """Abstract base for API providers."""

    @abstractmethod
    def call(self, prompt: str, **kwargs) -> dict:
        """Make API call. Returns: {text, input_tokens, output_tokens}"""
        pass

    @abstractmethod
    def get_pricing(self, model: str) -> dict:
        """Return {input, output} pricing per million tokens."""
        pass


class AnthropicProvider(APIProvider):
    def __init__(self, api_key: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)

    def call(self, prompt: str, model: str = "claude-haiku-4.5", **kwargs):
        response = self.client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return {
            "text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }

    def get_pricing(self, model: str):
        pricing_table = {
            "claude-haiku-4.5": {"input": 1.00, "output": 5.00},
            "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
        }
        return pricing_table.get(model, {"input": 3.00, "output": 15.00})


# TODO: Implement OpenAIProvider
# TODO: Implement GeminiProvider
```

**Success Criteria:**
- [ ] Created abstract base class with required methods
- [ ] Implemented AnthropicProvider (working)
- [ ] Implemented OpenAIProvider (working)
- [ ] Implemented GeminiProvider (working)
- [ ] All three providers return same dict format
- [ ] Pricing is accurate for each provider
- [ ] Client works with any provider via polymorphism
- [ ] Tested same prompt across all three providers

**Expected Outcome:**
You can instantiate your client with different providers and get consistent results:
```python
anthropic = ResilientAPIClient(provider=AnthropicProvider(api_key))
openai = ResilientAPIClient(provider=OpenAIProvider(api_key))

# Both work identically
result1 = anthropic.call("What is OSPF?")
result2 = openai.call("What is OSPF?")
```

**If You Finish Early:**
- Add streaming support
- Add provider-specific optimizations
- Create a factory pattern for provider creation
- Compare response quality across providers

---

### Lab 5: Budget Enforcement (75 minutes)

Add daily budget limits to prevent runaway API costs.

**Steps:**
1. Add budget configuration to your client: `max_daily_budget_usd`
2. Track today's spending in metrics
3. Before each call, check if budget allows it
4. At 80% budget: log warning
5. At 100% budget: reject calls with clear error

**Starter Code:**
```python
class BudgetEnforcer:
    """Enforce daily spending limits."""

    def __init__(self, max_daily_budget: float):
        self.max_budget = max_daily_budget
        self.today = datetime.now().date()
        self.daily_spend = 0.0

    def check_budget(self, estimated_cost: float) -> tuple[bool, str]:
        """
        Check if call is within budget.

        Returns:
            (allowed: bool, message: str)
        """
        # Reset if new day
        if datetime.now().date() != self.today:
            self.today = datetime.now().date()
            self.daily_spend = 0.0

        # Check current spend
        if self.daily_spend >= self.max_budget:
            return False, f"Daily budget exceeded: ${self.daily_spend:.2f}/{self.max_budget:.2f}"

        # Check if this call would exceed
        if self.daily_spend + estimated_cost > self.max_budget:
            return False, f"Call would exceed budget: ${self.daily_spend + estimated_cost:.2f}/{self.max_budget:.2f}"

        # Warn at 80%
        if (self.daily_spend + estimated_cost) / self.max_budget > 0.8:
            print(f"[WARNING] Budget 80% used: ${self.daily_spend:.2f}/{self.max_budget:.2f}")

        return True, "OK"

    def record_cost(self, cost: float):
        """Record actual cost after call."""
        self.daily_spend += cost
```

**Success Criteria:**
- [ ] Budget enforcer checks before each call
- [ ] Rejects calls when budget exceeded
- [ ] Logs warning at 80% budget usage
- [ ] Resets budget daily
- [ ] Can set different budgets for different environments
- [ ] Tested by setting low budget ($0.10) and triggering both warning and block
- [ ] Can retrieve current spending: `get_daily_spending()`

**Expected Outcome:**
```
Call 1: [OK] $0.02 spent / $0.10 budget
Call 2: [OK] $0.04 spent / $0.10 budget
...
Call 8: [WARNING] Budget 80% used: $0.08/$0.10
Call 9: [OK] $0.09 spent / $0.10 budget
Call 10: [BLOCKED] Daily budget exceeded: $0.09/$0.10
```

**If You Finish Early:**
- Add budget projections (estimated time until budget exhausted)
- Add per-user budget tracking
- Send alert email at budget thresholds
- Add budget rollover (unused budget carries to next day)

---

### Lab 6: Metrics Export (Advanced, 150 minutes)

Export your API usage metrics in Prometheus format for monitoring dashboards.

**Note:** This is an advanced lab. Requires understanding of metrics systems.

**Steps:**
1. Install prometheus_client: `pip install prometheus-client`
2. Define Prometheus metrics (Counter, Gauge, Histogram)
3. Update metrics on each API call
4. Start metrics HTTP server
5. Verify metrics at http://localhost:8000/metrics
6. (Optional) Set up Grafana dashboard

**Starter Code:**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['model', 'status']  # Labels
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request latency',
    ['model']
)

api_tokens_total = Counter(
    'api_tokens_total',
    'Total tokens used',
    ['model', 'direction']  # direction: input/output
)

api_cost_total = Counter(
    'api_cost_total',
    'Total API cost in USD',
    ['model']
)

current_rate_limit_hits = Gauge(
    'current_rate_limit_hits',
    'Current rate limit hits'
)

# Update metrics in your API call:
def call_with_prometheus(prompt, model):
    with api_request_duration_seconds.labels(model=model).time():
        try:
            response = client.messages.create(...)

            # Record success
            api_requests_total.labels(model=model, status='success').inc()
            api_tokens_total.labels(model=model, direction='input').inc(response.usage.input_tokens)
            api_tokens_total.labels(model=model, direction='output').inc(response.usage.output_tokens)

            cost = calculate_cost(response, model)
            api_cost_total.labels(model=model).inc(cost)

        except RateLimitError:
            api_requests_total.labels(model=model, status='rate_limited').inc()
            current_rate_limit_hits.inc()
            raise

# Start metrics server
start_http_server(8000)
print("Metrics available at http://localhost:8000/metrics")
```

**Success Criteria:**
- [ ] Installed prometheus_client successfully
- [ ] Defined metrics for requests, latency, tokens, cost
- [ ] Metrics update on each API call
- [ ] HTTP server running on port 8000
- [ ] Can curl http://localhost:8000/metrics and see data
- [ ] Metrics format is valid Prometheus format
- [ ] Labels (model, status) work correctly

**Expected Prometheus Output:**
```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{model="claude-haiku-4.5",status="success"} 42.0
api_requests_total{model="claude-sonnet-4.5",status="success"} 15.0

# HELP api_cost_total Total API cost in USD
# TYPE api_cost_total counter
api_cost_total{model="claude-haiku-4.5"} 0.234
```

**If You Finish Early:**
- Set up local Grafana instance
- Create dashboard with cost over time
- Add alerting rules (cost > threshold)
- Export to StatsD or InfluxDB instead

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
    model="claude-sonnet-4.5",
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
