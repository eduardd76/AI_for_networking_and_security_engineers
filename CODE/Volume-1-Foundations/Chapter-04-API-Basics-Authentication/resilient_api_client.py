#!/usr/bin/env python3
"""
ResilientAPIClient - Production-grade API client for network automation.

This is the main project from Chapter 4: API Basics and Authentication.

Features:
- Automatic retry with exponential backoff
- Rate limit handling
- Usage tracking and cost monitoring
- Comprehensive error handling

Usage:
    python resilient_api_client.py              # Run demo
    python resilient_api_client.py --metrics    # Show metrics only

Author: Eduard Dulharu (Ed Harmoosh)
Company: vExpertAI GmbH
"""

import os
import sys
import time
import logging
import argparse
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import Anthropic SDK
try:
    from anthropic import Anthropic, APIError, RateLimitError, APIConnectionError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Usage Metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Resilient API Client
# ---------------------------------------------------------------------------

class ResilientAPIClient:
    """
    Production-ready API client with built-in resilience.
    
    This is the foundation for all AI networking tools.
    Handles retries, rate limits, cost tracking, and errors automatically.
    """
    
    # Pricing per million tokens (January 2026)
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
            initial_delay: Starting delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
            timeout: Request timeout in seconds
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
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
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        if system:
            kwargs["system"] = system
        
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


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Client-side rate limiter to prevent hitting API limits.
    Uses token bucket algorithm.
    """
    
    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        from collections import deque
        self.max_requests = requests_per_minute
        self.window_seconds = 60.0
        self.request_times = deque()
    
    def acquire(self):
        """Acquire permission to make a request. Blocks if rate limit reached."""
        now = time.time()
        
        # Remove requests outside time window
        while self.request_times and self.request_times[0] < now - self.window_seconds:
            self.request_times.popleft()
        
        # Wait if at limit
        if len(self.request_times) >= self.max_requests:
            wait_time = self.window_seconds - (now - self.request_times[0])
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                return self.acquire()
        
        self.request_times.append(time.time())


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo():
    """Run a demo of the ResilientAPIClient."""
    print()
    print("🔧 ResilientAPIClient Demo")
    print("   Chapter 4: API Basics and Authentication")
    print("=" * 60)
    
    try:
        client = ResilientAPIClient(max_retries=3)
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\nSet your API key:")
        print("  export ANTHROPIC_API_KEY=sk-ant-api03-your-key")
        return
    except ImportError as e:
        print(f"\n❌ {e}")
        return
    
    # Test prompts
    prompts = [
        ("What is a VLAN?", 50),
        ("Explain BGP in 20 words.", 50),
        ("What does OSPF stand for?", 30),
    ]
    
    for prompt, max_tokens in prompts:
        print(f"\n📤 Prompt: {prompt}")
        result = client.call(prompt, max_tokens=max_tokens)
        
        if result:
            print(f"📥 Response: {result['text']}")
            print(f"   📊 {result['input_tokens']} → {result['output_tokens']} tokens")
            print(f"   💰 ${result['cost_usd']:.4f} | ⏱️ {result['latency_seconds']:.1f}s")
        else:
            print("❌ Request failed")
    
    # Print metrics
    print("\n" + "=" * 60)
    print("📊 Session Metrics")
    print("=" * 60)
    for key, value in client.get_metrics().items():
        print(f"   {key:20s}: {value}")
    print()


def main():
    parser = argparse.ArgumentParser(description="ResilientAPIClient - Chapter 4")
    parser.add_argument("--demo", action="store_true", help="Run demo (default)")
    args = parser.parse_args()
    
    run_demo()


if __name__ == "__main__":
    main()
