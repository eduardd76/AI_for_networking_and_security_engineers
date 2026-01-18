#!/usr/bin/env python3
"""
Production-grade API client with built-in retry, rate limiting, and monitoring.

From: AI for Networking Engineers - Volume 1, Chapter 4
Author: Eduard Dulharu

Features:
- Exponential backoff retry logic
- Rate limit handling
- Usage tracking and metrics
- Cost monitoring
- Comprehensive error handling
- Thread-safe operations

Usage:
    from resilient_api_client import ResilientAPIClient

    client = ResilientAPIClient()
    result = client.call("Explain BGP in one sentence")

    if result:
        print(result['text'])
        print(f"Cost: ${result['cost']:.6f}")
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from anthropic import Anthropic, APIError, RateLimitError, APIConnectionError
from dotenv import load_dotenv
import json
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class APIMetrics:
    """Track API usage metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limit_errors: int = 0
    connection_errors: int = 0
    total_tokens_used: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    min_latency: Optional[float] = None
    max_latency: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


class ResilientAPIClient:
    """
    Production-ready API client with retry logic, rate limiting, and monitoring.

    This client handles all the production concerns you need:
    - Automatic retries with exponential backoff
    - Rate limit detection and handling
    - Connection error recovery
    - Usage tracking and cost monitoring
    - Detailed logging for debugging
    """

    # Pricing per million tokens (as of January 2026)
    PRICING = {
        'claude-3-5-sonnet-20241022': {'input': 3.0, 'output': 15.0},
        'claude-3-5-haiku-20241022': {'input': 0.80, 'output': 4.0},
        'claude-opus-3-5-20250929': {'input': 15.0, 'output': 75.0},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = None,
        initial_retry_delay: float = None,
        timeout: int = None,
        enable_metrics: bool = None,
        metrics_file: Optional[str] = None
    ):
        """
        Initialize API client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            max_retries: Maximum retry attempts (default: 3)
            initial_retry_delay: Initial delay between retries (default: 1.0)
            timeout: Request timeout in seconds (default: 60)
            enable_metrics: Track usage metrics (default: True)
            metrics_file: File to save metrics (default: metrics/usage_metrics.jsonl)
        """
        # Get configuration from environment or use defaults
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in .env file or pass as parameter."
            )

        self.max_retries = max_retries if max_retries is not None else int(
            os.getenv('MAX_RETRIES', '3')
        )
        self.initial_retry_delay = initial_retry_delay if initial_retry_delay is not None else float(
            os.getenv('INITIAL_RETRY_DELAY', '1.0')
        )
        self.timeout = timeout if timeout is not None else int(
            os.getenv('API_TIMEOUT', '60')
        )

        # Initialize Anthropic client
        self.client = Anthropic(api_key=self.api_key, timeout=self.timeout)

        # Metrics tracking
        self.enable_metrics = enable_metrics if enable_metrics is not None else (
            os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        )
        self.metrics = APIMetrics()
        self.metrics_file = metrics_file or os.getenv(
            'METRICS_FILE',
            './metrics/usage_metrics.jsonl'
        )

        # Ensure metrics directory exists
        if self.enable_metrics:
            metrics_path = Path(self.metrics_file)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ResilientAPIClient (max_retries={self.max_retries})")

    def call(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make API call with retry logic.

        Args:
            prompt: User prompt
            model: Model identifier (default: claude-3-5-sonnet-20241022)
            max_tokens: Maximum tokens in response
            temperature: Randomness (0-1)
            system_prompt: Optional system instructions
            metadata: Optional metadata for logging

        Returns:
            Dictionary with response and metadata:
            {
                'text': str,              # Response text
                'model': str,             # Model used
                'input_tokens': int,      # Input token count
                'output_tokens': int,     # Output token count
                'latency': float,         # Response time (seconds)
                'cost': float,            # Cost in USD
                'timestamp': str          # ISO timestamp
            }
            Returns None if all retries failed.
        """
        model = model or os.getenv('DEFAULT_MODEL', 'claude-3-5-sonnet-20241022')

        # Build message
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        # Retry loop
        for attempt in range(self.max_retries + 1):
            try:
                self.metrics.total_requests += 1

                logger.info(
                    f"API call attempt {attempt + 1}/{self.max_retries + 1} "
                    f"(model={model}, temp={temperature})"
                )
                start_time = time.time()

                # Make API call
                response = self.client.messages.create(**kwargs)

                # Calculate latency
                latency = time.time() - start_time
                self.metrics.total_latency += latency

                # Update latency stats
                if self.metrics.min_latency is None or latency < self.metrics.min_latency:
                    self.metrics.min_latency = latency
                if self.metrics.max_latency is None or latency > self.metrics.max_latency:
                    self.metrics.max_latency = latency

                # Track token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                self.metrics.total_tokens_used += (input_tokens + output_tokens)
                self.metrics.total_input_tokens += input_tokens
                self.metrics.total_output_tokens += output_tokens

                # Calculate cost
                pricing = self.PRICING.get(model, self.PRICING['claude-3-5-sonnet-20241022'])
                cost = (input_tokens / 1_000_000) * pricing['input']
                cost += (output_tokens / 1_000_000) * pricing['output']
                self.metrics.total_cost += cost

                self.metrics.successful_requests += 1

                logger.info(
                    f"API call succeeded in {latency:.2f}s "
                    f"(tokens: {input_tokens} in, {output_tokens} out, cost: ${cost:.6f})"
                )

                result = {
                    "text": response.content[0].text,
                    "model": response.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency": latency,
                    "cost": cost,
                    "timestamp": datetime.utcnow().isoformat(),
                    "attempt": attempt + 1
                }

                # Log to metrics file
                if self.enable_metrics:
                    self._log_request(result, prompt, system_prompt, metadata)

                return result

            except RateLimitError as e:
                self.metrics.rate_limit_errors += 1
                delay = self._calculate_backoff(attempt)
                logger.warning(
                    f"Rate limit exceeded. Retrying in {delay:.1f}s... "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )

                if attempt < self.max_retries:
                    time.sleep(delay)
                else:
                    self.metrics.failed_requests += 1
                    logger.error("Max retries exceeded for rate limit")
                    return None

            except APIConnectionError as e:
                self.metrics.connection_errors += 1
                delay = self._calculate_backoff(attempt)
                logger.warning(
                    f"Connection error: {e}. Retrying in {delay:.1f}s... "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )

                if attempt < self.max_retries:
                    time.sleep(delay)
                else:
                    self.metrics.failed_requests += 1
                    logger.error("Max retries exceeded for connection error")
                    return None

            except APIError as e:
                if self._is_retryable(e):
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Retryable API error ({e}). Retrying in {delay:.1f}s..."
                    )

                    if attempt < self.max_retries:
                        time.sleep(delay)
                    else:
                        self.metrics.failed_requests += 1
                        logger.error("Max retries exceeded for server error")
                        return None
                else:
                    self.metrics.failed_requests += 1
                    logger.error(f"Permanent API error: {e}")
                    return None

            except Exception as e:
                self.metrics.failed_requests += 1
                logger.error(f"Unexpected error: {type(e).__name__}: {e}")
                return None

        return None

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return self.initial_retry_delay * (2 ** attempt)

    def _is_retryable(self, error: APIError) -> bool:
        """Check if error is retryable (5xx server errors)."""
        if hasattr(error, 'status_code'):
            return 500 <= error.status_code < 600
        return False

    def _log_request(
        self,
        result: Dict[str, Any],
        prompt: str,
        system_prompt: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Log request details to metrics file."""
        try:
            log_entry = {
                "timestamp": result['timestamp'],
                "model": result['model'],
                "input_tokens": result['input_tokens'],
                "output_tokens": result['output_tokens'],
                "total_tokens": result['input_tokens'] + result['output_tokens'],
                "latency": result['latency'],
                "cost": result['cost'],
                "attempt": result['attempt'],
                "success": True,
                "prompt_length": len(prompt),
                "has_system_prompt": system_prompt is not None,
                "metadata": metadata or {}
            }

            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            logger.warning(f"Failed to log request: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get usage metrics.

        Returns:
            Dictionary with current session metrics:
            - total_requests
            - successful_requests
            - failed_requests
            - success_rate (percentage)
            - rate_limit_errors
            - connection_errors
            - total_tokens (input + output)
            - total_input_tokens
            - total_output_tokens
            - total_cost (USD)
            - avg_latency (seconds)
            - min_latency (seconds)
            - max_latency (seconds)
        """
        success_rate = 0.0
        avg_latency = 0.0

        if self.metrics.total_requests > 0:
            success_rate = (
                self.metrics.successful_requests / self.metrics.total_requests * 100
            )

        if self.metrics.successful_requests > 0:
            avg_latency = (
                self.metrics.total_latency / self.metrics.successful_requests
            )

        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": f"{success_rate:.1f}%",
            "rate_limit_errors": self.metrics.rate_limit_errors,
            "connection_errors": self.metrics.connection_errors,
            "total_tokens": self.metrics.total_tokens_used,
            "total_input_tokens": self.metrics.total_input_tokens,
            "total_output_tokens": self.metrics.total_output_tokens,
            "total_cost": f"${self.metrics.total_cost:.4f}",
            "avg_latency": f"{avg_latency:.2f}s",
            "min_latency": f"{self.metrics.min_latency:.2f}s" if self.metrics.min_latency else "N/A",
            "max_latency": f"{self.metrics.max_latency:.2f}s" if self.metrics.max_latency else "N/A"
        }

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self.metrics = APIMetrics()
        logger.info("Metrics reset")

    def print_metrics(self) -> None:
        """Print formatted metrics to console."""
        metrics = self.get_metrics()

        print(f"\n{'='*60}")
        print("SESSION METRICS")
        print(f"{'='*60}")

        for key, value in metrics.items():
            formatted_key = key.replace('_', ' ').title()
            print(f"{formatted_key:25s}: {value}")

        print(f"{'='*60}\n")


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Resilient API Client Demo
    ========================================
    Testing production-ready API client with:
    - Automatic retries
    - Rate limit handling
    - Usage tracking
    - Cost monitoring
    ========================================
    """)

    # Initialize client
    try:
        client = ResilientAPIClient(max_retries=3)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tip: Copy .env.example to .env and add your ANTHROPIC_API_KEY")
        exit(1)

    # Test prompts
    prompts = [
        "Explain BGP in one sentence",
        "What is OSPF? Answer in one sentence.",
        "Describe VLANs briefly in one sentence"
    ]

    print(f"Running {len(prompts)} test requests...\n")

    # Make API calls
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'-'*60}")
        print(f"Request {i}/{len(prompts)}: {prompt}")
        print(f"{'-'*60}")

        result = client.call(
            prompt,
            max_tokens=200,
            metadata={"test": True, "prompt_id": i}
        )

        if result:
            print(f"✓ Response: {result['text'][:150]}...")
            print(f"  Tokens: {result['input_tokens']} in + {result['output_tokens']} out")
            print(f"  Latency: {result['latency']:.2f}s")
            print(f"  Cost: ${result['cost']:.6f}")
        else:
            print("✗ Request failed after all retries")

    # Print session metrics
    client.print_metrics()

    # Check if metrics file was created
    if client.enable_metrics:
        metrics_path = Path(client.metrics_file)
        if metrics_path.exists():
            print(f"📊 Metrics saved to: {metrics_path.absolute()}")
            print(f"   File size: {metrics_path.stat().st_size} bytes")
        else:
            print(f"⚠️  Metrics file not created: {metrics_path}")

    print("\n✅ Demo complete!")
    print(f"\n💡 Tip: Check {client.metrics_file} for detailed request logs")
