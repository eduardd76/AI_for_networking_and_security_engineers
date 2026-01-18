#!/usr/bin/env python3
"""
Client-side rate limiter using token bucket algorithm.

From: AI for Networking Engineers - Volume 1, Chapter 4
Author: Eduard Dulharu (Ed Harmoosh)

Prevents exceeding API provider rate limits by throttling requests on the client side.

Usage:
    from rate_limiter import RateLimiter

    # Allow 50 requests per minute
    limiter = RateLimiter(max_requests=50, time_window=60.0)

    # Use as decorator
    @limiter
    def make_api_call():
        # Your API call here
        pass

    # Or call explicitly
    limiter.acquire()  # Blocks until request is allowed
    result = api_call()
"""

import time
import threading
from collections import deque
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Limits requests per time window. Blocks when limit is reached until
    enough time has passed.

    Thread-safe for concurrent usage.
    """

    def __init__(
        self,
        max_requests: int,
        time_window: float = 60.0,
        name: str = "default"
    ):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds (default: 60 = 1 minute)
            name: Name for this limiter (for logging)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.name = name
        self.requests = deque()
        self.lock = threading.Lock()

        logger.info(
            f"RateLimiter '{name}' initialized "
            f"({max_requests} requests per {time_window:.0f}s)"
        )

    def acquire(self, block: bool = True) -> bool:
        """
        Acquire permission to make a request.

        Args:
            block: If True, wait until request is allowed. If False, return immediately.

        Returns:
            True if request is allowed, False if limit reached and block=False
        """
        with self.lock:
            now = time.time()

            # Remove requests outside time window
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()

            # Check if at limit
            if len(self.requests) >= self.max_requests:
                if not block:
                    logger.warning(
                        f"Rate limit reached for '{self.name}' "
                        f"({self.max_requests}/{self.time_window:.0f}s)"
                    )
                    return False

                # Calculate wait time
                oldest_request = self.requests[0]
                sleep_time = self.time_window - (now - oldest_request)

                if sleep_time > 0:
                    logger.info(
                        f"Rate limit reached for '{self.name}'. "
                        f"Sleeping {sleep_time:.1f}s..."
                    )

        # Sleep outside the lock to allow other threads
        if len(self.requests) >= self.max_requests and block:
            time.sleep(sleep_time + 0.1)  # Add small buffer
            return self.acquire(block=True)  # Try again recursively

        with self.lock:
            # Record this request
            self.requests.append(time.time())

        return True

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to rate-limit a function.

        Usage:
            @rate_limiter
            def my_function():
                pass
        """
        def wrapper(*args, **kwargs) -> Any:
            self.acquire()
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dict with current_requests, max_requests, time_window
        """
        with self.lock:
            now = time.time()

            # Clean old requests
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()

            current_requests = len(self.requests)
            utilization = (current_requests / self.max_requests) * 100

            oldest_request_age = None
            if self.requests:
                oldest_request_age = now - self.requests[0]

            return {
                "name": self.name,
                "current_requests": current_requests,
                "max_requests": self.max_requests,
                "time_window": f"{self.time_window:.0f}s",
                "utilization": f"{utilization:.1f}%",
                "oldest_request_age": f"{oldest_request_age:.1f}s" if oldest_request_age else "N/A"
            }

    def reset(self) -> None:
        """Clear all tracked requests."""
        with self.lock:
            self.requests.clear()
            logger.info(f"RateLimiter '{self.name}' reset")


class TokenBucketRateLimiter:
    """
    Advanced rate limiter that also tracks tokens (not just requests).

    Useful for APIs that have both request limits (RPM) and token limits (TPM).
    """

    def __init__(
        self,
        max_requests_per_minute: int,
        max_tokens_per_minute: int,
        name: str = "token-bucket"
    ):
        """
        Initialize token bucket rate limiter.

        Args:
            max_requests_per_minute: Maximum requests per minute
            max_tokens_per_minute: Maximum tokens per minute
            name: Name for this limiter
        """
        self.request_limiter = RateLimiter(
            max_requests=max_requests_per_minute,
            time_window=60.0,
            name=f"{name}-requests"
        )

        self.max_tokens_per_minute = max_tokens_per_minute
        self.name = name
        self.tokens = deque()  # (timestamp, token_count) tuples
        self.lock = threading.Lock()

        logger.info(
            f"TokenBucketRateLimiter '{name}' initialized "
            f"({max_requests_per_minute} RPM, {max_tokens_per_minute} TPM)"
        )

    def acquire(self, estimated_tokens: int = 1000, block: bool = True) -> bool:
        """
        Acquire permission for request with estimated token usage.

        Args:
            estimated_tokens: Estimated tokens for this request
            block: If True, wait until request is allowed

        Returns:
            True if request is allowed
        """
        # Check request rate limit first
        if not self.request_limiter.acquire(block=block):
            return False

        with self.lock:
            now = time.time()

            # Remove tokens outside 1-minute window
            while self.tokens and self.tokens[0][0] < now - 60.0:
                self.tokens.popleft()

            # Calculate current token usage
            current_tokens = sum(token_count for _, token_count in self.tokens)

            # Check if adding these tokens would exceed limit
            if current_tokens + estimated_tokens > self.max_tokens_per_minute:
                if not block:
                    logger.warning(
                        f"Token limit reached for '{self.name}' "
                        f"({current_tokens + estimated_tokens}/{self.max_tokens_per_minute} TPM)"
                    )
                    return False

                # Calculate wait time
                oldest_time = self.tokens[0][0]
                sleep_time = 60.0 - (now - oldest_time)

                if sleep_time > 0:
                    logger.info(
                        f"Token limit reached for '{self.name}'. "
                        f"Sleeping {sleep_time:.1f}s..."
                    )

        # Sleep outside lock
        if current_tokens + estimated_tokens > self.max_tokens_per_minute and block:
            time.sleep(sleep_time + 0.1)
            return self.acquire(estimated_tokens, block=True)

        with self.lock:
            # Record token usage
            self.tokens.append((time.time(), estimated_tokens))

        return True

    def record_actual_tokens(self, actual_tokens: int) -> None:
        """
        Update the last request's token count with actual usage.

        Call this after receiving API response to correct estimation.

        Args:
            actual_tokens: Actual tokens used by the request
        """
        with self.lock:
            if self.tokens:
                # Update most recent token count
                timestamp, estimated = self.tokens[-1]
                self.tokens[-1] = (timestamp, actual_tokens)

                logger.debug(
                    f"Updated token count: {estimated} estimated → {actual_tokens} actual"
                )

    def get_stats(self) -> dict:
        """Get current statistics."""
        request_stats = self.request_limiter.get_stats()

        with self.lock:
            now = time.time()

            # Clean old tokens
            while self.tokens and self.tokens[0][0] < now - 60.0:
                self.tokens.popleft()

            current_tokens = sum(token_count for _, token_count in self.tokens)
            token_utilization = (current_tokens / self.max_tokens_per_minute) * 100

            return {
                "name": self.name,
                "current_requests": request_stats["current_requests"],
                "max_requests": request_stats["max_requests"],
                "request_utilization": request_stats["utilization"],
                "current_tokens": current_tokens,
                "max_tokens": self.max_tokens_per_minute,
                "token_utilization": f"{token_utilization:.1f}%"
            }


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Rate Limiter Demo
    ========================================
    Testing client-side rate limiting
    ========================================
    """)

    # Test 1: Simple rate limiter
    print("\nTest 1: Simple RateLimiter (5 requests per 10 seconds)")
    print("-" * 60)

    limiter = RateLimiter(max_requests=5, time_window=10.0, name="simple-test")

    @limiter
    def make_request(i: int):
        print(f"  ✓ Request {i} at {time.time():.1f}")
        return i

    # This will rate-limit automatically
    print("Making 10 requests (should see rate limiting after 5th):\n")
    start_time = time.time()

    for i in range(1, 11):
        make_request(i)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Stats: {limiter.get_stats()}")

    # Test 2: Token bucket limiter
    print("\n\nTest 2: TokenBucketRateLimiter")
    print("-" * 60)

    token_limiter = TokenBucketRateLimiter(
        max_requests_per_minute=10,
        max_tokens_per_minute=5000,
        name="token-test"
    )

    print("Making requests with varying token usage:\n")

    for i in range(1, 6):
        # Simulate varying token usage
        estimated_tokens = i * 1000  # 1K, 2K, 3K, 4K, 5K tokens

        if token_limiter.acquire(estimated_tokens=estimated_tokens):
            print(f"  ✓ Request {i}: {estimated_tokens} tokens at {time.time():.1f}")

            # Simulate actual response (might differ from estimate)
            actual_tokens = estimated_tokens + (i * 100)
            token_limiter.record_actual_tokens(actual_tokens)
            print(f"     → Actual usage: {actual_tokens} tokens")
        else:
            print(f"  ✗ Request {i}: DENIED (rate limit)")

    print(f"\nStats: {token_limiter.get_stats()}")

    # Test 3: Non-blocking mode
    print("\n\nTest 3: Non-blocking mode")
    print("-" * 60)

    limiter2 = RateLimiter(max_requests=3, time_window=5.0, name="non-blocking-test")

    print("Making 5 requests in non-blocking mode (3 allowed, 2 denied):\n")

    for i in range(1, 6):
        if limiter2.acquire(block=False):
            print(f"  ✓ Request {i}: ALLOWED")
        else:
            print(f"  ✗ Request {i}: DENIED")

    print(f"\nStats: {limiter2.get_stats()}")

    print("\n✅ Demo complete!")
