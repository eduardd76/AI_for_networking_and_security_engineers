#!/usr/bin/env python3
"""
Caching Layer - Reduce API Costs with Response Caching

Cache AI responses to avoid redundant API calls. Typical savings: 50-70% of costs.

From: AI for Networking Engineers - Volume 1, Chapter 8
Author: Eduard Dulharu

Usage:
    from caching_layer import CachedAPIClient

    client = CachedAPIClient()
    result = client.call("Explain BGP")  # API call
    result2 = client.call("Explain BGP")  # Cached! No cost

    print(f"Cache hit rate: {client.get_cache_stats()['hit_rate']}")
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cached response entry."""
    key: str
    response: Dict[str, Any]
    timestamp: float
    hits: int = 0
    cost_saved: float = 0.0


class CachedAPIClient:
    """
    API client with response caching.

    Caches responses based on prompt hash. Configurable TTL and eviction policies.
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        ttl_seconds: int = 86400,  # 24 hours
        max_cache_size_mb: int = 100
    ):
        """
        Initialize cached API client.

        Args:
            cache_dir: Directory for cache files
            ttl_seconds: Time-to-live for cache entries (default: 24 hours)
            max_cache_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ttl_seconds = ttl_seconds
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024

        # In-memory cache
        self.cache: Dict[str, CacheEntry] = {}

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "cost_saved": 0.0
        }

        # Load existing cache
        self._load_cache()

        logger.info(f"CachedAPIClient initialized (TTL: {ttl_seconds}s, max size: {max_cache_size_mb}MB)")

    def call(
        self,
        prompt: str,
        model: str = "claude-3-5-sonnet",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Make API call with caching.

        Args:
            prompt: User prompt
            model: Model to use
            temperature: Temperature setting
            max_tokens: Max tokens in response
            force_refresh: Skip cache and force new API call

        Returns:
            Response dict (from cache or fresh API call)
        """
        self.stats["total_requests"] += 1

        # Generate cache key
        cache_key = self._generate_cache_key(prompt, model, temperature, max_tokens)

        # Check cache (unless force refresh)
        if not force_refresh:
            cached = self._get_from_cache(cache_key)
            if cached:
                self.stats["hits"] += 1
                logger.info(f"Cache HIT (key: {cache_key[:16]}...)")
                return cached

        # Cache miss - make actual API call
        self.stats["misses"] += 1
        logger.info(f"Cache MISS (key: {cache_key[:16]}...)")

        # TODO: Integrate with actual API client
        # For now, return mock response
        response = self._mock_api_call(prompt, model, temperature, max_tokens)

        # Save to cache
        if response and temperature == 0.0:  # Only cache deterministic responses
            self._save_to_cache(cache_key, response)

        return response

    def _generate_cache_key(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate unique cache key from parameters."""
        # Create deterministic hash of parameters
        key_data = f"{model}|{temperature}|{max_tokens}|{prompt}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve from cache if exists and not expired."""
        if key not in self.cache:
            return None

        entry = self.cache[key]
        age = time.time() - entry.timestamp

        # Check if expired
        if age > self.ttl_seconds:
            logger.debug(f"Cache entry expired (age: {age:.0f}s)")
            del self.cache[key]
            return None

        # Update hit count and cost saved
        entry.hits += 1
        self.stats["cost_saved"] += entry.cost_saved

        return entry.response

    def _save_to_cache(self, key: str, response: Dict[str, Any]) -> None:
        """Save response to cache."""
        # Check cache size limit
        if self._get_cache_size() > self.max_cache_size_bytes:
            self._evict_entries()

        # Create cache entry
        entry = CacheEntry(
            key=key,
            response=response,
            timestamp=time.time(),
            hits=0,
            cost_saved=response.get('cost', 0.0)  # Track cost of this response
        )

        self.cache[key] = entry
        logger.debug(f"Cached response (key: {key[:16]}...)")

        # Persist to disk
        self._persist_cache()

    def _evict_entries(self, target_percent: float = 0.8) -> None:
        """
        Evict cache entries to free space.

        Eviction policy: LRU (Least Recently Used)
        """
        if not self.cache:
            return

        # Sort by last access time (timestamp + hits as proxy)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1].timestamp, x[1].hits)
        )

        # Remove oldest 20% of entries
        num_to_remove = int(len(sorted_entries) * (1 - target_percent))
        for i in range(num_to_remove):
            key = sorted_entries[i][0]
            del self.cache[key]

        logger.info(f"Evicted {num_to_remove} cache entries")

    def _get_cache_size(self) -> int:
        """Get current cache size in bytes."""
        total_size = 0
        for entry in self.cache.values():
            # Estimate size of cached response
            response_str = json.dumps(entry.response)
            total_size += len(response_str.encode())
        return total_size

    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_file = self.cache_dir / "cache.json"
        if not cache_file.exists():
            return

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Reconstruct cache entries
            for key, entry_data in data.items():
                self.cache[key] = CacheEntry(**entry_data)

            logger.info(f"Loaded {len(self.cache)} entries from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def _persist_cache(self) -> None:
        """Persist cache to disk."""
        cache_file = self.cache_dir / "cache.json"

        try:
            # Convert cache to serializable format
            data = {
                key: asdict(entry)
                for key, entry in self.cache.items()
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to persist cache: {e}")

    def _mock_api_call(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Mock API call for testing."""
        # Simulate API call delay
        time.sleep(0.1)

        # Mock response
        return {
            "text": f"Mock response for: {prompt[:50]}...",
            "model": model,
            "input_tokens": len(prompt) // 4,
            "output_tokens": 50,
            "cost": 0.003,
            "latency": 0.1
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats["total_requests"]
        hits = self.stats["hits"]
        hit_rate = (hits / total * 100) if total > 0 else 0

        cache_size_bytes = self._get_cache_size()
        cache_size_mb = cache_size_bytes / (1024 * 1024)

        return {
            "total_requests": total,
            "cache_hits": hits,
            "cache_misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "cost_saved": f"${self.stats['cost_saved']:.4f}",
            "cache_entries": len(self.cache),
            "cache_size_mb": f"{cache_size_mb:.2f} MB",
            "ttl_seconds": self.ttl_seconds
        }

    def clear_cache(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Cache cleared")

        # Remove cache file
        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            cache_file.unlink()

    def get_cached_keys(self) -> list:
        """Get list of all cached keys."""
        return list(self.cache.keys())


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("""
    ========================================
    Caching Layer Demo
    ========================================
    Reduce costs with response caching
    ========================================
    """)

    # Initialize cached client
    client = CachedAPIClient(
        cache_dir="./demo_cache",
        ttl_seconds=3600,  # 1 hour
        max_cache_size_mb=10
    )

    # Test 1: First call (cache miss)
    print("\nTest 1: Initial Requests (Cache Misses)")
    print("-" * 60)

    prompts = [
        "Explain BGP in one sentence",
        "What is OSPF?",
        "Describe VLANs"
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        result = client.call(prompt)
        print(f"  Response: {result['text'][:50]}...")
        print(f"  Cost: ${result['cost']:.6f}")

    # Test 2: Repeat calls (cache hits)
    print("\n\nTest 2: Repeated Requests (Cache Hits)")
    print("-" * 60)

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        result = client.call(prompt)
        print(f"  Response: {result['text'][:50]}...")
        print(f"  Cost: $0.000000 (cached!)")

    # Test 3: Cache statistics
    print("\n\nTest 3: Cache Statistics")
    print("-" * 60)

    stats = client.get_cache_stats()
    print(f"\nCache Performance:")
    for key, value in stats.items():
        print(f"  {key:20s}: {value}")

    # Test 4: Force refresh
    print("\n\nTest 4: Force Refresh (Bypass Cache)")
    print("-" * 60)

    print(f"\nPrompt: {prompts[0]} (force_refresh=True)")
    result = client.call(prompts[0], force_refresh=True)
    print(f"  Response: {result['text'][:50]}...")
    print(f"  Cost: ${result['cost']:.6f} (fresh API call)")

    # Test 5: Cache keys
    print("\n\nTest 5: Cached Keys")
    print("-" * 60)

    keys = client.get_cached_keys()
    print(f"\nCached entries: {len(keys)}")
    for i, key in enumerate(keys[:3], 1):
        print(f"  {i}. {key[:32]}...")

    # Final stats
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)

    final_stats = client.get_cache_stats()
    for key, value in final_stats.items():
        print(f"{key:20s}: {value}")

    print("\n✅ Demo complete!")
    print("\n💡 Cache Benefits:")
    print("  - Typical cost savings: 50-70%")
    print("  - Instant responses for cached queries")
    print("  - Reduced API rate limit pressure")
    print("  - Works best with temperature=0 (deterministic)")
