"""
Simple caching utilities for the API.
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any


class SimpleCache:
    """
    Simple in-memory cache with TTL.
    """

    def __init__(self, ttl_seconds: int = 300):
        self.cache: dict[str, dict[str, Any]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)

    def _make_key(self, data: Any) -> str:
        """Create a cache key from data."""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()

    def get(self, data: Any) -> dict[str, Any] | None:
        """Get cached result if available and not expired."""
        key = self._make_key(data)

        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry["expires"]:
                return entry["value"]
            # Expired, remove it
            del self.cache[key]

        return None

    def set(self, data: Any, value: dict[str, Any]):
        """Cache a result."""
        key = self._make_key(data)
        self.cache[key] = {"value": value, "expires": datetime.now() + self.ttl}

    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()


# Global cache instance
visualization_cache = SimpleCache(ttl_seconds=300)  # 5 minute cache
