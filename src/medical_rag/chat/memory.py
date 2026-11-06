"""Short-term chat memory (last turns per session) and per-user rate limiting. Redis in the app; in-memory for tests."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from typing import Any, Protocol


class ShortTermMemory(Protocol):
    def recent(self, session_id: str) -> list[dict[str, str]] | None:
        """Last turns as [{"role", "content"}], or None when not cached (caller rehydrates from history)."""
        ...

    def append(self, session_id: str, turns: list[dict[str, str]]) -> None: ...
    def replace(self, session_id: str, turns: list[dict[str, str]]) -> None: ...
    def allow(self, user_id: str) -> bool:
        """Rate limit: True if the user may ask another question this minute."""
        ...


class InMemoryMemory:
    def __init__(self, max_turns: int = 6, rate_per_min: int = 20) -> None:
        self.max_turns, self.rate = max_turns, rate_per_min
        self.turns: dict[str, list[dict[str, str]]] = {}
        self.counts: dict[tuple[str, int], int] = defaultdict(int)
        self._lock = threading.Lock()

    def recent(self, session_id: str) -> list[dict[str, str]] | None:
        return list(self.turns[session_id]) if session_id in self.turns else None

    def append(self, session_id: str, turns: list[dict[str, str]]) -> None:
        with self._lock:
            self.turns[session_id] = (self.turns.get(session_id, []) + turns)[-self.max_turns :]

    def replace(self, session_id: str, turns: list[dict[str, str]]) -> None:
        with self._lock:
            self.turns[session_id] = turns[-self.max_turns :]

    def allow(self, user_id: str) -> bool:
        with self._lock:
            key = (user_id, int(time.time() // 60))
            self.counts[key] += 1
            return self.counts[key] <= self.rate


class RedisMemory:
    def __init__(self, client: Any, max_turns: int, ttl_s: int, rate_per_min: int) -> None:
        self.r, self.max_turns, self.ttl, self.rate = client, max_turns, ttl_s, rate_per_min

    @classmethod
    def from_settings(cls, settings: Any) -> RedisMemory:
        import redis

        client = redis.Redis.from_url(settings.redis_url, socket_timeout=2, socket_connect_timeout=2)
        client.ping()
        return cls(client, settings.chat_memory_turns, settings.chat_memory_ttl_s, settings.chat_rate_limit_per_min)

    @staticmethod
    def _key(session_id: str) -> str:
        return f"medrag:mem:{session_id}"

    def recent(self, session_id: str) -> list[dict[str, str]] | None:
        key = self._key(session_id)
        if not self.r.exists(key):
            return None
        return [json.loads(x) for x in self.r.lrange(key, 0, -1)]

    def append(self, session_id: str, turns: list[dict[str, str]]) -> None:
        key = self._key(session_id)
        pipe = self.r.pipeline()
        pipe.rpush(key, *[json.dumps(t) for t in turns])
        pipe.ltrim(key, -self.max_turns, -1)
        pipe.expire(key, self.ttl)
        pipe.execute()

    def replace(self, session_id: str, turns: list[dict[str, str]]) -> None:
        key = self._key(session_id)
        pipe = self.r.pipeline()
        pipe.delete(key)
        if turns:
            pipe.rpush(key, *[json.dumps(t) for t in turns[-self.max_turns :]])
            pipe.expire(key, self.ttl)
        pipe.execute()

    def allow(self, user_id: str) -> bool:
        key = f"medrag:rl:{user_id}:{int(time.time() // 60)}"
        pipe = self.r.pipeline()
        pipe.incr(key)
        pipe.expire(key, 120)
        count, _ = pipe.execute()
        return int(count) <= self.rate
