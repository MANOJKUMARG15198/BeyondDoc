"""Chat history storage. ClickHouse in the app; an in-memory store for tests. Both implement SessionStore.

ClickHouse layout (append-only, scales by user/session):
- chat_sessions: ReplacingMergeTree(updated_at) ORDER BY (user_id, session_id); latest row wins (rename, delete flag).
- chat_messages: MergeTree ORDER BY (session_id, created_at, message_id).
Both tables expire rows after the configured retention (TTL).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol


def now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class ChatSession:
    session_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    deleted: bool = False


@dataclass
class ChatMessage:
    message_id: str
    session_id: str
    user_id: str
    role: str  # "user" | "assistant"
    content: str
    created_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)  # citations, sources, timings, abstained, model …


class SessionStore(Protocol):
    def create_session(self, user_id: str, title: str) -> ChatSession: ...
    def get_session(self, user_id: str, session_id: str) -> ChatSession | None: ...
    def list_sessions(self, user_id: str, limit: int = 100) -> list[ChatSession]: ...
    def update_session(self, session: ChatSession) -> None: ...
    def add_message(self, message: ChatMessage) -> None: ...
    def list_messages(self, session_id: str, limit: int = 500) -> list[ChatMessage]: ...


class InMemoryStore:
    def __init__(self) -> None:
        self.sessions: dict[str, ChatSession] = {}
        self.messages: dict[str, list[ChatMessage]] = {}
        self._lock = threading.Lock()

    def create_session(self, user_id: str, title: str) -> ChatSession:
        t = now()
        s = ChatSession(new_id(), user_id, title, t, t)
        with self._lock:
            self.sessions[s.session_id] = s
        return s

    def get_session(self, user_id: str, session_id: str) -> ChatSession | None:
        s = self.sessions.get(session_id)
        return s if s and s.user_id == user_id and not s.deleted else None

    def list_sessions(self, user_id: str, limit: int = 100) -> list[ChatSession]:
        rows = [s for s in self.sessions.values() if s.user_id == user_id and not s.deleted]
        return sorted(rows, key=lambda s: s.updated_at, reverse=True)[:limit]

    def update_session(self, session: ChatSession) -> None:
        with self._lock:
            self.sessions[session.session_id] = session

    def add_message(self, message: ChatMessage) -> None:
        with self._lock:
            self.messages.setdefault(message.session_id, []).append(message)

    def list_messages(self, session_id: str, limit: int = 500) -> list[ChatMessage]:
        return sorted(self.messages.get(session_id, []), key=lambda m: m.created_at)[:limit]


class ClickHouseStore:
    def __init__(self, client: Any, database: str, retention_days: int) -> None:
        self.client, self.db, self.retention = client, database, retention_days

    @classmethod
    def from_settings(cls, settings: Any) -> ClickHouseStore:
        import clickhouse_connect

        client = clickhouse_connect.get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password.get_secret_value(),
            database=settings.clickhouse_database,
            connect_timeout=5,
            send_receive_timeout=30,
        )
        store = cls(client, settings.clickhouse_database, settings.chat_retention_days)
        store.ensure_schema()
        return store

    def ensure_schema(self) -> None:
        self.client.command(
            f"""CREATE TABLE IF NOT EXISTS {self.db}.chat_sessions (
                user_id String, session_id String, title String,
                created_at DateTime64(3, 'UTC'), updated_at DateTime64(3, 'UTC'),
                message_count UInt32, deleted UInt8
            ) ENGINE = ReplacingMergeTree(updated_at) ORDER BY (user_id, session_id)
            TTL toDateTime(updated_at) + INTERVAL {int(self.retention)} DAY"""
        )
        self.client.command(
            f"""CREATE TABLE IF NOT EXISTS {self.db}.chat_messages (
                session_id String, message_id String, user_id String,
                role LowCardinality(String), content String, payload String,
                created_at DateTime64(3, 'UTC')
            ) ENGINE = MergeTree ORDER BY (session_id, created_at, message_id)
            TTL toDateTime(created_at) + INTERVAL {int(self.retention)} DAY"""
        )

    def _insert_session(self, s: ChatSession) -> None:
        self.client.insert(
            f"{self.db}.chat_sessions",
            [[s.user_id, s.session_id, s.title, s.created_at, s.updated_at, s.message_count, int(s.deleted)]],
            column_names=["user_id", "session_id", "title", "created_at", "updated_at", "message_count", "deleted"],
        )

    def create_session(self, user_id: str, title: str) -> ChatSession:
        t = now()
        s = ChatSession(new_id(), user_id, title, t, t)
        self._insert_session(s)
        return s

    def _rows(self, where: str, params: dict[str, Any], limit: int) -> list[ChatSession]:
        result = self.client.query(
            f"""SELECT session_id, user_id, title, created_at, updated_at, message_count, deleted
                FROM {self.db}.chat_sessions FINAL WHERE {where} AND deleted = 0
                ORDER BY updated_at DESC LIMIT {int(limit)}""",
            parameters=params,
        )
        return [
            ChatSession(r[0], r[1], r[2], r[3].replace(tzinfo=UTC), r[4].replace(tzinfo=UTC), int(r[5]), bool(r[6]))
            for r in result.result_rows
        ]

    def get_session(self, user_id: str, session_id: str) -> ChatSession | None:
        rows = self._rows("user_id = {u:String} AND session_id = {s:String}", {"u": user_id, "s": session_id}, 1)
        return rows[0] if rows else None

    def list_sessions(self, user_id: str, limit: int = 100) -> list[ChatSession]:
        return self._rows("user_id = {u:String}", {"u": user_id}, limit)

    def update_session(self, session: ChatSession) -> None:
        self._insert_session(session)

    def add_message(self, m: ChatMessage) -> None:
        self.client.insert(
            f"{self.db}.chat_messages",
            [[m.session_id, m.message_id, m.user_id, m.role, m.content, json.dumps(m.payload), m.created_at]],
            column_names=["session_id", "message_id", "user_id", "role", "content", "payload", "created_at"],
        )

    def list_messages(self, session_id: str, limit: int = 500) -> list[ChatMessage]:
        result = self.client.query(
            f"""SELECT message_id, session_id, user_id, role, content, created_at, payload
                FROM {self.db}.chat_messages WHERE session_id = {{s:String}}
                ORDER BY created_at, message_id LIMIT {int(limit)}""",
            parameters={"s": session_id},
        )
        return [
            ChatMessage(r[0], r[1], r[2], r[3], r[4], r[5].replace(tzinfo=UTC), json.loads(r[6] or "{}"))
            for r in result.result_rows
        ]
