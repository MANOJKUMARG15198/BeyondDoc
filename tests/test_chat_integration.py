"""ClickHouse history and Redis memory against the local Docker services. Run: pytest -m integration"""

import uuid

import pytest

from medical_rag.chat.memory import RedisMemory
from medical_rag.chat.store import ChatMessage, ClickHouseStore, new_id, now
from medical_rag.core.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings():
    return Settings()


@pytest.fixture(scope="module")
def store(settings):
    try:
        return ClickHouseStore.from_settings(settings)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ClickHouse not available: {type(exc).__name__}")


@pytest.fixture(scope="module")
def memory(settings):
    try:
        return RedisMemory.from_settings(settings)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Redis not available: {type(exc).__name__}")


def test_sessions_are_scoped_renamed_and_deleted(store):
    user, other = uuid.uuid4().hex, uuid.uuid4().hex
    s = store.create_session(user, "New consultation")
    assert store.get_session(other, s.session_id) is None
    s.title, s.updated_at, s.message_count = "Renamed", now(), 2
    store.update_session(s)
    got = store.get_session(user, s.session_id)
    assert (got.title, got.message_count) == ("Renamed", 2)
    assert [x.session_id for x in store.list_sessions(user)] == [s.session_id]
    s.deleted, s.updated_at = True, now()
    store.update_session(s)
    assert store.get_session(user, s.session_id) is None and store.list_sessions(user) == []


def test_messages_roundtrip_in_order_with_payload(store):
    user = uuid.uuid4().hex
    s = store.create_session(user, "x")
    store.add_message(ChatMessage(new_id(), s.session_id, user, "user", "q?", now()))
    store.add_message(ChatMessage(new_id(), s.session_id, user, "assistant", "a [S1]", now(), {"cited": ["S1"]}))
    msgs = store.list_messages(s.session_id)
    assert [(m.role, m.content) for m in msgs] == [("user", "q?"), ("assistant", "a [S1]")]
    assert msgs[1].payload == {"cited": ["S1"]}


def test_redis_memory_keeps_last_turns_and_rate_limits(memory):
    sid, uid = uuid.uuid4().hex, uuid.uuid4().hex
    assert memory.recent(sid) is None
    memory.append(sid, [{"role": "user", "content": str(i)} for i in range(memory.max_turns + 3)])
    assert [t["content"] for t in memory.recent(sid)][-1] == str(memory.max_turns + 2)
    assert len(memory.recent(sid)) == memory.max_turns
    memory.replace(sid, [])
    assert memory.recent(sid) is None
    allowed = [memory.allow(uid) for _ in range(memory.rate + 1)]
    assert allowed[-1] is False and all(allowed[:-1])
