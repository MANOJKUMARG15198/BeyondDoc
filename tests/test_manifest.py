"""Run manifest shape (no OpenSearch: a fake client)."""

import hashlib

from medical_rag.evaluation.manifest import index_state, run_manifest


class FakeIndices:
    def get_alias(self, name):
        return {"medrag-chunks-abc": {"aliases": {name: {}}}}


class FakeClient:
    indices = FakeIndices()

    def count(self, index):
        return {"count": 5815}


class DownClient:
    class indices:  # noqa: N801
        @staticmethod
        def get_alias(name):
            raise ConnectionError("down")


def test_manifest_records_testset_hash_index_and_config(tmp_path):
    testset = tmp_path / "t.jsonl"
    testset.write_bytes(b'{"id": "q1"}\n')
    m = run_manifest(testset=testset, client=FakeClient(), alias="medrag-chunks", config={"k": 8})
    assert m["testset"]["sha256"] == hashlib.sha256(b'{"id": "q1"}\n').hexdigest()
    assert m["index"] == {"alias": "medrag-chunks", "indices": ["medrag-chunks-abc"], "docs": 5815}
    assert m["config"] == {"k": 8} and "ragas" in m["packages"] and "commit" in m["git"]


def test_index_state_never_raises():
    assert index_state(DownClient(), "medrag-chunks") == {"alias": "medrag-chunks", "error": "ConnectionError"}
