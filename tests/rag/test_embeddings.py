import unittest
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.rag.embedding import (
    _to_text,
    _safe_get,
    _ensure_list,
    MockEmbeddingAdapter,
    read_interview_document,
    generate_embedding_payloads,
)
from modules.rag.vector_insert import (
    VectorStoreAdapter,
    insert_vectors,
    upsert_vectors,
    upsert_interview_document,
)

class TestEmbeddings(unittest.TestCase):
    """Test suite for Interview Document embeddings and vector storage"""

    def setUp(self):
        # Sample minimal Interview Document in dict format
        self.sample_doc = {
            "document": {
                "document_id": "test_doc_001",
                "version": "1.2.3",
                "status": "approved",
                "language": "en",
                "updated_at": "2026-07-08T11:00:00Z"
            },
            "topic": {
                "domain": "backend",
                "topic_name": "concurrency",
                "role_targets": ["backend-engineer"],
                "job_levels": ["mid", "senior"]
            },
            "difficulty": {
                "level": "intermediate"
            },
            "knowledge": {
                "summary": "Concepts of multithreading and concurrency",
                "concepts": ["locks", "deadlocks"]
            },
            "expected_points": {
                "must_have": ["Understand race conditions"]
            },
            "common_mistakes": {
                "critical": ["Ignoring shared state mutations"]
            },
            "follow_up": {
                "questions": [
                    {
                        "question_id": "fu_q1",
                        "prompt": "How do you avoid deadlocks?"
                    }
                ]
            },
            "metadata": {
                "retrieval": {
                    "is_active": True,
                    "quality_score": 0.95
                }
            },
            "reference_source": {
                "primary": [
                    {
                        "title": "Concurrency in Practice"
                    }
                ]
            },
            "deliverables": {
                "rag_assets": ["vectors", "metadata"]
            }
        }

    def test_safe_get(self):
        d = {"a": {"b": {"c": 42}}}
        self.assertEqual(_safe_get(d, ["a", "b", "c"]), 42)
        self.assertEqual(_safe_get(d, ["a", "x"]), None)
        self.assertEqual(_safe_get(d, ["a", "x"], "default"), "default")

    def test_ensure_list(self):
        self.assertEqual(_ensure_list(None), [])
        self.assertEqual(_ensure_list("hello"), ["hello"])
        self.assertEqual(_ensure_list([1, 2]), [1, 2])

    def test_to_text(self):
        self.assertEqual(_to_text(None), "")
        self.assertEqual(_to_text("  hello  "), "hello")
        self.assertEqual(_to_text(123), "123")
        self.assertEqual(_to_text(["a", "b"]), "a\nb")
        d = {"k1": "v1", "k2": "v2"}
        self.assertEqual(_to_text(d), "k1: v1\nk2: v2")

    def test_read_interview_document_from_obj(self):
        res = read_interview_document(document_obj=self.sample_doc)
        self.assertEqual(res["document"]["document_id"], "test_doc_001")

    def test_read_interview_document_from_json_text(self):
        raw = json.dumps(self.sample_doc)
        res = read_interview_document(raw_text=raw)
        self.assertEqual(res["document"]["document_id"], "test_doc_001")

    def test_read_interview_document_from_yaml_text(self):
        yaml_text = """
document:
  document_id: "test_doc_yaml"
  version: "2.0.0"
topic:
  topic_name: "caching"
"""
        res = read_interview_document(raw_text=yaml_text)
        self.assertEqual(res["document"]["document_id"], "test_doc_yaml")
        self.assertEqual(res["topic"]["topic_name"], "caching")

    def test_read_interview_document_invalid_args(self):
        with self.assertRaises(ValueError):
            # Provide none
            read_interview_document()
        with self.assertRaises(ValueError):
            # Provide multiple
            read_interview_document(raw_text="{}", document_obj={})

    def test_mock_embedding_adapter(self):
        adapter = MockEmbeddingAdapter(dim=128)
        self.assertEqual(adapter.dim, 128)
        vectors = adapter.embed_texts(["hello", "world"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 128)
        self.assertEqual(len(vectors[1]), 128)

        # Check normalization
        norm_sq = sum(x * x for x in vectors[0])
        self.assertAlmostEqual(norm_sq, 1.0, places=5)

    def test_generate_embedding_payloads(self):
        adapter = MockEmbeddingAdapter(dim=64)
        records = generate_embedding_payloads(document_obj=self.sample_doc, adapter=adapter)

        # Chunks are expected for: knowledge, expected_points, common_mistakes, follow_up, reference_source, deliverables
        self.assertTrue(len(records) > 0)
        
        # Verify first record fields
        first = records[0]
        self.assertIn("id", first)
        self.assertTrue(first["id"].startswith("test_doc_001_chunk_"))
        self.assertIn("vector", first)
        self.assertEqual(len(first["vector"]), 64)
        self.assertIn("text", first)
        self.assertIn("metadata", first)

        meta = first["metadata"]
        self.assertEqual(meta["knowledge_unit_id"], "test_doc_001")
        self.assertEqual(meta["version"], "1.2.3")
        self.assertEqual(meta["topic"], "concurrency")
        self.assertEqual(meta["difficulty"], "intermediate")
        self.assertEqual(meta["language"], "en")
        self.assertEqual(meta["is_active"], True)
        self.assertEqual(meta["quality_score"], 0.95)
        self.assertEqual(meta["source"], "interview_document")
        self.assertIn("chunk_type", meta)

    def test_vector_store_adapter_mock(self):
        adapter = VectorStoreAdapter()
        # Should default to mock if env not set
        self.assertEqual(adapter.provider, "mock")
        
        records = [
            {"id": "c1", "vector": [0.1, 0.2], "text": "text1", "metadata": {}}
        ]
        res_insert = adapter.insert(records)
        self.assertTrue(res_insert["success"])
        self.assertEqual(res_insert["inserted"], 1)
        self.assertEqual(res_insert["provider"], "mock")

        res_upsert = adapter.upsert(records)
        self.assertTrue(res_upsert["success"])
        self.assertEqual(res_upsert["upserted"], 1)
        self.assertEqual(res_upsert["provider"], "mock")

    def test_upsert_interview_document_mock(self):
        # Full integration in mock mode
        res = upsert_interview_document(document_obj=self.sample_doc)
        self.assertTrue(res["success"])
        self.assertEqual(res["provider"], "mock")
        self.assertEqual(res["upserted"], res["records"])
        self.assertTrue(res["records"] > 0)

if __name__ == "__main__":
    unittest.main()
