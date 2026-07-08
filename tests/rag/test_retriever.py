import unittest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.rag.embedding import MockEmbeddingAdapter
from modules.rag.vector_insert import upsert_interview_document, VectorStoreAdapter
from modules.rag.retriever import Retriever

class TestRetriever(unittest.TestCase):
    """Test suite for Retriever and VectorStoreAdapter search integration"""

    @classmethod
    def setUpClass(cls):
        # Configure local mock DB file path for testing
        cls.test_db_file = "test_mock_vector_store.json"
        os.environ["MOCK_VECTOR_STORE_FILE"] = cls.test_db_file
        os.environ["VECTOR_DB_PROVIDER"] = "mock"

        # Sample documents
        cls.doc_concurrency = {
            "document": {
                "document_id": "doc_concurrency_01",
                "version": "1.0.0",
                "status": "approved",
                "language": "vi"
            },
            "topic": {
                "domain": "backend",
                "topic_name": "concurrency",
                "role_targets": ["backend-engineer"],
                "job_levels": ["mid"]
            },
            "difficulty": {
                "level": "intermediate"
            },
            "knowledge": {
                "summary": "Concurrency fundamentals",
                "concepts": ["locks", "mutex"]
            },
            "expected_points": {
                "must_have": ["Understand deadlocks"]
            },
            "metadata": {
                "retrieval": {
                    "is_active": True,
                    "quality_score": 0.90
                }
            }
        }

        cls.doc_caching = {
            "document": {
                "document_id": "doc_caching_01",
                "version": "1.0.0",
                "status": "approved",
                "language": "vi"
            },
            "topic": {
                "domain": "backend",
                "topic_name": "caching",
                "role_targets": ["backend-engineer"],
                "job_levels": ["senior"]
            },
            "difficulty": {
                "level": "hard"
            },
            "knowledge": {
                "summary": "Caching strategies",
                "concepts": ["cache-aside", "write-through"]
            },
            "expected_points": {
                "must_have": ["Eviction policies like LRU"]
            },
            "metadata": {
                "retrieval": {
                    "is_active": True,
                    "quality_score": 0.85
                }
            }
        }

        # Clear any existing test DB file
        if os.path.exists(cls.test_db_file):
            os.remove(cls.test_db_file)

        # Populate test db
        upsert_interview_document(document_obj=cls.doc_concurrency)
        upsert_interview_document(document_obj=cls.doc_caching)

    @classmethod
    def tearDownClass(cls):
        # Clean up test DB file
        if os.path.exists(cls.test_db_file):
            os.remove(cls.test_db_file)

    def setUp(self):
        self.retriever = Retriever()

    def test_retrieve_by_topic(self):
        # Retrieve concurrency topic
        results = self.retriever.retrieve(topic="concurrency", query_text="locks")
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertEqual(r["metadata"]["topic"], "concurrency")

        # Retrieve caching topic
        results_caching = self.retriever.retrieve(topic="caching", query_text="lru")
        self.assertTrue(len(results_caching) > 0)
        for r in results_caching:
            self.assertEqual(r["metadata"]["topic"], "caching")

    def test_retrieve_by_difficulty(self):
        # Retrieve intermediate difficulty (concurrency doc)
        results = self.retriever.retrieve(difficulty="intermediate", query_text="test")
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertEqual(r["metadata"]["difficulty"], "intermediate")

        # Retrieve hard difficulty (caching doc)
        results_hard = self.retriever.retrieve(difficulty="hard", query_text="test")
        self.assertTrue(len(results_hard) > 0)
        for r in results_hard:
            self.assertEqual(r["metadata"]["difficulty"], "hard")

    def test_retrieve_history_filtering(self):
        # Retrieve with no history
        results_all = self.retriever.retrieve(topic="concurrency", query_text="locks", k=5)
        self.assertTrue(len(results_all) > 0)

        # Take the first retrieved chunk ID and put it in history
        first_chunk_id = results_all[0]["id"]
        results_filtered = self.retriever.retrieve(
            topic="concurrency",
            query_text="locks",
            history=[first_chunk_id],
            k=5
        )

        # The filtered list should not contain the chunk in history
        for r in results_filtered:
            self.assertNotEqual(r["id"], first_chunk_id)

        # Filter by entire knowledge unit ID
        results_filtered_ku = self.retriever.retrieve(
            topic="concurrency",
            query_text="locks",
            history=["doc_concurrency_01"],
            k=5
        )
        self.assertEqual(len(results_filtered_ku), 0)

    def test_retrieve_k_limit(self):
        # Limit results to 1 document
        results = self.retriever.retrieve(topic="concurrency", query_text="locks", k=1)
        self.assertEqual(len(results), 1)

if __name__ == "__main__":
    unittest.main()
