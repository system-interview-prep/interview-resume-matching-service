import json
import os
import sys
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import create_app
from modules.rag.vector_insert import upsert_interview_document


class TestAIService(unittest.TestCase):
    """Integration test suite for the RAG RAG-AI Service Flask endpoints."""

    @classmethod
    def setUpClass(cls):
        # Configure local mock DB file path for testing
        cls.test_db_file = "test_ai_service_mock_vector_store.json"
        os.environ["MOCK_VECTOR_STORE_FILE"] = cls.test_db_file
        os.environ["VECTOR_DB_PROVIDER"] = "mock"
        os.environ["LLM_PROVIDER"] = "mock"

        # Remove previous test DB if it exists
        if os.path.exists(cls.test_db_file):
            try:
                os.remove(cls.test_db_file)
            except Exception:
                pass

        # Seed sample document
        cls.doc = {
            "document": {
                "document_id": "test_ai_service_doc_01",
                "version": "1.0.0",
                "status": "approved",
                "language": "vi",
            },
            "topic": {
                "domain": "backend",
                "topic_name": "python-concurrency",
                "role_targets": ["backend-engineer"],
                "job_levels": ["mid"],
            },
            "difficulty": {
                "level": "intermediate",
            },
            "knowledge": {
                "summary": "Concurrency in Python can be achieved via threading or multiprocessing.",
            },
            "expected_points": {
                "must_have": [
                    "Explained the Global Interpreter Lock (GIL) limitations",
                    "Suggested multiprocessing as a solution for CPU-bound tasks",
                ]
            },
            "follow_up": {
                "questions": [
                    "How do you pass data or share state safely between multiple processes?"
                ]
            },
            "deliverables": {
                "action_items": [
                    "Implement a task queue manager using the multiprocessing library"
                ]
            },
            "metadata": {
                "retrieval": {
                    "is_active": True,
                    "quality_score": 0.95,
                }
            },
        }
        upsert_interview_document(document_obj=cls.doc)

    @classmethod
    def tearDownClass(cls):
        # Clean up mock file
        if os.path.exists(cls.test_db_file):
            try:
                os.remove(cls.test_db_file)
            except Exception:
                pass

    def setUp(self):
        """Set up test Flask client"""
        self.app = create_app("testing")
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up context"""
        self.app_context.pop()

    def test_retrieve_endpoint(self):
        """Test /retrieve endpoint returns filtered chunks correctly."""
        payload = {
            "topic": "python-concurrency",
            "difficulty": "intermediate",
            "k": 3,
        }

        # Test POST
        resp = self.client.post("/retrieve", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("data", data)
        self.assertIn("ranked_chunks", data["data"])
        self.assertGreater(len(data["data"]["ranked_chunks"]), 0)

        # Test GET
        resp_get = self.client.get(
            "/api/v1/rag/retrieve?topic=python-concurrency&difficulty=intermediate&k=3"
        )
        self.assertEqual(resp_get.status_code, 200)
        data_get = json.loads(resp_get.data)
        self.assertTrue(data_get["success"])
        self.assertEqual(
            len(data_get["data"]["ranked_chunks"]),
            len(data["data"]["ranked_chunks"]),
        )

    def test_interview_initial_question_endpoint(self):
        """Test /interview generates an initial question."""
        payload = {
            "topic": "python-concurrency",
            "difficulty": "intermediate",
        }
        resp = self.client.post("/interview", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("question", data)
        self.assertEqual(data["type"], "initial")
        self.assertIn("CPython Global Interpreter Lock", data["question"])

    def test_interview_follow_up_question_endpoint(self):
        """Test /interview generates a follow-up question when conversation history is present."""
        payload = {
            "topic": "python-concurrency",
            "difficulty": "intermediate",
            "last_question": "What is Python GIL?",
            "candidate_answer": "It is a lock for threads.",
            "conversation_history": [
                {"role": "interviewer", "text": "What is Python GIL?"},
                {"role": "candidate", "text": "It is a lock for threads."},
            ],
        }
        resp = self.client.post("/api/v1/rag/interview", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("question", data)
        self.assertEqual(data["type"], "follow_up")
        self.assertIn("How do you pass data or share state", data["question"])

    def test_evaluate_endpoint(self):
        """Test /evaluate performs assessment and coaching feedback generation."""
        payload = {
            "topic": "python-concurrency",
            "difficulty": "intermediate",
            "question": "What is the CPython GIL?",
            "candidate_answer": "It prevents multiple threads from running simultaneously in CPython.",
        }
        resp = self.client.post("/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)

        self.assertTrue(data["success"])
        self.assertIn("evaluation", data)
        self.assertIn("feedback", data)

        # Check evaluation parsing
        eval_data = data["evaluation"]
        self.assertIsInstance(eval_data, dict)
        self.assertIn("must_have_covered", eval_data)
        self.assertIn("evaluation_score", eval_data)
        self.assertGreater(eval_data["evaluation_score"], 0.0)

        # Check coaching feedback
        feedback = data["feedback"]
        self.assertIn("Strengths", feedback)
        self.assertIn("multiprocessing", feedback.lower())

    def test_import_csv_endpoint(self):
        """Test /import-csv endpoint uploads and imports CSV data correctly."""
        import io

        csv_content = (
            "document_id,topic_name,difficulty_level,knowledge_summary,expected_points_must_have,is_active,quality_score\n"
            "csv_doc_python_testing_01,python-testing,intermediate,Testing basics in Python,Write unit tests;Run test suite,true,0.90\n"
        )

        data = {
            'file': (io.BytesIO(csv_content.encode('utf-8-sig')), 'test_import.csv')
        }

        resp = self.client.post('/import-csv', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)

        res_data = json.loads(resp.data)
        self.assertTrue(res_data["success"])
        self.assertIn("csv_doc_python_testing_01", res_data["imported_document_ids"])
        self.assertEqual(len(res_data["failed_rows"]), 0)

    def test_upsert_document_endpoint(self):
        """Test /upsert-document endpoint upserts JSON document data correctly."""
        doc_obj = {
            "document": {
                "document_id": "test_upsert_doc_json_01",
                "version": "1.0.0",
                "status": "approved",
                "language": "vi"
            },
            "topic": {
                "domain": "backend",
                "topic_name": "python-api",
                "role_targets": ["backend-engineer"],
                "job_levels": ["mid"]
            },
            "difficulty": {
                "level": "intermediate"
            },
            "knowledge": {
                "summary": "FastAPI is faster than Flask.",
                "concepts": ["FastAPI", "Uvicorn"]
            },
            "metadata": {
                "retrieval": {
                    "is_active": True,
                    "quality_score": 0.99
                }
            }
        }
        resp = self.client.post('/upsert-document', json=doc_obj)
        self.assertEqual(resp.status_code, 200)
        res_data = json.loads(resp.data)
        self.assertTrue(res_data["success"])
        self.assertIn("upserted successfully", res_data["message"])
        self.assertGreater(res_data["records"], 0)


