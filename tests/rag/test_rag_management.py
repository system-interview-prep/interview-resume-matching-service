import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import create_app
from modules.rag.vector_insert import VectorStoreAdapter

class TestRAGManagement(unittest.TestCase):
    """Integration test suite for RAG Knowledge Base management endpoints."""

    @classmethod
    def setUpClass(cls):
        os.environ["VECTOR_DB_PROVIDER"] = "postgresql"
        os.environ["EMBEDDING_PROVIDER"] = "bedrock"
        os.environ["EMBEDDING_MODEL"] = "cohere.embed-multilingual-v3"
        os.environ["EMBEDDING_DIMENSION"] = "1024"
        os.environ["LLM_PROVIDER"] = "bedrock"

        # Mock the embedding adapter to return static float vectors
        cls.embedding_patcher = patch('modules.rag.embedding.build_embedding_adapter_from_env')
        cls.mock_build_emb = cls.embedding_patcher.start()
        mock_adapter = MagicMock()
        mock_adapter.embed_texts.side_effect = lambda texts: [[0.1] * 1024 for _ in texts]
        cls.mock_build_emb.return_value = mock_adapter

        # Mock the LLMService text generation to return standard QA evaluation report
        cls.llm_patcher = patch('modules.rag.llm.LLMService.generate_text')
        cls.mock_generate = cls.llm_patcher.start()
        cls.mock_generate.return_value = json.dumps({
            "score": 0.90,
            "technical_depth_score": 0.95,
            "criteria_clarity_score": 0.85,
            "findings": ["Tài liệu chất lượng tốt."],
            "suggestions": ["Không có đề xuất thêm."],
            "adjusted_quality_score": 0.90
        })

        # Seed sample documents
        cls.doc1 = {
            "document": {
                "document_id": "test_mgt_doc_01",
                "version": "1.0.0",
                "status": "approved",
                "language": "vi",
                "updated_at": "2026-07-08T10:00:00Z"
            },
            "topic": {
                "domain": "backend",
                "topic_name": "python-oop",
                "role_targets": ["backend-engineer"],
                "job_levels": ["junior"]
            },
            "difficulty": {
                "level": "basic"
            },
            "knowledge": {
                "summary": "Object Oriented Programming in Python.",
                "concepts": ["Inheritance", "Polymorphism"]
            },
            "expected_points": {
                "must_have": ["Explain class and object"]
            },
            "metadata": {
                "retrieval": {
                    "is_active": True,
                    "quality_score": 0.85
                }
            }
        }

        cls.doc2 = {
            "document": {
                "document_id": "test_mgt_doc_02",
                "version": "1.0.0",
                "status": "approved",
                "language": "en",
                "updated_at": "2026-07-08T11:00:00Z"
            },
            "topic": {
                "domain": "frontend",
                "topic_name": "react-hooks",
                "role_targets": ["frontend-engineer"],
                "job_levels": ["mid"]
            },
            "difficulty": {
                "level": "intermediate"
            },
            "knowledge": {
                "summary": "State management via hooks in React.",
                "concepts": ["useState", "useEffect"]
            },
            "expected_points": {
                "must_have": ["Explain hook rules"]
            },
            "metadata": {
                "retrieval": {
                    "is_active": True,
                    "quality_score": 0.90
                }
            }
        }

        cls.app = create_app('testing')
        cls.client = cls.app.test_client()

        # Seed them into mock DB using REST API
        cls.client.post('/api/v1/rag/upsert-document', data=json.dumps(cls.doc1), content_type='application/json')
        cls.client.post('/api/v1/rag/upsert-document', data=json.dumps(cls.doc2), content_type='application/json')

    @classmethod
    def tearDownClass(cls):
        cls.client.delete('/api/v1/rag/documents/test_mgt_doc_01')
        cls.client.delete('/api/v1/rag/documents/test_mgt_doc_02')
        cls.embedding_patcher.stop()
        cls.llm_patcher.stop()

    def test_list_documents(self):
        """Test getting all unique documents in catalog."""
        response = self.client.get('/api/v1/rag/documents')
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data.decode('utf-8'))
        self.assertTrue(res_data['success'])
        
        docs = res_data['data']
        self.assertGreaterEqual(len(docs), 2)
        
        doc_ids = [d['document_id'] for d in docs]
        self.assertIn("test_mgt_doc_01", doc_ids)
        self.assertIn("test_mgt_doc_02", doc_ids)

        # Check document properties
        d1 = next(d for d in docs if d['document_id'] == "test_mgt_doc_01")
        self.assertEqual(d1['topic'], "python-oop")
        self.assertEqual(d1['difficulty'], "basic")
        self.assertEqual(d1['quality_score'], 0.85)

    def test_get_document_details(self):
        """Test getting details and reconstructed structure of a document."""
        response = self.client.get('/api/v1/rag/documents/test_mgt_doc_01')
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data.decode('utf-8'))
        self.assertTrue(res_data['success'])
        
        doc = res_data['data']['document']
        self.assertEqual(doc['document']['document_id'], "test_mgt_doc_01")
        self.assertEqual(doc['topic']['topic_name'], "python-oop")
        self.assertEqual(doc['difficulty']['level'], "basic")
        self.assertEqual(doc['knowledge']['summary'], "Object Oriented Programming in Python.")
        self.assertIn("Inheritance", doc['knowledge']['concepts'])

        chunks = res_data['data']['chunks']
        self.assertGreater(len(chunks), 0)

    def test_toggle_document_active(self):
        """Test active status toggling for a document."""
        # 1. Turn active off
        response = self.client.post('/api/v1/rag/documents/test_mgt_doc_02/toggle', 
                                    data=json.dumps({"is_active": False}), 
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        # Verify via get endpoint
        response = self.client.get('/api/v1/rag/documents/test_mgt_doc_02')
        res_data = json.loads(response.data.decode('utf-8'))
        self.assertFalse(res_data['data']['document']['metadata']['retrieval']['is_active'])

        # 2. Turn active back on
        response = self.client.post('/api/v1/rag/documents/test_mgt_doc_02/toggle', 
                                    data=json.dumps({"is_active": True}), 
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        response = self.client.get('/api/v1/rag/documents/test_mgt_doc_02')
        res_data = json.loads(response.data.decode('utf-8'))
        self.assertTrue(res_data['data']['document']['metadata']['retrieval']['is_active'])

    def test_evaluate_document(self):
        """Test AI quality evaluation response."""
        response = self.client.post('/api/v1/rag/documents/test_mgt_doc_01/evaluate')
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data.decode('utf-8'))
        self.assertTrue(res_data['success'])
        
        eval_report = res_data['evaluation']
        self.assertIn('score', eval_report)
        self.assertIn('findings', eval_report)
        self.assertIn('suggestions', eval_report)
        self.assertEqual(eval_report['adjusted_quality_score'], 0.90)

    def test_delete_document(self):
        """Test document deletion."""
        # Seed temporary document
        temp_doc = dict(self.doc1)
        temp_doc['document'] = dict(temp_doc['document'])
        temp_doc['document']['document_id'] = 'test_mgt_doc_temp'
        
        self.client.post('/api/v1/rag/upsert-document', data=json.dumps(temp_doc), content_type='application/json')
        
        # Verify it exists
        response = self.client.get('/api/v1/rag/documents/test_mgt_doc_temp')
        self.assertEqual(response.status_code, 200)

        # Delete it
        response = self.client.delete('/api/v1/rag/documents/test_mgt_doc_temp')
        self.assertEqual(response.status_code, 200)

        # Verify it is deleted
        response = self.client.get('/api/v1/rag/documents/test_mgt_doc_temp')
        self.assertEqual(response.status_code, 404)

if __name__ == '__main__':
    unittest.main()
