import os
import sys
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.rag.prompt_builder import PromptBuilder


class TestPromptBuilder(unittest.TestCase):
    """Test suite for RAG PromptBuilder prompt construction."""

    def setUp(self):
        self.builder = PromptBuilder()
        self.sample_context = [
            {
                "id": "chunk_01",
                "text": "Concurrency in Python can be achieved via threading or multiprocessing.",
                "metadata": {"chunk_type": "knowledge"},
            },
            {
                "id": "chunk_02",
                "text": "Must explain GIL (Global Interpreter Lock).",
                "metadata": {"chunk_type": "expected_points"},
            },
        ]

    def test_build_question_generation_prompt(self):
        """Test prompt construction for generating a technical interview question."""
        prompt = self.builder.build_question_generation_prompt(
            self.sample_context,
            topic="Python Concurrency",
            difficulty="intermediate",
            history=["What is a thread?"],
        )

        self.assertIn("Python Concurrency", prompt)
        self.assertIn("intermediate", prompt)
        self.assertIn("What is a thread?", prompt)
        self.assertIn("GIL (Global Interpreter Lock)", prompt)
        # Verify the strict instructions warning not to copy verbatim is included
        self.assertIn("Do NOT copy the Reference Knowledge Base text verbatim", prompt)

    def test_build_evaluation_prompt(self):
        """Test prompt construction for evaluating a response."""
        prompt = self.builder.build_evaluation_prompt(
            context_chunks=self.sample_context,
            question="Explain GIL in Python.",
            candidate_answer="It is a lock that prevents multiple threads from executing Python code at once.",
        )

        self.assertIn("Explain GIL in Python.", prompt)
        self.assertIn("multiple threads from executing Python code", prompt)
        self.assertIn("GIL (Global Interpreter Lock)", prompt)
        # Verify JSON schema is included in evaluation prompt
        self.assertIn('"must_have_covered"', prompt)
        self.assertIn('"evaluation_score"', prompt)
        # Verify warning
        self.assertIn("Do NOT copy the rubric or expected points verbatim", prompt)

    def test_build_follow_up_prompt(self):
        """Test prompt construction for generating a follow-up question."""
        history = [
            {"role": "interviewer", "text": "What is Python concurrency?"},
            {"role": "candidate", "text": "It lets you run code in threads."},
        ]

        prompt = self.builder.build_follow_up_prompt(
            self.sample_context,
            history=history,
            last_question="What is Python concurrency?",
            candidate_answer="It lets you run code in threads.",
        )

        self.assertIn("Interviewer: What is Python concurrency?", prompt)
        self.assertIn("Candidate: It lets you run code in threads.", prompt)
        self.assertIn("GIL (Global Interpreter Lock)", prompt)
        # Verify warning
        self.assertIn("Do NOT copy the Reference Follow-Up Topics verbatim", prompt)

    def test_build_feedback_prompt(self):
        """Test prompt construction for generating coaching feedback."""
        deliverables = [
            {
                "id": "chunk_03",
                "text": "Write a small project demonstrating asyncio task groups.",
                "metadata": {"chunk_type": "deliverables"},
            }
        ]

        prompt = self.builder.build_feedback_prompt(
            evaluation_details={"evaluation_score": 0.8},
            deliverables=deliverables,
        )

        self.assertIn('"evaluation_score": 0.8', prompt)
        self.assertIn("asyncio task groups", prompt)
        # Verify warning
        self.assertIn("Do NOT copy the Reference Deliverables", prompt)
        self.assertTrue("coach" in prompt.lower() or "coaching" in prompt.lower())
