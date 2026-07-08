import os
import sys
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.rag.quality_layer import QualityLayer


class TestQualityLayer(unittest.TestCase):
    """Test suite for the retrieved data quality control layer."""

    def setUp(self):
        self.quality_layer = QualityLayer(similarity_weight=0.7, quality_weight=0.3)
        self.sample_chunks = [
            {
                "id": "chunk_01",
                "text": "Concurrency basics and thread safety.",
                "score": 0.85,
                "metadata": {
                    "topic": "concurrency",
                    "difficulty": "intermediate",
                    "quality_score": 0.9,
                    "chunk_type": "knowledge",
                },
            },
            {
                "id": "chunk_02",
                "text": "What is a race condition?",
                "score": 0.70,
                "metadata": {
                    "topic": "concurrency",
                    "difficulty": "intermediate",
                    "quality_score": 0.95,
                    "chunk_type": "expected_points",
                },
            },
            {
                "id": "chunk_03",
                "text": "Concurrency basics and thread safety.",  # Duplicate text of chunk_01 but higher score
                "score": 0.90,
                "metadata": {
                    "topic": "concurrency",
                    "difficulty": "intermediate",
                    "quality_score": 0.90,
                    "chunk_type": "knowledge",
                },
            },
            {
                "id": "chunk_04",
                "text": "Follow-up query about lock-free queues.",
                "score": 0.80,
                "metadata": {
                    "topic": "concurrency",
                    "difficulty": "advanced",  # Different difficulty
                    "quality_score": 0.85,
                    "chunk_type": "follow_up",
                },
            },
            {
                "id": "chunk_05",
                "text": "Design a concurrent hashmap.",
                "score": 0.75,
                "metadata": {
                    "topic": "system-design",  # Different topic
                    "difficulty": "advanced",
                    "quality_score": 0.80,
                    "chunk_type": "deliverables",
                },
            },
        ]

    def test_filter_duplicates(self):
        """Test duplicate removal based on exact text or ID."""
        # chunk_01 and chunk_03 have the exact same text content.
        # chunk_03 has a higher similarity score (0.90 vs 0.85), so it should be kept.
        filtered = self.quality_layer.filter_duplicates(self.sample_chunks)
        ids = [c["id"] for c in filtered]

        self.assertIn("chunk_03", ids)
        self.assertNotIn("chunk_01", ids)  # chunk_01 is duplicate of chunk_03 and has lower score
        self.assertEqual(len(filtered), 4)

    def test_filter_topic(self):
        """Test topic filtering (case-insensitive)."""
        filtered = self.quality_layer.filter_topic(self.sample_chunks, "CONCURRENCY")
        topics = [c["metadata"]["topic"] for c in filtered]
        self.assertTrue(all(t == "concurrency" for t in topics))
        self.assertEqual(len(filtered), 4)

    def test_filter_difficulty(self):
        """Test difficulty level filtering (case-insensitive)."""
        filtered = self.quality_layer.filter_difficulty(self.sample_chunks, "intermediate")
        difficulties = [c["metadata"]["difficulty"] for c in filtered]
        self.assertTrue(all(d == "intermediate" for d in difficulties))
        self.assertEqual(len(filtered), 3)

    def test_rank_chunks(self):
        """Test re-ranking calculations and order."""
        # For chunk_02: similarity=0.7, quality=0.95 -> combined = 0.7*0.7 + 0.95*0.3 = 0.49 + 0.285 = 0.775
        # For chunk_03: similarity=0.9, quality=0.90 -> combined = 0.9*0.7 + 0.90*0.3 = 0.63 + 0.27 = 0.90
        # For chunk_04: similarity=0.8, quality=0.85 -> combined = 0.8*0.7 + 0.85*0.3 = 0.56 + 0.255 = 0.815
        # For chunk_05: similarity=0.75, quality=0.80 -> combined = 0.75*0.7 + 0.80*0.3 = 0.525 + 0.24 = 0.765

        ranked = self.quality_layer.rank_chunks([
            self.sample_chunks[1],  # chunk_02
            self.sample_chunks[2],  # chunk_03
            self.sample_chunks[3],  # chunk_04
            self.sample_chunks[4],  # chunk_05
        ])

        # Verification of descending order
        self.assertEqual(ranked[0]["id"], "chunk_03")  # score ~ 0.90
        self.assertEqual(ranked[1]["id"], "chunk_04")  # score ~ 0.815
        self.assertEqual(ranked[2]["id"], "chunk_02")  # score ~ 0.775
        self.assertEqual(ranked[3]["id"], "chunk_05")  # score ~ 0.765

        self.assertAlmostEqual(ranked[0]["score"], 0.90, places=3)
        self.assertAlmostEqual(ranked[1]["score"], 0.815, places=3)
        self.assertAlmostEqual(ranked[2]["score"], 0.775, places=3)
        self.assertAlmostEqual(ranked[3]["score"], 0.765, places=3)

        # Check transparency fields
        self.assertIn("combined_quality_score", ranked[0]["metadata"])
        self.assertIn("raw_similarity_score", ranked[0]["metadata"])

    def test_select_components(self):
        """Test extraction of follow-up questions and deliverables."""
        follow_ups = self.quality_layer.select_follow_ups(self.sample_chunks)
        self.assertEqual(len(follow_ups), 1)
        self.assertEqual(follow_ups[0]["id"], "chunk_04")

        deliverables = self.quality_layer.select_deliverables(self.sample_chunks)
        self.assertEqual(len(deliverables), 1)
        self.assertEqual(deliverables[0]["id"], "chunk_05")

    def test_full_process(self):
        """Test the full quality control pipeline with filters, ranking, and components."""
        res = self.quality_layer.process(
            self.sample_chunks,
            topic="concurrency",
            difficulty="intermediate",
            k=2,
        )

        ranked = res["ranked_chunks"]
        follow_ups = res["follow_ups"]
        deliverables = res["deliverables"]
        meta = res["metadata"]

        # Check counts
        self.assertEqual(len(ranked), 2)  # Limited by k=2
        # After duplicate removal (5 -> 4), topic filter (4 -> 4), difficulty filter (4 -> 2)
        # Remaining: chunk_02 (intermediate) and chunk_03 (intermediate)
        ranked_ids = [c["id"] for c in ranked]
        self.assertIn("chunk_02", ranked_ids)
        self.assertIn("chunk_03", ranked_ids)

        # Since chunk_04 (advanced) and chunk_05 (system-design, advanced) are filtered out:
        self.assertEqual(len(follow_ups), 0)
        self.assertEqual(len(deliverables), 0)

        # Verify metadata counters
        self.assertEqual(meta["original_count"], 5)
        self.assertEqual(meta["duplicates_removed"], 1)  # chunk_01
        self.assertEqual(meta["topic_mismatches_removed"], 1)  # chunk_05 (system-design)
        self.assertEqual(meta["difficulty_mismatches_removed"], 1)  # chunk_04 (advanced)
        self.assertEqual(meta["final_count"], 2)
