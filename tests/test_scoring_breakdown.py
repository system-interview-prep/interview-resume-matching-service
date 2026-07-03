import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.algorithm_manager import AlgorithmManager
from algorithms.similarity.jaccard_similarity import JaccardSimilarityAnalyzer
from algorithms.similarity.bm25_analyzer import BM25Analyzer

class TestScoringBreakdown(unittest.TestCase):
    """Test suite for the new 3-pillar scoring breakdown and updated analyzers"""
    
    def test_jaccard_similarity_calculation(self):
        """Test that JaccardSimilarityAnalyzer calculates true set-based Jaccard similarity"""
        analyzer = JaccardSimilarityAnalyzer()
        resume = "Python React Javascript"
        job = "Python Docker AWS React"
        
        result = analyzer.process_single(resume, job)
        details = result['details']
        
        # Unique tokens in resume (excluding stopwords/short): {'python', 'react', 'javascript'}
        # Unique tokens in job: {'python', 'docker', 'aws', 'react'}
        # Intersection: {'python', 'react'} (size 2)
        # Union: {'python', 'react', 'javascript', 'docker', 'aws'} (size 5)
        # Jaccard = 2 / 5 = 0.40
        self.assertAlmostEqual(details['token_jaccard'], 0.40, places=2)
        self.assertIn('python', details['matching_skills'])
        self.assertIn('react', details['matching_skills'])
        self.assertNotIn('docker', details['matching_skills'])
        
    def test_bm25_analyzer_independent(self):
        """Test BM25Analyzer operates independently and outputs proper score range"""
        analyzer = BM25Analyzer()
        resume = "Backend software developer with experience in Python, Django, AWS, Postgres."
        job = "Looking for a Python Django backend developer."
        
        result = analyzer.process_single(resume, job)
        self.assertEqual(result['algorithm'], 'bm25')
        self.assertGreater(result['score'], 0.0)
        self.assertLessEqual(result['score'], 1.0)
        self.assertIn('raw_bm25_score', result['details'])

    def test_algorithm_manager_3_pillars_breakdown(self):
        """Test AlgorithmManager organizes scores into 3 pillars and outputs score_breakdown"""
        manager = AlgorithmManager({
            'weights': {
                'sbert': 0.50,
                'cosine': 0.30,
                'requirements': 0.20
            }
        })
        
        # Mock individual results
        individual_scores = {
            'sbert': [{'score': 0.80, 'details': {}}],
            'cosine': [{'score': 0.60, 'details': {}}],
            'requirements': [{'score': 0.70, 'details': {
                'must_have_ok': True,
                'constraints_ok': True,
                'missing_must_count': 0,
                'must_have_score': 0.90,
                'nice_to_have_score': 0.50,
                'constraints_score': 1.0,
                'required_years_experience': 3,
                'resume_years_experience': 5,
                'experience_ok': True
            }}]
        }
        
        combined = manager._combine_algorithm_results(individual_scores, total_resumes=1)
        self.assertEqual(len(combined), 1)
        
        res = combined[0]
        self.assertIn('score_breakdown', res)
        breakdown = res['score_breakdown']
        
        # Verify active states of pillars
        self.assertTrue(breakdown['semantic']['active'])
        self.assertTrue(breakdown['lexical']['active'])
        self.assertTrue(breakdown['requirements']['active'])
        
        # Verify scores match individual inputs
        self.assertAlmostEqual(breakdown['semantic']['score'], 0.80, places=2)
        self.assertAlmostEqual(breakdown['lexical']['score'], 0.60, places=2)
        self.assertAlmostEqual(breakdown['requirements']['score'], 0.70, places=2)
        
        # Verify breakdown weights (40% semantic, 30% lexical, 30% requirements)
        # Normalized active weights: 0.40 / 1.0 = 0.40, 0.30 / 1.0 = 0.30, 0.30 / 1.0 = 0.30
        self.assertAlmostEqual(breakdown['semantic']['weight'], 0.40, places=2)
        self.assertAlmostEqual(breakdown['lexical']['weight'], 0.30, places=2)
        self.assertAlmostEqual(breakdown['requirements']['weight'], 0.30, places=2)
        
        # Raw weighted score = 0.80*0.40 + 0.60*0.30 + 0.70*0.30 = 0.32 + 0.18 + 0.21 = 0.71
        self.assertAlmostEqual(res['weighted_score'], 0.71, places=2)
        self.assertAlmostEqual(res['combined_score'], 0.71, places=2)

    def test_must_have_penalty_application(self):
        """Test soft must-have penalty applies correctly in AlgorithmManager"""
        manager = AlgorithmManager({
            'weights': {'sbert': 1.0, 'requirements': 1.0},
            'must_have_penalty_per_missing': 0.05,
            'must_have_penalty_floor': 0.70
        })
        
        # 2 missing must-haves
        individual_scores = {
            'sbert': [{'score': 0.80, 'details': {}}],
            'requirements': [{'score': 0.50, 'details': {
                'must_have_ok': False,
                'missing_must_count': 2,
                'missing_must_have': ['Docker', 'AWS']
            }}]
        }
        
        combined = manager._combine_algorithm_results(individual_scores, total_resumes=1)
        res = combined[0]
        breakdown = res['score_breakdown']
        
        # Penalty multiplier: 1.0 - (2 * 0.05) = 0.90
        self.assertAlmostEqual(breakdown['penalties']['must_have_penalty_multiplier'], 0.90, places=2)
        self.assertEqual(breakdown['penalties']['missing_must_count'], 2)
        self.assertIn('Docker', breakdown['penalties']['missing_must_list'])
        
        # Raw weighted score = (0.80*0.40 + 0.50*0.30) / (0.40 + 0.30) = (0.32 + 0.15) / 0.7 = 0.6714
        # Combined score with penalty = 0.6714 * 0.90 = 0.604
        self.assertAlmostEqual(res['combined_score'], res['weighted_score'] * 0.90, places=3)

if __name__ == '__main__':
    unittest.main()
