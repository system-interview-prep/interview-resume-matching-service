import os
import numpy as np
import logging
from sentence_transformers import CrossEncoder
from ..base_algorithm import BaseAlgorithm

logger = logging.getLogger(__name__)

class CrossEncoderAnalyzer(BaseAlgorithm):
    """Cross-Encoder Reranker algorithm for token-level interaction"""
    
    def __init__(self, config: dict = None):
        super().__init__('cross_encoder', config)
        self.model_name = self.config.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.max_length = int(self.config.get('max_length', 512))
        
    def load_model(self):
        """Load Cross-Encoder model"""
        try:
            logger.info(f"Loading Cross-Encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name, max_length=self.max_length)
            self.is_loaded = True
            logger.info("Cross-Encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model: {e}")
            raise
            
    def process_single(self, resume_text: str, job_description: str, 
                       position: str = None, job_id: str = None, cv_id: str = None,
                       chunk_level: bool = True) -> dict:
        """Process single CV against JD using Cross-Encoder"""
        if not self.is_loaded:
            self.load_model()
            
        try:
            from utils.text_chunker import get_chunks
            
            # Cross-Encoder evaluates segments
            cv_chunks = get_chunks(resume_text, chunk_size=120, overlap=25)
            jd_chunks = get_chunks(job_description, chunk_size=120, overlap=25)
            
            if not cv_chunks or not jd_chunks:
                return {'algorithm': self.name, 'score': 0.0, 'details': {'error': 'Empty chunks'}}
                
            # Compute pairwise scores
            pairs = []
            for jd_c in jd_chunks:
                for cv_c in cv_chunks:
                    pairs.append([jd_c, cv_c])
                    
            # Predict scores in batches to avoid CPU overload
            scores = self.model.predict(pairs, batch_size=32)
            
            # Map back to matrix form
            num_cv = len(cv_chunks)
            num_jd = len(jd_chunks)
            scores_matrix = np.array(scores).reshape(num_jd, num_cv)
            
            # MaxSim: for each JD chunk, find maximum similarity in CV chunks
            max_sims = np.max(scores_matrix, axis=1)
            
            # Normalize scores from raw logits to [0, 1]
            # Logits for ms-marco-MiniLM-L-6-v2 typically range from -10 to +4
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
                
            normalized_max_sims = sigmoid(max_sims)
            final_score = float(np.mean(normalized_max_sims))
            
            return {
                'algorithm': self.name,
                'score': final_score,
                'details': {
                    'num_cv_chunks': num_cv,
                    'num_jd_chunks': num_jd,
                    'raw_max_logits': max_sims.tolist(),
                    'normalized_scores': normalized_max_sims.tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in CrossEncoder process_single: {e}", exc_info=True)
            return {'algorithm': self.name, 'score': 0.0, 'details': {'error': str(e)}}
