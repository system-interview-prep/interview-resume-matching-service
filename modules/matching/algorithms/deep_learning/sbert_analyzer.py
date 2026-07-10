from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ..base_algorithm import BaseAlgorithm
import logging

logger = logging.getLogger(__name__)

class SBERTAnalyzer(BaseAlgorithm):
    """Sentence-BERT based semantic analysis for resume ranking"""
    
    def __init__(self, config: dict = None):
        super().__init__('sbert', config)
        self.model_name = self.config.get('model_name', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
    def load_model(self):
        """Load S-BERT model"""
        try:
            logger.info(f"Loading S-BERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.is_loaded = True
            logger.info("S-BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load S-BERT model: {e}")
            raise
    
    def _get_jd_embedding(self, job_description: str, job_id: str = None) -> np.ndarray:
        """Retrieve JD embedding using pgvector database cache with checksum and version verification."""
        import hashlib
        checksum = hashlib.sha256(job_description.encode('utf-8')).hexdigest()

        vector_db = getattr(self, 'vector_db', None)
        if job_id and vector_db:
            try:
                cache = vector_db.get_jd_cache(job_id)
                if cache and cache.get('checksum') == checksum:
                    db_vector = cache.get('sbert')
                    if db_vector is not None:
                        logger.info(f"SBERT Cache Hit (DB): JD embedding retrieved from pgvector (version={cache.get('version')})")
                        return np.array(db_vector)
            except Exception as e:
                logger.warning(f"Failed to fetch sBERT vector from database cache: {e}")

        # Cache miss: Encode JD text
        logger.info("SBERT Cache Miss: Encoding JD text using model")
        embedding = self.model.encode([job_description], convert_to_tensor=False)[0]

        # Save to Database Cache
        if job_id and vector_db:
            try:
                vector_db.upsert_jd_cache(job_id, checksum, 'sbert', embedding.tolist())
            except Exception as e:
                logger.warning(f"Failed to save sBERT embedding to database cache: {e}")

        return embedding

    def process_single(self, resume_text: str, job_description: str, 
                      position: str = None, job_id: str = None) -> dict:
        """Process single resume with S-BERT (utilizing caching)"""
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Get embeddings
            job_embedding = self._get_jd_embedding(job_description, job_id)
            resume_embedding = self.model.encode([resume_text], convert_to_tensor=False)[0]
            
            # Calculate cosine similarity
            similarity_score = cosine_similarity([resume_embedding], [job_embedding])[0][0]
            normalized_score = max(0, min(1, float(similarity_score)))
            
            return {
                'algorithm': self.name,
                'score': normalized_score,
                'similarity_score': float(similarity_score),
                'details': {
                    'embedding_dimension': len(job_embedding),
                    'model_used': self.model_name,
                    'sentence_optimized': True,
                    'max_seq_length': self.model.get_max_seq_length()
                }
            }
            
        except Exception as e:
            logger.error(f"S-BERT processing failed: {e}")
            raise

    def process_batch(self, resume_texts: list, job_description: str, 
                     position: str = None, job_id: str = None) -> list:
        """Process multiple resumes in batch using optimal batch encoding and JD vector caching."""
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()
        results = []

        try:
            # 1. Retrieve JD embedding (with L1/L2 cache)
            job_embedding = self._get_jd_embedding(job_description, job_id)

            # 2. Encode all resumes in one batch
            logger.info(f"SBERT: Batch encoding {len(resume_texts)} resumes")
            resume_embeddings = self.model.encode(resume_texts, convert_to_tensor=False)

            # 3. Calculate similarity for each resume
            for i, r_emb in enumerate(resume_embeddings):
                try:
                    similarity_score = cosine_similarity([r_emb], [job_embedding])[0][0]
                    normalized_score = max(0, min(1, float(similarity_score)))

                    results.append({
                        'resume_index': i,
                        'algorithm': self.name,
                        'score': normalized_score,
                        'similarity_score': float(similarity_score),
                        'details': {
                            'embedding_dimension': len(job_embedding),
                            'model_used': self.model_name,
                            'sentence_optimized': True,
                            'max_seq_length': self.model.get_max_seq_length()
                        }
                    })
                    self._performance_metrics['total_processed'] += 1
                except Exception as e:
                    logger.error(f"Error processing resume {i} with {self.name}: {e}")
                    self._performance_metrics['errors'] += 1
                    results.append({
                        'resume_index': i,
                        'score': 0.0,
                        'error': str(e),
                        'algorithm': self.name
                    })

            processing_time = time.time() - start_time
            self._performance_metrics['total_time'] += processing_time
            self._performance_metrics['average_time'] = (
                self._performance_metrics['total_time'] / 
                max(self._performance_metrics['total_processed'], 1)
            )

            logger.info(f"{self.name} processed {len(resume_texts)} resumes in {processing_time:.2f}s (cached)")

        except Exception as e:
            logger.error(f"Batch processing failed for {self.name}: {e}")
            raise

        return results
