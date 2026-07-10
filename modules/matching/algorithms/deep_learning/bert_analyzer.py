import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ..base_algorithm import BaseAlgorithm
import logging

logger = logging.getLogger(__name__)

class BERTAnalyzer(BaseAlgorithm):
    """BERT-based semantic analysis for resume ranking"""
    
    def __init__(self, config: dict = None):
        super().__init__('bert', config)
        self.model_name = self.config.get('model_name', 'bert-base-uncased')
        self.max_length = self.config.get('max_length', 512)
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self):
        """Load BERT model and tokenizer"""
        try:
            logger.info(f"Loading BERT model: {self.model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise
    
    def _get_embeddings(self, text: str) -> np.ndarray:
        """Get BERT embeddings for text"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                max_length=self.max_length, 
                truncation=True, 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings
    
    def _get_jd_embedding(self, job_description: str, job_id: str = None) -> np.ndarray:
        """Retrieve JD embedding using pgvector database cache with checksum and version verification for BERT."""
        import hashlib
        checksum = hashlib.sha256(job_description.encode('utf-8')).hexdigest()

        vector_db = getattr(self, 'vector_db', None)
        if job_id and vector_db:
            try:
                cache = vector_db.get_jd_cache(job_id)
                if cache and cache.get('checksum') == checksum:
                    db_vector = cache.get('bert')
                    if db_vector is not None:
                        logger.info(f"BERT Cache Hit (DB): JD embedding retrieved from pgvector (version={cache.get('version')})")
                        return np.array([db_vector])  # shape (1, 768)
            except Exception as e:
                logger.warning(f"Failed to fetch BERT vector from database cache: {e}")

        # Cache miss: Compute BERT CLS token embedding
        logger.info("BERT Cache Miss: Encoding JD text using model")
        embedding = self._get_embeddings(job_description)  # shape (1, 768)

        # Save to Database Cache
        if job_id and vector_db:
            try:
                vector_db.upsert_jd_cache(job_id, checksum, 'bert', embedding[0].tolist())
            except Exception as e:
                logger.warning(f"Failed to save BERT embedding to database cache: {e}")

    def _get_cv_embedding(self, cv_text: str, cv_id: str = None) -> np.ndarray:
        """Retrieve CV embedding using pgvector database cache with checksum and version verification for BERT."""
        import hashlib
        checksum = hashlib.sha256(cv_text.encode('utf-8')).hexdigest()

        vector_db = getattr(self, 'vector_db', None)
        if cv_id and vector_db:
            try:
                cache = vector_db.get_cv_cache(cv_id)
                if cache and cache.get('checksum') == checksum:
                    db_vector = cache.get('bert')
                    if db_vector is not None:
                        logger.info(f"BERT Cache Hit (DB): CV embedding retrieved from pgvector (version={cache.get('version')})")
                        return np.array([db_vector])  # shape (1, 768)
            except Exception as e:
                logger.warning(f"Failed to fetch BERT CV vector from database cache: {e}")

        # Cache miss: Compute BERT CLS token embedding
        logger.info("BERT Cache Miss: Encoding CV text using model")
        embedding = self._get_embeddings(cv_text)  # shape (1, 768)

        # Save to Database Cache
        if cv_id and vector_db:
            try:
                vector_db.upsert_cv_cache(cv_id, checksum, 'bert', embedding[0].tolist())
            except Exception as e:
                logger.warning(f"Failed to save BERT CV embedding to database cache: {e}")

        return embedding

    def process_single(self, resume_text: str, job_description: str, 
                      position: str = None, job_id: str = None, cv_id: str = None) -> dict:
        """Process single resume with BERT (utilizing caching)"""
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Get embeddings
            resume_embedding = self._get_cv_embedding(resume_text, cv_id)
            job_embedding = self._get_jd_embedding(job_description, job_id)
            
            # Calculate similarity
            similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0]
            normalized_score = max(0, min(1, (similarity_score + 1) / 2))
            
            return {
                'algorithm': self.name,
                'score': float(normalized_score),
                'similarity_score': float(similarity_score),
                'details': {
                    'embedding_dimension': resume_embedding.shape[1],
                    'model_used': self.model_name,
                    'max_length': self.max_length
                }
            }
            
        except Exception as e:
            logger.error(f"BERT processing failed: {e}")
            raise

    def process_batch(self, resume_texts: list, job_description: str, 
                      position: str = None, job_id: str = None, cv_id: str = None) -> list:
        """Process multiple resumes in batch using JD embedding cache."""
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()
        results = []

        try:
            # 1. Retrieve JD embedding (with L1/L2 cache)
            job_embedding = self._get_jd_embedding(job_description, job_id)

            # 2. Process each resume
            for i, resume_text in enumerate(resume_texts):
                try:
                    if cv_id and len(resume_texts) == 1:
                        resume_embedding = self._get_cv_embedding(resume_text, cv_id)
                    else:
                        resume_embedding = self._get_embeddings(resume_text)
                    similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0]
                    normalized_score = max(0, min(1, (similarity_score + 1) / 2))

                    results.append({
                        'resume_index': i,
                        'algorithm': self.name,
                        'score': float(normalized_score),
                        'similarity_score': float(similarity_score),
                        'details': {
                            'embedding_dimension': resume_embedding.shape[1],
                            'model_used': self.model_name,
                            'max_length': self.max_length
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
