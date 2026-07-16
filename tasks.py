import os
import logging
import hashlib
from celery import Celery
from config.settings import config_dict
from modules.matching.manager import AlgorithmManager
from modules.matching.vector_db import VectorDB

# Setup logging for Celery
logger = logging.getLogger("celery_tasks")
logger.setLevel(logging.INFO)

# Retrieve broker URL from environment
broker_url = os.getenv("CELERY_BROKER_URL", "amqp://interview:interview_password@rabbitmq:5672")

celery_app = Celery(
    "matching_tasks",
    broker=broker_url,
    backend=None  # Write-only cache, no backend needed
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
)

class ConfigWrapper:
    def __init__(self, cfg):
        self._cfg = cfg
    def get(self, key, default=None):
        if hasattr(self._cfg, key):
            return getattr(self._cfg, key)
        if isinstance(self._cfg, dict):
            return self._cfg.get(key, default)
        return default
    def __getitem__(self, key):
        if isinstance(self._cfg, dict):
            return self._cfg[key]
        return getattr(self._cfg, key)

# Initialize VectorDB and AlgorithmManager
config_name = os.getenv("FLASK_ENV", "default")
raw_config = config_dict.get(config_name, config_dict["default"])
config = ConfigWrapper(raw_config)

db_url = config.get("DATABASE_URL", os.getenv("DATABASE_URL", ""))
vec_dim = config.get("VECTOR_DIMENSION", 384)
vector_db = VectorDB(db_url, vec_dim)

algorithm_manager = AlgorithmManager(config)
if hasattr(algorithm_manager, "vector_db"):
    algorithm_manager.vector_db = vector_db

@celery_app.task(name="tasks.vectorize_jd_task")
def vectorize_jd_task(job_id: str, job_text: str):
    """Celery background task to vectorize a JD and store in PostgreSQL pgvector"""
    logger.info(f"Celery vectorize_jd_task started for job_id={job_id}")
    try:
        models_to_load = []
        for m in ['sbert', 'bert', 'distilbert']:
            if m in algorithm_manager.algorithm_registry:
                models_to_load.append(m)

        algorithm_manager.initialize_algorithms(models_to_load)
        checksum = hashlib.sha256(job_text.encode('utf-8')).hexdigest()

        # sBERT
        sbert_alg = algorithm_manager.algorithms.get('sbert')
        if sbert_alg and hasattr(sbert_alg, 'model'):
            sbert_vector = sbert_alg.model.encode([job_text])[0].tolist()
            vector_db.upsert_jd_cache(job_id, checksum, 'sbert', sbert_vector, job_text=job_text)
            logger.info(f"sBERT vector computed & stored for job_id={job_id}")

        # BERT
        bert_alg = algorithm_manager.algorithms.get('bert')
        if bert_alg and hasattr(bert_alg, '_get_embeddings'):
            bert_vector = bert_alg._get_embeddings(job_text)[0].tolist()
            vector_db.upsert_jd_cache(job_id, checksum, 'bert', bert_vector, job_text=job_text)
            logger.info(f"BERT vector computed & stored for job_id={job_id}")

        # DistilBERT
        distilbert_alg = algorithm_manager.algorithms.get('distilbert')
        if distilbert_alg and hasattr(distilbert_alg, '_embed'):
            cleaned = distilbert_alg._clean(job_text)
            distilbert_vector = distilbert_alg._embed(cleaned)[0].tolist()
            vector_db.upsert_jd_cache(job_id, checksum, 'distilbert', distilbert_vector, job_text=job_text)
            logger.info(f"DistilBERT vector computed & stored for job_id={job_id}")

        logger.info(f"Celery vectorize_jd_task completed successfully for job_id={job_id}")
        return True
    except Exception as e:
        logger.error(f"Error in vectorize_jd_task for job_id={job_id}: {e}", exc_info=True)
        raise

@celery_app.task(name="tasks.vectorize_cv_task")
def vectorize_cv_task(cv_id: str, cv_text: str):
    """Celery background task to vectorize a CV and store in PostgreSQL pgvector"""
    logger.info(f"Celery vectorize_cv_task started for cv_id={cv_id}")
    try:
        models_to_load = []
        for m in ['sbert', 'bert', 'distilbert']:
            if m in algorithm_manager.algorithm_registry:
                models_to_load.append(m)

        algorithm_manager.initialize_algorithms(models_to_load)
        checksum = hashlib.sha256(cv_text.encode('utf-8')).hexdigest()

        # sBERT
        sbert_alg = algorithm_manager.algorithms.get('sbert')
        if sbert_alg and hasattr(sbert_alg, 'model'):
            sbert_vector = sbert_alg.model.encode([cv_text])[0].tolist()
            vector_db.upsert_cv_cache(cv_id, checksum, 'sbert', sbert_vector, cv_text=cv_text)
            logger.info(f"sBERT vector computed & stored for cv_id={cv_id}")

        # BERT
        bert_alg = algorithm_manager.algorithms.get('bert')
        if bert_alg and hasattr(bert_alg, '_get_embeddings'):
            bert_vector = bert_alg._get_embeddings(cv_text)[0].tolist()
            vector_db.upsert_cv_cache(cv_id, checksum, 'bert', bert_vector, cv_text=cv_text)
            logger.info(f"BERT vector computed & stored for cv_id={cv_id}")

        # DistilBERT
        distilbert_alg = algorithm_manager.algorithms.get('distilbert')
        if distilbert_alg and hasattr(distilbert_alg, '_embed'):
            cleaned = distilbert_alg._clean(cv_text)
            distilbert_vector = distilbert_alg._embed(cleaned)[0].tolist()
            vector_db.upsert_cv_cache(cv_id, checksum, 'distilbert', distilbert_vector, cv_text=cv_text)
            logger.info(f"DistilBERT vector computed & stored for cv_id={cv_id}")

        logger.info(f"Celery vectorize_cv_task completed successfully for cv_id={cv_id}")
        return True
    except Exception as e:
        logger.error(f"Error in vectorize_cv_task for cv_id={cv_id}: {e}", exc_info=True)
        raise
