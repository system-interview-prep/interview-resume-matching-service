import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    PORT = int(os.getenv('PORT', 5001))
    
    # File Upload Settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
    MAX_FILES_PER_REQUEST = 50
    
    # Model Settings
    MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', 'models_cache')
    DEVICE = 'cuda' if os.getenv('USE_GPU', 'False').lower() == 'true' else 'cpu'
    
    # Algorithm Settings
    DEFAULT_ALGORITHMS = ['requirements', 'sbert', 'bm25', 'cosine', 'ner']
    ALGORITHM_TIMEOUT = 300  # 5 minutes
    BATCH_SIZE = 32

    # PostgreSQL + pgvector (vector database for JD embeddings)
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/matching_db')
    VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', 384))  # all-MiniLM-L6-v2 = 384

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    
class TestingConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False

config_dict = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
