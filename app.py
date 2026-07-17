"""CV–JD scoring service API (multi-criteria fusion). Academic/train endpoints removed."""

import ssl
try:
    import requests
    from requests.adapters import HTTPAdapter
    # NOTE: Do not disable TLS verification globally. If you need a custom CA bundle,
    # configure REQUESTS_CA_BUNDLE / SSL_CERT_FILE instead.
    from huggingface_hub import configure_http_backend
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.verify = False
        return session
    configure_http_backend(backend_factory=backend_factory)
    
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import contextvars
import asyncio
import time

import logging

logger = logging.getLogger(__name__)

_fastapi_request_var = contextvars.ContextVar("fastapi_request")

class MockMultiDict:
    def __init__(self, data):
        self.data = data or {}

    def get(self, key, default=None):
        val = self.data.get(key, default)
        if isinstance(val, list):
            return val[0] if len(val) > 0 else default
        return val

    def getlist(self, key):
        val = self.data.get(key, [])
        if not isinstance(val, list):
            return [val]
        return val

class FlaskFileMock:
    def __init__(self, fastapi_file):
        self.fastapi_file = fastapi_file
        self.filename = fastapi_file.filename

    def seek(self, offset, whence=0):
        return self.fastapi_file.file.seek(offset, whence)

    def tell(self):
        return self.fastapi_file.file.tell()

    def read(self, size=-1):
        return self.fastapi_file.file.read(size)

class FastAPIRequestWrapper:
    def __init__(self, fastapi_request, payload_json=None, form_data=None):
        self.fastapi_request = fastapi_request
        self._payload_json = payload_json
        self._form_data = form_data
        self.path = fastapi_request.url.path
        self.method = fastapi_request.method
        self.url = str(fastapi_request.url)
        self.headers = fastapi_request.headers
        self.remote_addr = fastapi_request.client.host if fastapi_request.client else "127.0.0.1"

    @property
    def is_json(self):
        content_type = self.headers.get("content-type", "")
        return "application/json" in content_type

    @property
    def content_length(self):
        cl = self.headers.get("content-length")
        return int(cl) if cl else None

    @property
    def content_type(self):
        return self.headers.get("content-type", "")

    def get_json(self, silent=True):
        return self._payload_json or {}

    @property
    def form(self):
        return MockMultiDict(self._form_data)

    @property
    def files(self):
        files_dict = {}
        if self._form_data:
            for k, v in self._form_data.items():
                if hasattr(v, 'filename') or (isinstance(v, list) and any(hasattr(item, 'filename') for item in v)):
                    files_dict[k] = v
        return MockMultiDict(files_dict)

    @property
    def args(self):
        return MockMultiDict({k: v for k, v in self.fastapi_request.query_params.items()})

class FlaskRequestProxy:
    def __getattr__(self, name):
        req = _fastapi_request_var.get()
        return getattr(req, name)

    def get_json(self, silent=True):
        req = _fastapi_request_var.get()
        return req.get_json(silent=silent)

    @property
    def is_json(self):
        req = _fastapi_request_var.get()
        return req.is_json

    @property
    def content_length(self):
        req = _fastapi_request_var.get()
        return req.content_length

    @property
    def content_type(self):
        req = _fastapi_request_var.get()
        return req.content_type

    @property
    def form(self):
        req = _fastapi_request_var.get()
        return req.form

    @property
    def files(self):
        req = _fastapi_request_var.get()
        return req.files

    @property
    def path(self):
        req = _fastapi_request_var.get()
        return req.path

    @property
    def method(self):
        req = _fastapi_request_var.get()
        return req.method

    @property
    def url(self):
        req = _fastapi_request_var.get()
        return req.url

    @property
    def remote_addr(self):
        req = _fastapi_request_var.get()
        return req.remote_addr

    @property
    def args(self):
        req = _fastapi_request_var.get()
        return req.args

request = FlaskRequestProxy()

async def parse_and_set_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "")
    payload_json = None
    form_data = {}

    if "application/json" in content_type:
        try:
            payload_json = await raw_request.json()
        except Exception:
            payload_json = {}
    elif "multipart/form-data" in content_type:
        try:
            raw_form = await raw_request.form()
            form_data = {}
            for key in raw_form.keys():
                items = raw_form.getlist(key)
                wrapped_items = []
                for item in items:
                    if hasattr(item, "filename"):
                        wrapped_items.append(FlaskFileMock(item))
                    else:
                        wrapped_items.append(item)
                if len(wrapped_items) == 1 and not hasattr(raw_form[key], "filename"):
                    form_data[key] = wrapped_items[0]
                else:
                    form_data[key] = wrapped_items
        except Exception as e:
            logger.error(f"Error parsing form data: {e}")

    wrapper = FastAPIRequestWrapper(raw_request, payload_json, form_data)
    _fastapi_request_var.set(wrapper)

def jsonify(data, status_code=200):
    serializable_data = convert_to_json_serializable(data)
    return JSONResponse(content=serializable_data, status_code=status_code)

import os

from datetime import datetime
import traceback
import json
import numpy as np
from typing import Any

from config.settings import config_dict
from modules.matching.manager import AlgorithmManager
from modules.matching.vector_db import VectorDB
from utils.file_processor import FileProcessor
from utils.validators import RequestValidator


from modules.rag import Retriever, QualityLayer, PromptBuilder, LLMService, VectorStoreAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# JSON serialization helper function
def convert_to_json_serializable(obj):
    """Convert numpy/torch types to JSON serializable types"""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # For single-element numpy arrays
        return obj.item()
    else:
        return obj


def parse_json_from_llm(raw_text: str) -> dict:
    """Extract and parse JSON object robustly from LLM responses, even if wrapped in markdown code blocks."""
    if not raw_text:
        return {}
    
    clean_text = raw_text.strip()
    
    # 1. Try parsing directly
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass
        
    # 2. Try extracting from markdown code block ```json ... ```
    if "```" in clean_text:
        parts = clean_text.split("```")
        for part in parts:
            part_clean = part.strip()
            if part_clean.lower().startswith("json"):
                part_clean = part_clean[4:].strip()
            try:
                return json.loads(part_clean)
            except json.JSONDecodeError:
                pass
                
    # 3. Try finding first '{' and last '}'
    start_idx = clean_text.find('{')
    end_idx = clean_text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_candidate = clean_text[start_idx:end_idx + 1].strip()
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass
            
    # Fallback: return raw text inside dictionary
    return {
        "raw_text": raw_text,
        "parsing_failed": True
    }

def create_app(config_name='default'):
    """Application factory"""
    app = FastAPI()
    
    # Load configuration
    config = config_dict[config_name]
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
    app.config = ConfigWrapper(config)
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Response-Time", "X-Request-ID"]
    )
    
    @app.middleware("http")
    async def custom_middleware(raw_request: Request, call_next):
        start_time = time.time()
        request_id = f"req_{int(time.time())}_{hash(raw_request.url.path) % 10000}"
        logger.info("[%s] %s %s - Start", request_id, raw_request.method, raw_request.url.path)

        content_length = raw_request.headers.get("content-length")
        if content_length:
            try:
                content_length = int(content_length)
                max_size = 100 * 1024 * 1024
                if content_length > max_size:
                    return JSONResponse(
                        status_code=413,
                        content={
                            'error': 'Request too large',
                            'max_size_mb': max_size // (1024 * 1024),
                            'received_size_mb': content_length // (1024 * 1024),
                        }
                    )
            except ValueError:
                pass

        response = await call_next(raw_request)

        total_time = time.time() - start_time
        logger.info("[%s] %s - %.3fs", request_id, response.status_code, total_time)
        response.headers['X-Response-Time'] = f"{total_time:.3f}s"
        response.headers['X-Request-ID'] = request_id
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(raw_request: Request, exc: Exception):
        logger.error(f"Global exception: {exc}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': 'Internal Server Error',
                'message': str(exc),
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    # Initialize components
    algorithm_manager = AlgorithmManager(app.config)
    file_processor = FileProcessor(app.config)
    validator = RequestValidator(app.config)
    
    # Initialize Vector Database (PostgreSQL + pgvector)
    vector_db = None
    try:
        db_url = app.config.get('DATABASE_URL', '')
        vec_dim = app.config.get('VECTOR_DIMENSION', 384)
        if db_url:
            vector_db = VectorDB(db_url, vec_dim)
            vector_db.init_tables()
            vector_db.init_cv_table()
            algorithm_manager.vector_db = vector_db
            logger.info("Vector database initialized successfully")
    except Exception as e:
        logger.warning(f"Vector database not available: {e}. JD pre-computation endpoints will be disabled.")
        vector_db = None
    
    # Initialize RAG components
    retriever = Retriever()
    quality_layer = QualityLayer()
    prompt_builder = PromptBuilder()
    llm_service = LLMService()

    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    def _stringify_list(items):
        if not items:
            return ""
        return "\n".join([str(x).strip() for x in items if str(x).strip()])

    def _cv_json_to_text(cv: dict) -> str:
        """Convert structured CV JSON to a plain text resume."""
        if not isinstance(cv, dict):
            return str(cv or "")

        basics = cv.get('basics') or {}
        parts = []

        name = basics.get('name')
        headline = basics.get('headline')
        summary = basics.get('summary')
        email = basics.get('email')
        phone = basics.get('phone')
        location = basics.get('location')

        header_lines = []
        if name: header_lines.append(f"Name: {name}")
        if headline: header_lines.append(f"Headline: {headline}")
        if email: header_lines.append(f"Email: {email}")
        if phone: header_lines.append(f"Phone: {phone}")
        if location: header_lines.append(f"Location: {location}")
        if header_lines:
            parts.append("BASICS\n" + "\n".join(header_lines))

        if summary:
            parts.append("SUMMARY\n" + str(summary).strip())

        skills = cv.get('skills') or []
        if isinstance(skills, list) and skills:
            parts.append("SKILLS\n" + ", ".join([str(s).strip() for s in skills if str(s).strip()]))

        education = cv.get('education') or []
        if isinstance(education, list) and education:
            edu_lines = []
            for e in education:
                if not isinstance(e, dict):
                    continue
                inst = e.get('institution')
                field = e.get('field_of_study')
                sd = e.get('start_date')
                ed = e.get('end_date')
                line = " - ".join([str(x).strip() for x in [inst, field, f"{sd}–{ed}" if sd or ed else None] if x])
                if line:
                    edu_lines.append(line)
            if edu_lines:
                parts.append("EDUCATION\n" + "\n".join(edu_lines))

        experience = cv.get('experience') or []
        if isinstance(experience, list) and experience:
            exp_lines = []
            for ex in experience:
                if not isinstance(ex, dict):
                    continue
                proj = ex.get('project') or ex.get('title')
                timeline = ex.get('timeline')
                desc = ex.get('description')
                tech = ex.get('technologies') or []
                tasks = ex.get('tasks') or []

                block = []
                title_line = " - ".join([str(x).strip() for x in [proj, timeline] if x])
                if title_line:
                    block.append(title_line)
                if desc:
                    block.append(str(desc).strip())
                if isinstance(tech, list) and tech:
                    block.append("Tech: " + ", ".join([str(t).strip() for t in tech if str(t).strip()]))
                if isinstance(tasks, list) and tasks:
                    block.append("Tasks:\n" + "\n".join([f"- {str(t).strip()}" for t in tasks if str(t).strip()]))

                if block:
                    exp_lines.append("\n".join(block))
            if exp_lines:
                parts.append("EXPERIENCE\n" + "\n\n".join(exp_lines))

        certifications = cv.get('certifications') or []
        if isinstance(certifications, list) and certifications:
            parts.append("CERTIFICATIONS\n" + _stringify_list(certifications))

        languages = cv.get('languages') or []
        if isinstance(languages, list) and languages:
            parts.append("LANGUAGES\n" + _stringify_list(languages))

        return "\n\n".join([p for p in parts if p.strip()])

    def _job_json_to_text(job: dict) -> str:
        """Convert structured JD JSON to a plain text job description."""
        if not isinstance(job, dict):
            return str(job or "")

        parts = []
        title = (job.get('title') or {}).get('value') if isinstance(job.get('title'), dict) else job.get('title')
        if title:
            parts.append(f"TITLE\n{title}")

        tech = job.get('techStack')
        if isinstance(tech, dict):
            tech_val = tech.get('value') if isinstance(tech.get('value'), dict) else None
            if isinstance(tech_val, dict):
                langs = tech_val.get('languages') or []
                frws = tech_val.get('frameworks') or []
                tools = tech_val.get('tools') or []
                plats = tech_val.get('platforms') or []
                if langs or frws or tools or plats:
                    parts.append(
                        "TECH STACK\n"
                        + (f"Languages: {', '.join(langs)}\n" if langs else "")
                        + (f"Frameworks: {', '.join(frws)}\n" if frws else "")
                        + (f"Tools: {', '.join(tools)}\n" if tools else "")
                        + (f"Platforms: {', '.join(plats)}\n" if plats else "")
                    ).strip()

        responsibilities = job.get('responsibilities')
        if isinstance(responsibilities, dict):
            resp_val = responsibilities.get('value')
            if isinstance(resp_val, list) and resp_val:
                parts.append("RESPONSIBILITIES\n" + "\n".join([f"- {str(x).strip()}" for x in resp_val if str(x).strip()]))

        req = job.get('requirements')
        if isinstance(req, dict):
            req_val = req.get('value')
            if isinstance(req_val, dict):
                must = req_val.get('mustHave') or []
                nice = req_val.get('niceToHave') or []
                if must:
                    parts.append("MUST HAVE\n" + "\n".join([f"- {str(x).strip()}" for x in must if str(x).strip()]))
                if nice:
                    parts.append("NICE TO HAVE\n" + "\n".join([f"- {str(x).strip()}" for x in nice if str(x).strip()]))

        soft = job.get('softSkills')
        if isinstance(soft, dict):
            soft_val = soft.get('value')
            if isinstance(soft_val, list) and soft_val:
                parts.append("SOFT SKILLS\n" + "\n".join([f"- {str(x).strip()}" for x in soft_val if str(x).strip()]))

        constraints = job.get('constraints')
        if isinstance(constraints, dict):
            con_val = constraints.get('value')
            if isinstance(con_val, list) and con_val:
                parts.append("CONSTRAINTS\n" + "\n".join([f"- {str(x).strip()}" for x in con_val if str(x).strip()]))

        return "\n\n".join([p for p in parts if p.strip()])

    def _requirements_to_text(requirements: dict) -> str:
        """Convert explicit structured requirements to JD text for rule-based scoring."""
        if not isinstance(requirements, dict):
            return ""

        def _items(*keys):
            for key in keys:
                value = requirements.get(key)
                if isinstance(value, list):
                    return [str(x).strip() for x in value if str(x).strip()]
                if isinstance(value, str) and value.strip():
                    return [value.strip()]
            return []

        must = _items('mustHave', 'must_have', 'haveMust', 'have_must', 'required')
        nice = _items('niceToHave', 'nice_to_have', 'preferred')
        constraints = _items('constraints')

        parts = []
        if must:
            parts.append("MUST HAVE\n" + "\n".join([f"- {x}" for x in must]))
        if nice:
            parts.append("NICE TO HAVE\n" + "\n".join([f"- {x}" for x in nice]))
        if constraints:
            parts.append("CONSTRAINTS\n" + "\n".join([f"- {x}" for x in constraints]))
        return "\n\n".join(parts)
    
    # ===== VECTOR DB ENDPOINTS (Pre-computation) =====

    @app.api_route('/api/v1/jd/vectorize', methods=['POST'])
    async def vectorize_jd(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Pre-compute and store JD embedding in PostgreSQL pgvector.
        
        Body: { "job_id": "...", "job_text": "..." }
        Called by NestJS backend when a new Job Profile is finalized.
        """
        if not vector_db:
            return jsonify({'error': 'Vector database is not configured'}), 503

        data = request.get_json(silent=True) or {}
        job_id = str(data.get('job_id', '')).strip()
        job_text = str(data.get('job_text', '')).strip()

        # Also accept structured JD JSON
        if not job_text and isinstance(data.get('job'), dict):
            job_text = _job_json_to_text(data.get('job'))

        # Also accept requirements and prepend them
        requirements_text = _requirements_to_text(data.get('requirements'))
        if requirements_text:
            job_text = "\n\n".join([requirements_text, job_text]).strip()

        if not job_id:
            return jsonify({'error': 'job_id is required'}), 400
        if not job_text:
            return jsonify({'error': 'job_text, job JSON, or requirements is required'}), 400

        try:
            # Trigger background Celery task for JD vectorization
            from tasks import vectorize_jd_task
            vectorize_jd_task.delay(job_id, job_text)

            return jsonify({
                'success': True,
                'job_id': job_id,
                'message': 'JD vectorization task queued successfully in background Celery worker'
            }, 202)

        except Exception as e:
            logger.error(f"JD vectorize error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.api_route('/api/v1/jd/delete', methods=['POST'])
    async def delete_jd_vector(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Delete a stored JD vector embedding.
        
        Body: { "job_id": "..." }
        """
        if not vector_db:
            return jsonify({'error': 'Vector database is not configured'}), 503

        data = request.get_json(silent=True) or {}
        job_id = str(data.get('job_id', '')).strip()
        if not job_id:
            return jsonify({'error': 'job_id is required'}), 400

        try:
            vector_db.delete_jd_vector(job_id)
            return jsonify({'success': True, 'job_id': job_id, 'message': 'JD vector deleted'}), 200
        except Exception as e:
            logger.error(f"JD delete error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.api_route('/api/v1/cv/vectorize', methods=['POST'])
    async def vectorize_cv(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Pre-compute and store CV embedding in PostgreSQL pgvector.
        
        Body: { "cv_id": "...", "cv_text": "..." }
        """
        if not vector_db:
            return jsonify({'error': 'Vector database is not configured'}), 503

        data = request.get_json(silent=True) or {}
        cv_id = str(data.get('cv_id', '')).strip()
        cv_text = str(data.get('cv_text', '')).strip()

        if not cv_id:
            return jsonify({'error': 'cv_id is required'}), 400
        if not cv_text:
            return jsonify({'error': 'cv_text is required'}), 400

        try:
            # Trigger background Celery task for CV vectorization
            from tasks import vectorize_cv_task
            vectorize_cv_task.delay(cv_id, cv_text)

            return jsonify({
                'success': True,
                'cv_id': cv_id,
                'message': 'CV vectorization task queued successfully in background Celery worker'
            }, 202)

        except Exception as e:
            logger.error(f"CV vectorize error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.api_route('/api/v1/cv/delete', methods=['POST'])
    async def delete_cv_vector(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Delete a stored CV vector embedding.
        
        Body: { "cv_id": "..." }
        """
        if not vector_db:
            return jsonify({'error': 'Vector database is not configured'}), 503

        data = request.get_json(silent=True) or {}
        cv_id = str(data.get('cv_id', '')).strip()
        if not cv_id:
            return jsonify({'error': 'cv_id is required'}), 400

        try:
            vector_db.delete_cv_vector(cv_id)
            return jsonify({'success': True, 'cv_id': cv_id, 'message': 'CV vector deleted'}), 200
        except Exception as e:
            logger.error(f"CV delete error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.api_route('/api/v1/match', methods=['POST'])
    async def match_cv_jd(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Match a CV against a pre-stored JD vector using sBERT cosine similarity.
        
        Body: { "job_id": "...", "cv_text": "..." } or { "job_id": "...", "cv": {...} }
        Returns the sBERT similarity score computed via pgvector.
        """
        if not vector_db:
            return jsonify({'error': 'Vector database is not configured'}), 503

        data = request.get_json(silent=True) or {}
        job_id = str(data.get('job_id', '')).strip()
        cv_text = str(data.get('cv_text', '')).strip()

        # Accept structured CV JSON
        if not cv_text and isinstance(data.get('cv'), dict):
            cv_text = _cv_json_to_text(data.get('cv'))

        if not job_id:
            return jsonify({'error': 'job_id is required'}), 400
        if not cv_text:
            return jsonify({'error': 'cv_text (or cv JSON) is required'}), 400

        try:
            # Ensure sBERT model is loaded
            algorithm_manager.initialize_algorithms(['sbert'])
            sbert_alg = algorithm_manager.algorithms.get('sbert')
            if not sbert_alg or not hasattr(sbert_alg, 'model'):
                return jsonify({'error': 'sBERT model is not available'}), 503

            # Encode CV text into vector
            cv_embedding = sbert_alg.model.encode([cv_text])[0]
            cv_vector = cv_embedding.tolist()

            # Compute similarity using pgvector (no need to re-encode JD!)
            similarity = vector_db.cosine_similarity_score(job_id, cv_vector)

            if similarity is None:
                return jsonify({
                    'error': f'JD vector not found for job_id={job_id}. Call /api/v1/jd/vectorize first.'
                }), 404

            return jsonify({
                'success': True,
                'job_id': job_id,
                'sbert_score': round(float(max(0.0, min(1.0, similarity))), 4),
                'message': 'CV-JD matching completed using cached JD vector'
            }), 200

        except Exception as e:
            logger.error(f"CV-JD match error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    # ===== RAG AI SERVICE ENDPOINTS =====

    @app.api_route('/retrieve', methods=['GET', 'POST'])
    @app.api_route('/api/v1/rag/retrieve', methods=['GET', 'POST'])
    async def rag_retrieve(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Retrieve and process clean context chunks for interview prep."""
        try:
            if request.method == 'POST':
                data = request.get_json(silent=True) or {}
            else:
                data = request.args

            topic = data.get('topic')
            difficulty = data.get('difficulty')
            history = data.get('history')
            query_text = data.get('query_text')
            k = int(data.get('k', 5))

            raw_chunks = retriever.retrieve(
                topic=topic,
                difficulty=difficulty,
                history=history,
                query_text=query_text,
                k=k
            )
            similarity_weight = data.get('similarity_weight')
            quality_weight = data.get('quality_weight')
            if similarity_weight is not None and quality_weight is not None:
                ql = QualityLayer(
                    similarity_weight=float(similarity_weight),
                    quality_weight=float(quality_weight)
                )
            else:
                ql = quality_layer

            processed_res = ql.process(
                raw_chunks,
                topic=topic,
                difficulty=difficulty,
                k=k
            )

            prompt_preview = prompt_builder.build_question_generation_prompt(
                context_chunks=processed_res['ranked_chunks'],
                topic=topic or "Python OOP",
                difficulty=difficulty or "intermediate"
            )

            res_dict = dict(processed_res)
            res_dict['prompt_preview'] = prompt_preview

            return jsonify({
                'success': True,
                'data': convert_to_json_serializable(res_dict)
            }), 200

        except Exception as e:
            logger.error(f"RAG retrieve API error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.api_route('/interview', methods=['POST'])
    @app.api_route('/api/v1/rag/interview', methods=['POST'])
    async def rag_interview(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Generate a new question or follow-up question based on interview context."""
        try:
            data = request.get_json(silent=True) or {}
            topic = data.get('topic')
            difficulty = data.get('difficulty')
            history = data.get('history')  # list of chunk_ids
            conv_history = data.get('conversation_history')  # list of turns
            last_question = data.get('last_question')
            candidate_answer = data.get('candidate_answer')

            if not topic:
                return jsonify({'error': 'topic is required'}), 400
            if not difficulty:
                return jsonify({'error': 'difficulty is required'}), 400

            # 1. Retrieve & filter context chunks
            raw_chunks = retriever.retrieve(
                topic=topic,
                difficulty=difficulty,
                history=history,
                k=5
            )
            processed = quality_layer.process(
                raw_chunks,
                topic=topic,
                difficulty=difficulty
            )

            # 2. Decide if follow-up or initial question
            if candidate_answer and last_question:
                prompt = prompt_builder.build_follow_up_prompt(
                    context_chunks=processed['follow_ups'],
                    history=conv_history or [],
                    last_question=last_question,
                    candidate_answer=candidate_answer
                )
                q_type = "follow_up"
            else:
                prompt = prompt_builder.build_question_generation_prompt(
                    context_chunks=processed['ranked_chunks'],
                    topic=topic,
                    difficulty=difficulty,
                    history=history
                )
                q_type = "initial"

            # 3. Call LLM to generate the question
            question_text = llm_service.generate_text(prompt)

            return jsonify({
                'success': True,
                'question': question_text.strip(),
                'type': q_type,
                'prompt': prompt
            }), 200

        except Exception as e:
            logger.error(f"RAG interview API error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.api_route('/evaluate', methods=['POST'])
    @app.api_route('/api/v1/rag/evaluate', methods=['POST'])
    async def rag_evaluate(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Evaluate candidate's answer and generate constructive feedback."""
        try:
            data = request.get_json(silent=True) or {}
            question = data.get('question')
            candidate_answer = data.get('candidate_answer')
            topic = data.get('topic')
            difficulty = data.get('difficulty')

            if not question:
                return jsonify({'error': 'question is required'}), 400
            if not candidate_answer:
                return jsonify({'error': 'candidate_answer is required'}), 400

            # 1. Retrieve criteria chunks
            raw_chunks = retriever.retrieve(
                topic=topic,
                difficulty=difficulty,
                query_text=question,
                k=5
            )
            processed = quality_layer.process(
                raw_chunks,
                topic=topic,
                difficulty=difficulty
            )

            # 2. Build and run evaluation prompt
            eval_prompt = prompt_builder.build_evaluation_prompt(
                context_chunks=processed['ranked_chunks'],
                question=question,
                candidate_answer=candidate_answer
            )
            eval_raw = llm_service.generate_text(eval_prompt)

            # Parse evaluation result if LLM returned valid JSON
            evaluation_data = parse_json_from_llm(eval_raw)

            # 3. Build and run feedback prompt (coaching)
            feedback_prompt = prompt_builder.build_feedback_prompt(
                evaluation_details=evaluation_data,
                deliverables=processed['deliverables']
            )
            feedback_text = llm_service.generate_text(feedback_prompt)

            return jsonify({
                'success': True,
                'evaluation': evaluation_data,
                'feedback': feedback_text.strip()
            }), 200

        except Exception as e:
            logger.error(f"RAG evaluate API error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.api_route('/import-csv', methods=['POST'])
    @app.api_route('/api/v1/rag/import-csv', methods=['POST'])
    async def rag_import_csv(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Import multiple interview documents from a CSV file."""
        import csv
        import io
        from modules.rag import upsert_interview_document

        try:
            file = request.files.get('file')
            if not file:
                return jsonify({'error': 'No file uploaded under key "file"'}), 400

            filename = file.filename or ''
            if not filename.endswith('.csv'):
                return jsonify({'error': 'Uploaded file must be a CSV file (.csv)'}), 400

            # Decode stream
            try:
                stream = io.StringIO(file.stream.read().decode("utf-8-sig"), newline=None)
            except Exception as dec_err:
                logger.error(f"Failed to decode CSV file: {dec_err}")
                return jsonify({'error': 'Failed to decode CSV file. Ensure it is encoded in UTF-8.'}), 400

            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                return jsonify({'error': 'CSV file is empty or missing headers'}), 400

            def _parse_csv_list(val: Any) -> list:
                if not val:
                    return []
                val_str = str(val).strip()
                if not val_str:
                    return []
                # Split by semicolon or newline
                delimiters = [';', '\n']
                parts = [val_str]
                for delim in delimiters:
                    new_parts = []
                    for p in parts:
                        new_parts.extend(p.split(delim))
                    parts = new_parts
                return [p.strip() for p in parts if p.strip()]

            imported_docs = []
            failed_rows = []

            for idx, row in enumerate(reader, 1):
                # Clean headers and values
                row_clean = {str(k).strip().lower(): v for k, v in row.items() if k is not None}

                doc_id = row_clean.get('document_id') or row_clean.get('id')
                topic_name = row_clean.get('topic_name') or row_clean.get('topic')
                difficulty_level = row_clean.get('difficulty_level') or row_clean.get('difficulty') or row_clean.get('level')

                # Skip completely empty rows or rows missing identity
                if not doc_id and not topic_name:
                    continue

                if not doc_id:
                    failed_rows.append({'row': idx, 'error': 'Missing document_id or id'})
                    continue
                if not topic_name:
                    failed_rows.append({'row': idx, 'error': 'Missing topic_name or topic'})
                    continue

                # Build nested document object matching schema
                doc_obj = {
                    "document": {
                        "document_id": str(doc_id).strip(),
                        "version": str(row_clean.get('version', '1.0.0')).strip(),
                        "status": str(row_clean.get('status', 'approved')).strip(),
                        "language": str(row_clean.get('language', 'vi')).strip()
                    },
                    "topic": {
                        "domain": str(row_clean.get('domain', 'general')).strip(),
                        "topic_name": str(topic_name).strip(),
                        "role_targets": _parse_csv_list(row_clean.get('role_targets') or row_clean.get('roles')),
                        "job_levels": _parse_csv_list(row_clean.get('job_levels') or row_clean.get('levels'))
                    },
                    "difficulty": {
                        "level": str(difficulty_level or 'intermediate').strip()
                    },
                    "knowledge": {
                        "summary": str(row_clean.get('knowledge_summary') or row_clean.get('knowledge') or row_clean.get('summary') or '').strip(),
                        "concepts": _parse_csv_list(row_clean.get('knowledge_concepts') or row_clean.get('concepts'))
                    },
                    "expected_points": {
                        "must_have": _parse_csv_list(row_clean.get('expected_points_must_have') or row_clean.get('expected_points') or row_clean.get('must_have'))
                    },
                    "common_mistakes": {
                        "mistakes": _parse_csv_list(row_clean.get('common_mistakes') or row_clean.get('mistakes'))
                    },
                    "follow_up": {
                        "questions": _parse_csv_list(row_clean.get('follow_up_questions') or row_clean.get('follow_up'))
                    },
                    "deliverables": {
                        "action_items": _parse_csv_list(row_clean.get('deliverables_action_items') or row_clean.get('deliverables') or row_clean.get('action_items'))
                    },
                    "metadata": {
                        "retrieval": {
                            "is_active": str(row_clean.get('is_active', 'true')).strip().lower() in ('true', '1', 'yes'),
                            "quality_score": float(row_clean.get('quality_score') or 0.8)
                        }
                    }
                }

                try:
                    upsert_interview_document(document_obj=doc_obj)
                    imported_docs.append(str(doc_id).strip())
                except Exception as row_err:
                    logger.error(f"Failed to upsert RAG document from CSV row {idx}: {row_err}")
                    failed_rows.append({'row': idx, 'document_id': doc_id, 'error': str(row_err)})

            return jsonify({
                'success': True,
                'message': f'Imported {len(imported_docs)} documents successfully from CSV.',
                'imported_document_ids': imported_docs,
                'failed_rows': failed_rows
            }), 200

        except Exception as e:
            logger.error(f"RAG import-csv API error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.api_route('/upsert-document', methods=['POST'])
    @app.api_route('/api/v1/rag/upsert-document', methods=['POST'])
    async def rag_upsert_document(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Upsert a single interview document from a raw JSON payload."""
        from modules.rag import upsert_interview_document
        try:
            data = request.get_json(silent=True) or {}
            doc_id = data.get('document', {}).get('document_id')
            if not doc_id:
                return jsonify({'error': 'document.document_id is required'}), 400

            res = upsert_interview_document(document_obj=data)
            return jsonify({
                'success': True,
                'message': f"Document '{doc_id}' upserted successfully.",
                'records': res.get('records', 0)
            }), 200
        except Exception as e:
            logger.error(f"RAG upsert-document API error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.api_route('/api/v1/rag/documents', methods=['GET'])
    async def rag_list_documents(raw_request: Request):
        await parse_and_set_request(raw_request)
        """List all unique documents in the RAG knowledge base."""
        try:
            adapter = VectorStoreAdapter()
            docs = adapter.get_all_documents()
            return jsonify({
                'success': True,
                'data': docs
            }), 200
        except Exception as e:
            logger.error(f"RAG list-documents API error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.api_route('/api/v1/rag/documents/{document_id}', methods=['GET'])
    async def rag_get_document(document_id: str, raw_request: Request):
        await parse_and_set_request(raw_request)
        """Get a document's reconstructed details and all its chunks."""
        try:
            adapter = VectorStoreAdapter()
            chunks = adapter.get_document_chunks(document_id)
            if not chunks:
                return jsonify({'error': f"Document '{document_id}' not found"}), 404
            
            doc_obj = {
                "document": {
                    "document_id": document_id,
                    "version": "1.0.0",
                    "status": "approved",
                    "language": "vi",
                    "updated_at": ""
                },
                "topic": {
                    "domain": "general",
                    "topic_name": "",
                    "role_targets": [],
                    "job_levels": []
                },
                "difficulty": {
                    "level": "intermediate"
                },
                "knowledge": {
                    "summary": "",
                    "concepts": []
                },
                "expected_points": {
                    "must_have": []
                },
                "common_mistakes": {
                    "mistakes": []
                },
                "follow_up": {
                    "questions": []
                },
                "deliverables": {
                    "action_items": []
                },
                "metadata": {
                    "retrieval": {
                        "is_active": True,
                        "quality_score": 0.8
                    }
                }
            }
            
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                doc_obj["document"]["version"] = meta.get("version", "1.0.0")
                doc_obj["document"]["language"] = meta.get("language", "vi")
                doc_obj["document"]["updated_at"] = meta.get("updated_at", "")
                doc_obj["topic"]["domain"] = meta.get("domain", "general")
                doc_obj["topic"]["topic_name"] = meta.get("topic", "")
                doc_obj["topic"]["role_targets"] = meta.get("roles", [])
                doc_obj["topic"]["job_levels"] = meta.get("job_levels", [])
                doc_obj["difficulty"]["level"] = meta.get("difficulty", "intermediate")
                doc_obj["metadata"]["retrieval"]["is_active"] = meta.get("is_active", True)
                doc_obj["metadata"]["retrieval"]["quality_score"] = meta.get("quality_score", 0.8)
                
                c_type = meta.get("chunk_type")
                text = chunk.get("text", "")
                if c_type == "knowledge":
                    lines = text.split('\n')
                    summary_lines = []
                    concepts_lines = []
                    in_concepts = False
                    for line in lines:
                        if line.startswith("summary:"):
                            summary_lines.append(line.replace("summary:", "", 1).strip())
                        elif line.startswith("concepts:"):
                            in_concepts = True
                            concepts_lines.append(line.replace("concepts:", "", 1).strip())
                        else:
                            if in_concepts:
                                concepts_lines.append(line.strip())
                            else:
                                summary_lines.append(line.strip())
                    doc_obj["knowledge"]["summary"] = "\n".join(summary_lines).strip()
                    doc_obj["knowledge"]["concepts"] = concepts_lines
                elif c_type == "expected_points":
                    lines = [l.replace("must_have:", "", 1).strip() if l.startswith("must_have:") else l.strip() for l in text.split('\n') if l.strip()]
                    doc_obj["expected_points"]["must_have"] = lines
                elif c_type == "common_mistakes":
                    lines = [l.replace("mistakes:", "", 1).strip() if l.startswith("mistakes:") else l.strip() for l in text.split('\n') if l.strip()]
                    doc_obj["common_mistakes"]["mistakes"] = lines
                elif c_type == "follow_up":
                    lines = [l.replace("questions:", "", 1).strip() if l.startswith("questions:") else l.strip() for l in text.split('\n') if l.strip()]
                    doc_obj["follow_up"]["questions"] = lines
                elif c_type == "deliverables":
                    lines = [l.replace("action_items:", "", 1).strip() if l.startswith("action_items:") else l.strip() for l in text.split('\n') if l.strip()]
                    doc_obj["deliverables"]["action_items"] = lines
            
            return jsonify({
                'success': True,
                'data': {
                    'document': doc_obj,
                    'chunks': chunks
                }
            }), 200
        except Exception as e:
            logger.error(f"RAG get-document API error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.api_route('/api/v1/rag/documents/{document_id}/toggle', methods=['POST'])
    async def rag_toggle_document(document_id: str, raw_request: Request):
        await parse_and_set_request(raw_request)
        """Toggle active status of a document."""
        try:
            data = request.get_json(silent=True) or {}
            is_active = data.get('is_active', True)
            adapter = VectorStoreAdapter()
            res = adapter.toggle_document_active(document_id, is_active)
            return jsonify({
                'success': True,
                'message': f"Document '{document_id}' active status set to {is_active}.",
                'updated': res.get('updated', 0)
            }), 200
        except Exception as e:
            logger.error(f"RAG toggle-document API error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.api_route('/api/v1/rag/documents/{document_id}', methods=['DELETE'])
    async def rag_delete_document(document_id: str, raw_request: Request):
        await parse_and_set_request(raw_request)
        """Delete all chunks for a document."""
        try:
            adapter = VectorStoreAdapter()
            res = adapter.delete_document(document_id)
            return jsonify({
                'success': True,
                'message': f"Document '{document_id}' deleted successfully.",
                'deleted': res.get('deleted', 0)
            }), 200
        except Exception as e:
            logger.error(f"RAG delete-document API error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.api_route('/api/v1/rag/documents/bulk-delete', methods=['POST'])
    async def rag_bulk_delete_documents(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Delete all chunks for multiple documents in a single request."""
        try:
            data = request.get_json(silent=True) or {}
            document_ids = data.get('document_ids', [])
            if not document_ids or not isinstance(document_ids, list):
                return jsonify({'error': 'document_ids must be a non-empty list'}), 400
            
            adapter = VectorStoreAdapter()
            total_deleted = 0
            if adapter.provider in {"postgresql", "postgres"}:
                adapter.init_rag_table()
                conn = adapter._get_db_connection()
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM rag_knowledge_chunks WHERE document_id = ANY(%s);", (document_ids,))
                    total_deleted = cur.rowcount
                conn.commit()
                conn.close()
                success = True
            else:
                success = False
                for doc_id in document_ids:
                    res = adapter.delete_document(doc_id)
                    total_deleted += res.get('deleted', 0)
                    if res.get('success'):
                        success = True
                        
            return jsonify({
                'success': True,
                'message': f"Deleted {len(document_ids)} documents successfully.",
                'deleted': total_deleted
            }), 200
        except Exception as e:
            logger.error(f"RAG bulk delete-document API error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.api_route('/api/v1/rag/documents/bulk-toggle', methods=['POST'])
    async def rag_bulk_toggle_documents(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Toggle active status of multiple documents at once."""
        try:
            data = request.get_json(silent=True) or {}
            document_ids = data.get('document_ids', [])
            is_active = data.get('is_active')
            if not document_ids or not isinstance(document_ids, list):
                return jsonify({'error': 'document_ids must be a non-empty list'}), 400
            if is_active is None:
                return jsonify({'error': 'is_active is required'}), 400
            
            adapter = VectorStoreAdapter()
            total_updated = 0
            if adapter.provider in {"postgresql", "postgres"}:
                adapter.init_rag_table()
                conn = adapter._get_db_connection()
                with conn.cursor() as cur:
                    active_str = "true" if is_active else "false"
                    cur.execute(f"""
                        UPDATE rag_knowledge_chunks
                        SET metadata = jsonb_set(metadata, '{{is_active}}', %s::jsonb)
                        WHERE document_id = ANY(%s);
                    """, (active_str, document_ids))
                    total_updated = cur.rowcount
                conn.commit()
                conn.close()
                success = True
            else:
                success = False
                for doc_id in document_ids:
                    res = adapter.toggle_document_active(doc_id, is_active)
                    total_updated += res.get('updated', 0)
                    if res.get('success'):
                        success = True
            
            return jsonify({
                'success': True,
                'message': f"Updated {len(document_ids)} documents successfully.",
                'updated': total_updated
            }), 200
        except Exception as e:
            logger.error(f"RAG bulk toggle-document API error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.api_route('/api/v1/rag/chunks/{chunk_id}', methods=['PUT'])
    async def rag_update_chunk(chunk_id: str, raw_request: Request):
        await parse_and_set_request(raw_request)
        """Update a specific chunk text and/or quality score."""
        try:
            data = request.get_json(silent=True) or {}
            new_text = data.get('text')
            new_metadata = data.get('metadata', {})
            if not new_text:
                return jsonify({'error': 'text is required'}), 400
            
            adapter = VectorStoreAdapter()
            res = adapter.update_chunk(chunk_id, new_text, new_metadata)
            if not res.get('success'):
                return jsonify({'error': res.get('error', 'Update failed')}), 400
            
            return jsonify({
                'success': True,
                'message': f"Chunk '{chunk_id}' updated successfully."
            }), 200
        except Exception as e:
            logger.error(f"RAG update-chunk API error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.api_route('/api/v1/rag/documents/{document_id}/evaluate', methods=['POST'])
    async def rag_evaluate_document(document_id: str, raw_request: Request):
        await parse_and_set_request(raw_request)
        """Evaluate document quality using LLM."""
        try:
            adapter = VectorStoreAdapter()
            chunks = adapter.get_document_chunks(document_id)
            if not chunks:
                return jsonify({'error': f"Document '{document_id}' not found"}), 404
            
            doc_data = {
                "document_id": document_id,
                "chunks": [{"type": c.get("metadata", {}).get("chunk_type"), "text": c.get("text")} for c in chunks]
            }
            
            prompt = prompt_builder.build_quality_evaluation_prompt(doc_data)
            response_text = llm_service.generate_text(prompt)
            
            eval_data = parse_json_from_llm(response_text)
            if "parsing_failed" in eval_data:
                eval_data = {
                    "score": 0.8,
                    "findings": ["Không thể phân tích phản hồi JSON từ LLM."],
                    "suggestions": [],
                    "adjusted_quality_score": 0.8,
                    "raw_response": response_text
                }

            return jsonify({
                'success': True,
                'evaluation': eval_data
            }), 200
        except Exception as e:
            logger.error(f"RAG evaluate-document API error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    # ===== EXISTING ENDPOINTS =====
    
    @app.api_route('/api/health', methods=['GET'])
    async def health_check(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'algorithms': algorithm_manager.get_algorithm_status(),
            'version': '1.0.0'
        })
    
    @app.api_route('/api/algorithms', methods=['GET'])
    async def get_algorithms(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Get available algorithms"""
        return jsonify(algorithm_manager.get_algorithm_status())
    
    
    @app.api_route('/api/process-resumes', methods=['POST'])
    async def process_resumes(raw_request: Request):
        await parse_and_set_request(raw_request)
        """Main endpoint for processing resumes"""
        start_time = datetime.utcnow()
        
        try:
            # Validate request
            validation_result = validator.validate_process_request(request)
            if not validation_result['valid']:
                logger.warning(f"Invalid process request: {validation_result}")
                return jsonify({'error': validation_result['error']}), 400

            # Extract request data (supports both multipart and JSON)
            default_methods = list(app.config.get('DEFAULT_ALGORITHMS', ['requirements', 'sbert', 'bm25', 'cosine', 'ner']))

            files = []
            successful_files = []
            failed_files = []
            resume_texts = []
            job_id = None
            cv_id = None

            if request.is_json:
                payload = request.get_json(silent=True) or {}
                position = payload.get('position', 'general')

                methods = payload.get('methods') or default_methods
                if isinstance(methods, str):
                    methods = [methods]

                options = payload.get('options') if isinstance(payload.get('options'), dict) else {}
                metadata = payload.get('metadata') if isinstance(payload.get('metadata'), dict) else {}
                job_id = metadata.get('jobId') or metadata.get('job_id') or payload.get('job_id')
                cv_id = metadata.get('candidateId') or metadata.get('candidate_id') or metadata.get('cv_id') or payload.get('candidate_id') or payload.get('cv_id')

                if not job_id:
                    return jsonify({'error': 'job_id is required in metadata or request payload'}), 400
                if not vector_db:
                    return jsonify({'error': 'Vector database is not configured/available'}), 503

                jd_cache = vector_db.get_jd_cache(job_id)
                if not jd_cache or not jd_cache.get('job_text'):
                    return jsonify({'error': f"Job ID {job_id} not found in database cache. Please vectorize the JD first."}), 404

                job_description = jd_cache['job_text']

                resumes = payload.get('resumes') or payload.get('cvs') or payload.get('cv') or []
                if isinstance(resumes, (dict, str)):
                    resumes = [resumes]

                for idx, cv in enumerate(resumes):
                    try:
                        if isinstance(cv, dict):
                            text = _cv_json_to_text(cv)
                            name = ((cv.get('basics') or {}).get('name')) if isinstance(cv.get('basics'), dict) else None
                            filename = f"{name}.json" if name else f"resume_{idx + 1}.json"
                        else:
                            text = str(cv or "")
                            filename = f"resume_{idx + 1}.txt"

                        if not str(text).strip():
                            raise ValueError("Empty resume text after conversion")

                        successful_files.append({
                            'success': True,
                            'filename': filename,
                            'text': text,
                            'size': len(text.encode('utf-8', errors='ignore')),
                            'word_count': len(str(text).split()),
                            'char_count': len(str(text))
                        })
                    except Exception as e:
                        failed_files.append({
                            'success': False,
                            'filename': f"resume_{idx + 1}",
                            'error': str(e)
                        })

                if not successful_files:
                    return jsonify({
                        'error': 'No resumes could be processed successfully',
                        'failed_files': failed_files
                    }), 400

                resume_texts = [f['text'] for f in successful_files]
                logger.info(f"Processing JSON request: {len(successful_files)} resumes, {len(methods)} algorithms")

            else:
                files = request.files.getlist('resumes')
                position = request.form.get('position', 'general')
                methods = request.form.getlist('methods') or default_methods

                # Parse options
                options_str = request.form.get('options', '{}')
                try:
                    options = json.loads(options_str)
                except json.JSONDecodeError:
                    options = {}

                # Parse metadata
                metadata_str = request.form.get('metadata', '{}')
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    metadata = {}
                job_id = metadata.get('jobId') or metadata.get('job_id')
                cv_id = metadata.get('candidateId') or metadata.get('candidate_id') or metadata.get('cv_id')

                if not job_id:
                    return jsonify({'error': 'job_id is required in metadata form field'}), 400
                if not vector_db:
                    return jsonify({'error': 'Vector database is not configured/available'}), 503

                jd_cache = vector_db.get_jd_cache(job_id)
                if not jd_cache or not jd_cache.get('job_text'):
                    return jsonify({'error': f"Job ID {job_id} not found in database cache. Please vectorize the JD first."}), 404

                job_description = jd_cache['job_text']

                logger.info(f"Processing multipart request: {len(files)} files, {len(methods)} algorithms")

                # Process files
                processed_files = file_processor.process_files(files)

                # Filter successful extractions
                successful_files = [f for f in processed_files if f['success']]
                failed_files = [f for f in processed_files if not f['success']]

                if not successful_files:
                    return jsonify({
                        'error': 'No files could be processed successfully',
                        'failed_files': failed_files
                    }), 400

                # Extract resume texts
                resume_texts = [f['text'] for f in successful_files]
            
            algorithm_results = algorithm_manager.process_resumes_parallel(
                resume_texts, job_description, methods, position, job_id=job_id, cv_id=cv_id, options=options
            )
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # Prepare response with JSON serialization
            response = {
                'success': True,
                'timestamp': end_time.isoformat(),
                'processing_time_seconds': float(processing_time),
                'summary': {
                    'total_resumes_uploaded': len(files) if files else len(successful_files),
                    'successfully_processed': len(successful_files),
                    'failed_to_process': len(failed_files),
                    'algorithms_used': methods,
                    'job_position': position,
                    'processing_options': options
                },
                'results': [],
                'failed_files': failed_files,
                'algorithm_performance': convert_to_json_serializable(algorithm_results.get('algorithm_performance', {})),
                'metadata': {
                    **metadata,
                    'processing_completed_at': end_time.isoformat(),
                    'server_version': '1.0.0',
                }
            }
            
            # Format results for frontend
            for i, combined_result in enumerate(algorithm_results['combined_results']):
                original_file_info = successful_files[combined_result['resume_index']]
                
                # Calculate explanation
                explanation = _generate_explanation(combined_result, job_description, position)
                
                # Extract skills if NER was used
                extracted_skills = []
                if 'ner' in combined_result['algorithm_scores']:
                    ner_details = combined_result['algorithm_scores']['ner'].get('details', {})
                    extracted_skills = _extract_skills_list(ner_details.get('extracted_skills', {}))
                
                result_entry = {
                    'filename': original_file_info['filename'],
                    'rank': int(combined_result['rank']),
                    'final_score': float(combined_result['combined_score']),
                    'weighted_score': float(combined_result['weighted_score']),
                    'scores': {alg: float(data['score']) for alg, data in combined_result['algorithm_scores'].items()},
                    'score_breakdown': convert_to_json_serializable(combined_result.get('score_breakdown', {})),
                    'explanation': explanation,
                    'extracted_skills': extracted_skills[:20],
                    'file_info': {
                        'size': int(original_file_info['size']),
                        'word_count': int(original_file_info['word_count']),
                        'char_count': int(original_file_info['char_count'])
                    },
                    'algorithm_details': convert_to_json_serializable(combined_result['algorithm_scores']),
                    'errors': combined_result.get('errors', [])
                }
                
                response['results'].append(result_entry)
            
            # Sort results by rank
            response['results'].sort(key=lambda x: x['rank'])
            
            logger.info(f"Successfully processed {len(successful_files)} resumes in {processing_time:.2f}s")
            
            # Final conversion to ensure everything is JSON serializable
            response = convert_to_json_serializable(response)
            
            return jsonify(response)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Processing error: {error_msg}")
            logger.error(traceback.format_exc())
            
            return jsonify({
                'success': False,
                'error': error_msg,
                'timestamp': datetime.utcnow().isoformat()
            }), 500

    def _generate_explanation(combined_result: dict, job_description: str, position: str = None) -> str:
        """Generate human-readable explanation for ranking based on the three-pillar architecture"""
        try:
            score = float(combined_result['combined_score'])
            breakdown = combined_result.get('score_breakdown', {})
            
            if score >= 0.8:
                rating = "Excellent match"
            elif score >= 0.6:
                rating = "Good match"
            elif score >= 0.4:
                rating = "Fair match"
            else:
                rating = "Poor match"
                
            pos_str = f" for {position}" if position else ""
            explanation_parts = [f"{rating}{pos_str} (Overall: {score:.1%})."]
            
            # Semantic feedback
            if breakdown.get('semantic', {}).get('active'):
                sem_score = breakdown['semantic']['score']
                if sem_score >= 0.75:
                    explanation_parts.append(f"Strong semantic alignment with the job context ({sem_score:.1%}).")
                elif sem_score < 0.50:
                    explanation_parts.append(f"Weak semantic match with the role's requirements ({sem_score:.1%}).")
                    
            # Lexical feedback
            if breakdown.get('lexical', {}).get('active'):
                lex_score = breakdown['lexical']['score']
                # Try to extract key matching terms from cosine or bm25
                matching_terms = []
                for alg_name in ('cosine', 'bm25', 'jaccard'):
                    alg_data = combined_result['algorithm_scores'].get(alg_name, {})
                    top_terms = alg_data.get('details', {}).get('top_matching_terms', [])
                    if top_terms:
                        if isinstance(top_terms[0], dict):
                            matching_terms = [t['term'] for t in top_terms[:3]]
                        else:
                            matching_terms = [str(t) for t in top_terms[:3]]
                        break
                    
                term_str = f" (key matches: {', '.join(matching_terms)})" if matching_terms else ""
                if lex_score >= 0.60:
                    explanation_parts.append(f"Good keyword matching{term_str}.")
                    
            # Requirements & penalties feedback
            penalties = breakdown.get('penalties', {})
            missing_must_count = penalties.get('missing_must_count', 0)
            missing_list = penalties.get('missing_must_list', [])
            
            # Get experience details if available
            req_info = breakdown.get('requirements', {})
            has_exp_info = 'required_years_experience' in req_info
            
            if has_exp_info:
                req_yrs = req_info.get('required_years_experience', 0)
                res_yrs = req_info.get('resume_years_experience', 0)
                if req_yrs > 0:
                    explanation_parts.append(f"Experience: {res_yrs} years found (Required: {req_yrs} years).")
            
            if missing_must_count > 0:
                must_have_mult = penalties.get('must_have_penalty_multiplier', 1.0)
                penalty_pct = (1.0 - must_have_mult) * 100
                list_str = f" ({', '.join(missing_list[:3])})" if missing_list else ""
                explanation_parts.append(f"Missing {missing_must_count} must-have requirement(s){list_str}, resulting in a -{penalty_pct:.0f}% penalty.")
                
            if penalties.get('constraints_ok') is False:
                explanation_parts.append("Failed one or more constraints (e.g., location or language), applying a -8% penalty.")
                
            return " ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Analysis completed with combined score of {float(combined_result.get('combined_score', 0)):.1%}"
    
    def _extract_skills_list(extracted_skills: dict) -> list:
        """Extract flat list of skills from NER results"""
        skills = []
        try:
            for category, skill_list in extracted_skills.items():
                for skill_data in skill_list:
                    if isinstance(skill_data, dict) and 'skill' in skill_data:
                        skills.append(skill_data['skill'].title())
                    elif isinstance(skill_data, str):
                        skills.append(skill_data.title())
            return list(set(skills))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error extracting skills: {e}")
            return []
    
    return app

# Expose global ASGI app variable for uvicorn
app = create_app(os.getenv('FLASK_ENV', 'development'))

if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', 5001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
