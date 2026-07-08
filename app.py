"""CV–JD scoring service API (multi-criteria fusion). Academic/train endpoints removed."""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime
import traceback
import json
import numpy as np

from config.settings import config_dict
from modules.matching.manager import AlgorithmManager
from modules.matching.vector_db import VectorDB
from utils.file_processor import FileProcessor
from utils.validators import RequestValidator
from api.error_handlers import register_error_handlers
from api.middleware import setup_middleware

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

def create_app(config_name='default'):
    """Application factory"""
    app = Flask(__name__)
    
    # Load configuration
    config = config_dict[config_name]
    app.config.from_object(config)
    
    # Enable CORS
    CORS(app, origins=['https://resume.createfast.tech', 'http://localhost:3001'])
    
    # Setup middlewareTr
    setup_middleware(app)
    
    # Register error handlers
    register_error_handlers(app)
    
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
            algorithm_manager.vector_db = vector_db
            logger.info("Vector database initialized successfully")
    except Exception as e:
        logger.warning(f"Vector database not available: {e}. JD pre-computation endpoints will be disabled.")
        vector_db = None
    
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

    @app.route('/api/v1/jd/vectorize', methods=['POST'])
    def vectorize_jd():
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
            # Ensure semantic models are loaded
            models_to_load = []
            for m in ['sbert', 'bert', 'distilbert']:
                if m in algorithm_manager.algorithm_registry:
                    models_to_load.append(m)

            algorithm_manager.initialize_algorithms(models_to_load)

            import hashlib
            checksum = hashlib.sha256(job_text.encode('utf-8')).hexdigest()

            # Encode and upsert for each active model
            # sBERT
            sbert_alg = algorithm_manager.algorithms.get('sbert')
            if sbert_alg and hasattr(sbert_alg, 'model'):
                sbert_vector = sbert_alg.model.encode([job_text])[0].tolist()
                vector_db.upsert_jd_cache(job_id, checksum, 'sbert', sbert_vector, job_text=job_text)

            # BERT
            bert_alg = algorithm_manager.algorithms.get('bert')
            if bert_alg and hasattr(bert_alg, '_get_embeddings'):
                bert_vector = bert_alg._get_embeddings(job_text)[0].tolist()
                vector_db.upsert_jd_cache(job_id, checksum, 'bert', bert_vector, job_text=job_text)

            # DistilBERT
            distilbert_alg = algorithm_manager.algorithms.get('distilbert')
            if distilbert_alg and hasattr(distilbert_alg, '_embed'):
                cleaned = distilbert_alg._clean(job_text)
                distilbert_vector = distilbert_alg._embed(cleaned)[0].tolist()
                vector_db.upsert_jd_cache(job_id, checksum, 'distilbert', distilbert_vector, job_text=job_text)

            return jsonify({
                'success': True,
                'job_id': job_id,
                'message': 'JD vectorized and stored successfully across semantic models'
            }), 200

        except Exception as e:
            logger.error(f"JD vectorize error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.route('/api/v1/jd/delete', methods=['POST'])
    def delete_jd_vector():
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

    @app.route('/api/v1/match', methods=['POST'])
    def match_cv_jd():
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

    # ===== EXISTING ENDPOINTS =====
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'algorithms': algorithm_manager.get_algorithm_status(),
            'version': '1.0.0'
        })
    
    @app.route('/api/algorithms', methods=['GET'])
    def get_algorithms():
        """Get available algorithms"""
        return jsonify(algorithm_manager.get_algorithm_status())
    
    
    @app.route('/api/process-resumes', methods=['POST'])
    def process_resumes():
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

            if request.is_json:
                payload = request.get_json(silent=True) or {}
                position = payload.get('position', 'general')

                methods = payload.get('methods') or default_methods
                if isinstance(methods, str):
                    methods = [methods]

                options = payload.get('options') if isinstance(payload.get('options'), dict) else {}
                metadata = payload.get('metadata') if isinstance(payload.get('metadata'), dict) else {}
                job_id = metadata.get('jobId') or metadata.get('job_id') or payload.get('job_id')

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
                resume_texts, job_description, methods, position, job_id=job_id
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

if __name__ == '__main__':
    app = create_app(os.getenv('FLASK_ENV', 'development'))
    port = int(os.getenv('PORT', app.config.get('PORT', 5001)))
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])
