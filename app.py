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
from core.algorithm_manager import AlgorithmManager
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
    
    @app.route('/api/positions', methods=['GET'])
    def get_positions():
        """Get available job positions"""
        positions = [
            {'value': 'sde', 'label': 'Software Development Engineer', 'icon': '💻'},
            {'value': 'swe', 'label': 'Software Engineer', 'icon': '⚙️'},
            {'value': 'ml_engineer', 'label': 'ML Engineer', 'icon': '🤖'},
            {'value': 'data_scientist', 'label': 'Data Scientist', 'icon': '📊'},
            {'value': 'devops', 'label': 'DevOps Engineer', 'icon': '🔧'},
            {'value': 'frontend', 'label': 'Frontend Developer', 'icon': '🎨'},
            {'value': 'backend', 'label': 'Backend Developer', 'icon': '🗄️'},
            {'value': 'fullstack', 'label': 'Full Stack Developer', 'icon': '🚀'},
            {'value': 'product_manager', 'label': 'Product Manager', 'icon': '📱'},
            {'value': 'designer', 'label': 'UI/UX Designer', 'icon': '🎭'},
            {'value': 'general', 'label': 'General', 'icon': '📋'}
        ]
        return jsonify(positions)
    
    @app.route('/api/supported-formats', methods=['GET'])
    def get_supported_formats():
        """Get supported file formats"""
        return jsonify({
            'formats': ['.pdf', '.docx', '.doc'],
            'max_file_size': app.config['MAX_CONTENT_LENGTH'],
            'max_files': app.config['MAX_FILES_PER_REQUEST'],
            'supported_mime_types': [
                'application/pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/msword'
            ]
        })
    
    @app.route('/api/validate-files', methods=['POST'])
    def validate_files():
        """Validate uploaded files without processing"""
        try:
            files = request.files.getlist('files')
            
            if not files:
                return jsonify({'error': 'No files provided'}), 400
            
            validation_results = []
            for i, file in enumerate(files):
                result = file_processor.validate_file(file)
                result['index'] = i
                validation_results.append(result)
            
            return jsonify({
                'results': validation_results,
                'total_files': len(files),
                'valid_files': sum(1 for r in validation_results if r['valid'])
            })
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return jsonify({'error': 'File validation failed'}), 500
    
    @app.route('/api/process-resumes', methods=['POST'])
    def process_resumes():
        """Main endpoint for processing resumes"""
        start_time = datetime.utcnow()
        
        try:
            # Validate request
            validation_result = validator.validate_process_request(request)
            if not validation_result['valid']:
                return jsonify({'error': validation_result['error']}), 400

            # Extract request data (supports both multipart and JSON)
            default_methods = list(app.config.get('DEFAULT_ALGORITHMS', ['must_have', 'sbert', 'bm25', 'cosine', 'ner']))

            files = []
            successful_files = []
            failed_files = []
            resume_texts = []

            if request.is_json:
                payload = request.get_json(silent=True) or {}
                position = payload.get('position', 'general')

                methods = payload.get('methods') or default_methods
                if isinstance(methods, str):
                    methods = [methods]

                options = payload.get('options') if isinstance(payload.get('options'), dict) else {}
                metadata = payload.get('metadata') if isinstance(payload.get('metadata'), dict) else {}

                job_description = str(payload.get('jobDescription') or '').strip()
                if not job_description and isinstance(payload.get('job'), dict):
                    job_description = _job_json_to_text(payload.get('job'))
                requirements_text = _requirements_to_text(payload.get('requirements'))
                if requirements_text:
                    job_description = "\n\n".join([requirements_text, job_description]).strip()

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
                job_description = request.form.get('jobDescription', '').strip()
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
                resume_texts, job_description, methods, position
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

    def _generate_explanation(combined_result: dict, job_description: str, position: str) -> str:
        """Generate human-readable explanation for ranking"""
        try:
            score = float(combined_result['combined_score'])
            algorithm_scores = combined_result['algorithm_scores']
            
            if score >= 0.8:
                rating = "Excellent match"
            elif score >= 0.6:
                rating = "Good match"
            elif score >= 0.4:
                rating = "Fair match"
            else:
                rating = "Poor match"
            
            # Find best performing algorithm
            best_alg = max(algorithm_scores.keys(), 
                          key=lambda k: float(algorithm_scores[k]['score']), 
                          default='unknown')
            best_score = float(algorithm_scores.get(best_alg, {}).get('score', 0))
            
            explanation = f"{rating} for {position} position (Overall: {score:.1%}). "
            explanation += f"Strongest performance in {best_alg.upper()} analysis ({best_score:.1%}). "

            # Add algorithm-specific insights
            if 'ner' in algorithm_scores:
                ner_details = algorithm_scores['ner'].get('details', {})
                skill_categories = len(ner_details.get('extracted_skills', {}))
                if skill_categories > 0:
                    explanation += f"Identified skills across {skill_categories} categories. "
            
            if 'cosine' in algorithm_scores:
                cosine_details = algorithm_scores['cosine'].get('details', {})
                matching_terms = len(cosine_details.get('top_matching_terms', []))
                if matching_terms > 0:
                    explanation += f"Found {matching_terms} key matching terms. "
            
            return explanation
            
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
