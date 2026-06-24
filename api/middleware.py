import logging
import time

from flask import current_app, g, jsonify, request

logger = logging.getLogger(__name__)


def setup_middleware(app):
    """Register request logging, headers, preflight handling, and size limits."""

    @app.before_request
    def before_request():
        g.start_time = time.time()
        g.request_id = f"req_{int(time.time())}_{hash(request.path) % 10000}"
        logger.info("[%s] %s %s - Start", g.request_id, request.method, request.path)

    @app.after_request
    def after_request(response):
        if hasattr(g, 'start_time'):
            total_time = time.time() - g.start_time
            logger.info("[%s] %s - %.3fs", g.request_id, response.status_code, total_time)
            response.headers['X-Response-Time'] = f"{total_time:.3f}s"
            response.headers['X-Request-ID'] = g.request_id

        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = (
            'Content-Type, Authorization, X-Processing-Methods, X-Files-Count'
        )
        response.headers['Access-Control-Expose-Headers'] = 'X-Response-Time, X-Request-ID'
        return response

    @app.before_request
    def handle_preflight():
        if request.method == 'OPTIONS':
            return jsonify({'message': 'OK'})

    @app.before_request
    def request_size_limit():
        if request.content_length:
            max_size = current_app.config.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024)
            if request.content_length > max_size:
                return jsonify({
                    'error': 'Request too large',
                    'max_size_mb': max_size // (1024 * 1024),
                    'received_size_mb': request.content_length // (1024 * 1024),
                }), 413
