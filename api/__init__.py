"""API package for CV–JD scoring service."""

from .middleware import setup_middleware
from .error_handlers import register_error_handlers

__all__ = [
    'setup_middleware',
    'register_error_handlers',
]
