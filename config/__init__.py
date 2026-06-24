"""Configuration package for Resume Ranking System"""

from .settings import Config, config_dict

__version__ = "1.0.0"
__author__ = "Resume Ranking Team"

# Export main configuration classes
__all__ = [
    'Config',
    'config_dict',
]
