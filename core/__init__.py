"""Core processing package for CV–JD scoring."""

from .algorithm_manager import AlgorithmManager
from .score_combiner import ScoreCombiner

__all__ = [
    'AlgorithmManager',
    'ScoreCombiner',
]
