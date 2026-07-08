"""Matching package for CV–JD scoring.

Exposes AlgorithmManager and VectorDB.
"""

from modules.matching.manager import AlgorithmManager
from modules.matching.vector_db import VectorDB

__all__ = [
    "AlgorithmManager",
    "VectorDB",
]
