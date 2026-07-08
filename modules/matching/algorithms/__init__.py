"""Algorithms package for CV–JD scoring."""

from .base_algorithm import BaseAlgorithm

try:
    from .deep_learning.bert_analyzer import BERTAnalyzer
except ImportError:
    BERTAnalyzer = None
try:
    from .deep_learning.distilbert_analyzer import DistilBERTAnalyzer
except ImportError:
    DistilBERTAnalyzer = None
try:
    from .deep_learning.sbert_analyzer import SBERTAnalyzer
except ImportError:
    SBERTAnalyzer = None

from .similarity.cosine_similarity import CosineSimilarityAnalyzer
from .similarity.jaccard_similarity import JaccardSimilarityAnalyzer
from .similarity.bm25_analyzer import BM25Analyzer
from .similarity.ner_analyzer import NERAnalyzer

__all__ = [
    'BaseAlgorithm',
    'BERTAnalyzer',
    'DistilBERTAnalyzer',
    'SBERTAnalyzer',
    'CosineSimilarityAnalyzer',
    'JaccardSimilarityAnalyzer',
    'BM25Analyzer',
    'NERAnalyzer',
]
