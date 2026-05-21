# algorithms/manager/algorithm_manager.py
from typing import List, Dict, Any
import concurrent.futures
import logging

# Deep learning analyzers
try:
    from algorithms.deep_learning.bert_analyzer import BERTAnalyzer
except ImportError:
    BERTAnalyzer = None

try:
    from algorithms.deep_learning.distilbert_analyzer import DistilBERTAnalyzer
except ImportError:
    DistilBERTAnalyzer = None  # Fallback handled below

try:
    from algorithms.deep_learning.sbert_analyzer import SBERTAnalyzer
except ImportError:
    SBERTAnalyzer = None

# Similarity analyzers
from algorithms.similarity.cosine_similarity import CosineSimilarityAnalyzer
try:
    # Ensure this import points to a strict-skill Jaccard, not cosine
    from algorithms.similarity.jaccard_similarity import JaccardSimilarityAnalyzer
except ImportError:
    JaccardSimilarityAnalyzer = None
try:
    from algorithms.similarity.bm25_analyzer import BM25Analyzer
except ImportError:
    BM25Analyzer = None

from algorithms.similarity.ner_analyzer import NERAnalyzer
from algorithms.similarity.must_have_analyzer import MustHaveAnalyzer

logger = logging.getLogger(__name__)


class AlgorithmManager:
    """Manages and orchestrates multiple ranking algorithms with distinct behaviors"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.algorithms: Dict[str, Any] = {}

        # Registry with distinct implementations/paths
        registry: Dict[str, Any] = {}

        # Semantic models
        if BERTAnalyzer:
            registry['bert'] = BERTAnalyzer
        if DistilBERTAnalyzer:
            registry['distilbert'] = DistilBERTAnalyzer
        elif BERTAnalyzer:
            registry['distilbert'] = BERTAnalyzer  # Will pass distilbert config
        if SBERTAnalyzer:
            registry['sbert'] = SBERTAnalyzer
        elif BERTAnalyzer:
            registry['sbert'] = BERTAnalyzer  # Will pass sbert/miniLM config

        # Similarity models
        registry['cosine'] = CosineSimilarityAnalyzer
        if BM25Analyzer:
            registry['bm25'] = BM25Analyzer
        if JaccardSimilarityAnalyzer:
            registry['jaccard'] = JaccardSimilarityAnalyzer
        else:
            registry['jaccard'] = CosineSimilarityAnalyzer  # Warn at init time

        # Entity extraction
        registry['must_have'] = MustHaveAnalyzer
        registry['ner'] = NERAnalyzer

        self.algorithm_registry = registry
        self.max_workers = self.config.get('max_workers', 4)

    def initialize_algorithms(self, algorithm_names: List[str]) -> None:
        """Initialize specified algorithms with distinct configurations"""
        for name in algorithm_names:
            if name in self.algorithm_registry and name not in self.algorithms:
                try:
                    logger.info(f"Initializing algorithm: {name}")
                    algorithm_class = self.algorithm_registry[name]
                    algorithm_config = dict(self.config.get(name, {}))  # copy

                    # Distinct configurations to ensure different behavior
                    if name == 'bert':
                        algorithm_config.setdefault('model_name', 'bert-base-uncased')
                        algorithm_config.setdefault('pooling', 'mean')
                        algorithm_config.setdefault('calibration', 'linear')
                        algorithm_config.setdefault('section_aware', True)

                    elif name == 'distilbert':
                        algorithm_config.setdefault('model_name', 'distilbert-base-uncased')
                        algorithm_config.setdefault('pooling', 'mean')
                        algorithm_config.setdefault('calibration', 'aggressive')
                        algorithm_config.setdefault('section_aware', True)

                    elif name == 'sbert':
                        algorithm_config.setdefault('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
                        algorithm_config.setdefault('sbert_mode', True)
                        algorithm_config.setdefault('pooling', 'sentence')
                        algorithm_config.setdefault('section_aware', True)

                    elif name == 'cosine':
                        algorithm_config.setdefault('ngrams', (1, 2))
                        algorithm_config.setdefault('stopwords', 'english')
                        algorithm_config.setdefault('boost_terms', True)
                        algorithm_config.setdefault('section_aware', True)

                    elif name == 'jaccard':
                        algorithm_config.setdefault('legacy_alias_for', 'bm25')
                        algorithm_config.setdefault('algorithm_label', 'BM25')

                    elif name == 'bm25':
                        algorithm_config.setdefault('mode', 'strict_skill_coverage')
                        algorithm_config.setdefault('must_have_weight', 0.60)
                        algorithm_config.setdefault('should_have_weight', 0.30)
                        algorithm_config.setdefault('bonus_weight', 0.10)
                        algorithm_config.setdefault('penalty_per_missing_must', 0.15)

                    elif name == 'ner':
                        algorithm_config.setdefault('entities', ['SKILL', 'TITLE', 'ORG', 'CERT', 'EDU', 'EXP_YEARS'])
                        algorithm_config.setdefault('return_must_have_ok', True)

                    elif name == 'must_have':
                        algorithm_config.setdefault('strict', True)

                    if name == 'jaccard' and self.algorithm_registry['jaccard'] is CosineSimilarityAnalyzer:
                        logger.warning("JaccardAnalyzer not found; falling back to Cosine. Add algorithms/similarity/jaccard_similarity.py for strict coverage scoring.")

                    self.algorithms[name] = algorithm_class(algorithm_config)
                    if hasattr(self.algorithms[name], 'load_model'):
                        self.algorithms[name].load_model()

                    logger.info(f"Algorithm {name} initialized successfully")

                except Exception as e:
                    logger.error(f"Failed to initialize algorithm {name}: {e}")
                    continue

    def process_resumes_parallel(self, resume_texts: List[str],
                                 job_description: str, algorithm_names: List[str],
                                 position: str = None) -> Dict[str, Any]:
        """Process resumes using multiple algorithms in parallel"""
        self.initialize_algorithms(algorithm_names)

        available_algorithms = [name for name in algorithm_names if name in self.algorithms]
        if not available_algorithms:
            raise Exception("No algorithms available for processing")

        logger.info(f"Processing {len(resume_texts)} resumes with {len(available_algorithms)} algorithms")

        results: Dict[str, Any] = {
            'metadata': {
                'total_resumes': len(resume_texts),
                'algorithms_used': available_algorithms,
                'job_position': position,
                'processing_timestamp': None
            },
            'individual_scores': {},
            'combined_results': [],
            'algorithm_performance': {}
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_algorithm = {}
            for alg_name in available_algorithms:
                alg = self.algorithms[alg_name]
                if hasattr(alg, 'process_batch'):
                    future = executor.submit(alg.process_batch, resume_texts, job_description, position)
                else:
                    future = executor.submit(
                        lambda: [alg.process_single(rt, job_description, position) for rt in resume_texts]
                    )
                future_to_algorithm[future] = alg_name

            for future in concurrent.futures.as_completed(future_to_algorithm):
                alg_name = future_to_algorithm[future]
                try:
                    alg_results = future.result()
                    results['individual_scores'][alg_name] = alg_results
                    if hasattr(self.algorithms[alg_name], 'get_performance_metrics'):
                        results['algorithm_performance'][alg_name] = self.algorithms[alg_name].get_performance_metrics()
                    else:
                        results['algorithm_performance'][alg_name] = {}
                    logger.info(f"Completed processing with {alg_name}")
                except Exception as e:
                    logger.error(f"Algorithm {alg_name} failed: {e}")
                    results['individual_scores'][alg_name] = []
                    results['algorithm_performance'][alg_name] = {'error': str(e)}

        results['combined_results'] = self._combine_algorithm_results(
            results['individual_scores'], len(resume_texts)
        )

        return results

    def _resolve_weights(self) -> Dict[str, float]:
        """Resolve fusion weights for active scorers, then normalize."""
        weights = {
            'bert': 0.25,
            'distilbert': 0.10,
            'sbert': 0.25,
            'cosine': 0.20,
            'bm25': 0.10,
            'jaccard': 0.05,
            'must_have': 0.15,
            'ner': 0.10,
        }
        cfg_weights = self.config.get('weights', {})
        weights.update(cfg_weights)

        # Normalize positive weights
        s = sum(v for v in weights.values() if v > 0)
        if s > 0:
            for k, v in list(weights.items()):
                if v > 0:
                    weights[k] = v / s
        return weights

    def _combine_algorithm_results(self, individual_scores: Dict[str, List],
                                   total_resumes: int) -> List[Dict[str, Any]]:
        """Combine scores from multiple algorithms with distinct, non-overlapping weights"""
        combined_results: List[Dict[str, Any]] = []

        weights = self._resolve_weights()

        for resume_idx in range(total_resumes):
            resume_result = {
                'resume_index': resume_idx,
                'algorithm_scores': {},
                'combined_score': 0.0,
                'weighted_score': 0.0,
                'rank': 0,
                'details': {'contributions': []},
                'errors': []
            }

            total_weight = 0.0
            score_sum = 0.0

            for alg_name, alg_results in individual_scores.items():
                if resume_idx < len(alg_results):
                    result = alg_results[resume_idx] or {}
                    if 'error' not in result:
                        # Extract a numeric score, fallback to common keys
                        raw = result.get('score', None)
                        if raw is None:
                            details = result.get('details', {})
                            raw = details.get('probability') or details.get('confidence') or 0.0
                        try:
                            score = float(raw)
                        except Exception:
                            logger.warning(f"{alg_name} returned non-numeric score for resume {resume_idx}. Defaulting to 0.")
                            score = 0.0
                        score = max(0.0, min(1.0, score))
                        weight = float(weights.get(alg_name, 0.0))

                        resume_result['algorithm_scores'][alg_name] = {
                            'score': score,
                            'weight': weight,
                            'details': result.get('details', {})
                        }
                        resume_result['details']['contributions'].append(
                            {'alg': alg_name, 'score': score, 'weight': weight}
                        )

                        score_sum += score * weight
                        total_weight += weight
                    else:
                        resume_result['errors'].append({'algorithm': alg_name, 'error': result['error']})

            if total_weight > 0:
                resume_result['weighted_score'] = score_sum / total_weight
                resume_result['combined_score'] = resume_result['weighted_score']

            # Apply must-have penalty if any analyzer indicates failure.
            for alg_scores in resume_result['algorithm_scores'].values():
                details = alg_scores.get('details', {})
                must_ok = details.get('must_have_ok', True)
                missing_must = details.get('missing_must_count', 0)
                if not must_ok or missing_must:
                    resume_result['combined_score'] *= 0.5
                    break

            combined_results.append(resume_result)

        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        for idx, result in enumerate(combined_results):
            result['rank'] = idx + 1

        return combined_results

    def get_algorithm_status(self) -> Dict[str, Any]:
        status = {
            'available_algorithms': list(self.algorithm_registry.keys()),
            'loaded_algorithms': list(self.algorithms.keys()),
            'algorithm_details': {}
        }
        for name, algorithm in self.algorithms.items():
            if hasattr(algorithm, 'get_performance_metrics'):
                status['algorithm_details'][name] = algorithm.get_performance_metrics()
            else:
                status['algorithm_details'][name] = {}
        return status

    def cleanup(self) -> None:
        for algorithm in self.algorithms.values():
            try:
                if hasattr(algorithm, 'cleanup'):
                    algorithm.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up algorithm: {e}")
        self.algorithms.clear()
        logger.info("Algorithm manager cleaned up")
