from .jaccard_similarity import JaccardSimilarityAnalyzer


class BM25Analyzer(JaccardSimilarityAnalyzer):
    """Compatibility wrapper exposing the BM25 implementation under its real name."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.name = 'bm25'
