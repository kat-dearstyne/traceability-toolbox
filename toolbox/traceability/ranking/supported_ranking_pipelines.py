from toolbox.traceability.ranking.clustering_ranking_pipeline import ClusteringRankingPipeline
from toolbox.traceability.ranking.embedding_ranking_pipeline import EmbeddingRankingPipeline
from toolbox.traceability.ranking.llm_ranking_pipeline import LLMRankingPipeline
from toolbox.traceability.ranking.search_pipeline import SearchPipeline
from toolbox.util.supported_enum import SupportedEnum


class SupportedRankingPipelines(SupportedEnum):
    """
    Enumerates the methods of ranking artifacts to their parents.
    """
    LLM = LLMRankingPipeline
    EMBEDDING = EmbeddingRankingPipeline
    SEARCH = SearchPipeline
    CLUSTERING = ClusteringRankingPipeline
