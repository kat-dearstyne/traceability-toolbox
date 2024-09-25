from typing import List

from toolbox.data.objects.trace import Trace
from toolbox.traceability.ranking.trace_selectors.i_selection_method import iSelector
from toolbox.util.ranking_util import RankingUtil


class SelectByThreshold(iSelector):

    @staticmethod
    def select(candidate_entries: List[Trace], threshold: float, **kwargs) -> List[Trace]:
        """
        Filters the candidate links based on score threshold
        :param candidate_entries: Candidate trace entries
        :param threshold: The threshold to filter by
        :return: filtered list of entries
        """
        return RankingUtil.select_traces_by_threshold(candidate_entries, threshold)
