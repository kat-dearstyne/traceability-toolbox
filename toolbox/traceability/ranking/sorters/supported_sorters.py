from toolbox.traceability.ranking.sorters.combined_sorter import CombinedSorter
from toolbox.traceability.ranking.sorters.transformer_sorter import TransformerSorter
from toolbox.traceability.ranking.sorters.vsm_sorter import VSMSorter
from toolbox.util.supported_enum import SupportedEnum


class SupportedSorter(SupportedEnum):
    VSM = VSMSorter
    TRANSFORMER = TransformerSorter
    COMBINED = CombinedSorter
