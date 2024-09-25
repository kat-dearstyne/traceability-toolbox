from enum import auto

from toolbox.util.supported_enum import SupportedEnum


class SupportedSeedClusteringMethods(SupportedEnum):
    """
    Enumerates all supporting clustering methods.
    """
    CENTROID_CHOOSES_ARTIFACTS = auto()
    ARTIFACTS_CHOOSE_CENTROIDS = auto()
