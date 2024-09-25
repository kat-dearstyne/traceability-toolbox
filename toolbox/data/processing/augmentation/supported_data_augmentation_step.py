from enum import Enum

from toolbox.data.processing.augmentation.resample_step import ResampleStep
from toolbox.data.processing.augmentation.simple_word_replacement_step import SimpleWordReplacementStep
from toolbox.data.processing.augmentation.source_target_swap_step import SourceTargetSwapStep


class SupportedAugmentationStep(Enum):
    SIMPLE_WORD_REPLACEMENT = SimpleWordReplacementStep
    RESAMPLE = ResampleStep
    SOURCE_TARGET_SWAP = SourceTargetSwapStep
