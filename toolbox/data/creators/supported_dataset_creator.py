from enum import Enum

from toolbox.data.creators.mlm_pre_train_dataset_creator import MLMPreTrainDatasetCreator
from toolbox.data.creators.multi_trace_dataset_creator import MultiTraceDatasetCreator
from toolbox.data.creators.prompt_dataset_creator import PromptDatasetCreator
from toolbox.data.creators.split_dataset_creator import SplitDatasetCreator
from toolbox.data.creators.trace_dataset_creator import TraceDatasetCreator


class SupportedDatasetCreator(Enum):
    MLM_PRE_TRAIN = MLMPreTrainDatasetCreator
    SPLIT = SplitDatasetCreator
    TRACE = TraceDatasetCreator
    MULTI = MultiTraceDatasetCreator
    PROMPT = PromptDatasetCreator
