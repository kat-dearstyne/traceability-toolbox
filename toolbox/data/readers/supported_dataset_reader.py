from toolbox.data.readers.api_project_reader import ApiProjectReader
from toolbox.data.readers.artifact_project_reader import ArtifactProjectReader
from toolbox.data.readers.csv_project_reader import CsvProjectReader
from toolbox.data.readers.dataframe_project_reader import DataFrameProjectReader
from toolbox.data.readers.hub_project_reader import HubProjectReader
from toolbox.data.readers.pre_train_project_reader import PreTrainProjectReader
from toolbox.data.readers.pre_train_trace_reader import PreTrainTraceReader
from toolbox.data.readers.repository_project_reader import RepositoryProjectReader
from toolbox.data.readers.structured_project_reader import StructuredProjectReader
from toolbox.util.supported_enum import SupportedEnum


class SupportedDatasetReader(SupportedEnum):
    ARTIFACT = ArtifactProjectReader
    DATAFRAME = DataFrameProjectReader
    STRUCTURE = StructuredProjectReader
    REPOSITORY = RepositoryProjectReader
    CSV = CsvProjectReader
    SAFA = StructuredProjectReader
    API = ApiProjectReader
    MLM_PRETRAIN = PreTrainProjectReader
    PRE_TRAIN_TRACE = PreTrainTraceReader
    HUB = HubProjectReader
