from enum import Enum

from toolbox.data.exporters.csv_exporter import CSVExporter
from toolbox.data.exporters.dataframe_exporter import DataFrameExporter
from toolbox.data.exporters.safa_exporter import SafaExporter


class SupportedDatasetExporter(Enum):
    SAFA = SafaExporter
    CSV = CSVExporter
    DF = DataFrameExporter
