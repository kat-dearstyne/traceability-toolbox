from typing import Dict, List, Type

from toolbox.data.readers.entity.formats.abstract_entity_format import AbstractEntityFormat
from toolbox.data.readers.entity.formats.csv_entity_format import CsvEntityFormat
from toolbox_test.base.paths.format_paths import toolbox_TEST_FORMAT_CSV_PATH
from toolbox_test.data.readers.entity.formats.abstract_entity_format_test import AbstractEntityFormatTest
from toolbox_test.testprojects.csv_test_project import CsvTestProject


class TestCsvEntityFormat(AbstractEntityFormatTest):
    """
    Tests ability of the CSV format to read csv files into data frames.
    """
    test_project = CsvTestProject()

    def test_extensions(self):
        self.verify_extensions()

    def test_parser(self):
        self.verify_parser()

    @property
    def entity_format(self) -> Type[AbstractEntityFormat]:
        return CsvEntityFormat

    @property
    def data_path(self):
        return toolbox_TEST_FORMAT_CSV_PATH

    @classmethod
    def get_entities(cls) -> List[Dict]:
        return cls.test_project.get_csv_entries()
