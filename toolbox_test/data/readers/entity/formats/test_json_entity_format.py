from typing import Dict, List, Type

from toolbox.data.readers.entity.formats.abstract_entity_format import AbstractEntityFormat
from toolbox.data.readers.entity.formats.json_entity_format import JsonEntityFormat
from toolbox_test.base.paths.format_paths import toolbox_TEST_FORMAT_JSON_PATH
from toolbox_test.data.readers.entity.formats.abstract_entity_format_test import AbstractEntityFormatTest


class TestJsonEntityFormat(AbstractEntityFormatTest):
    """
    Tests ability to parser json file as entities.
    """

    def test_extensions(self):
        self.verify_extensions()

    def test_parser(self):
        self.verify_parser()

    @property
    def entity_format(self) -> Type[AbstractEntityFormat]:
        return JsonEntityFormat

    @property
    def data_path(self) -> str:
        return toolbox_TEST_FORMAT_JSON_PATH

    @staticmethod
    def get_entities() -> List[Dict]:
        return [{"name": "1"}, {"name": "2"}]
