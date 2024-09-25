from toolbox.data.keys.structure_keys import StructuredKeys
from toolbox.data.readers.entity.formats.csv_entity_format import CsvEntityFormat
from toolbox.data.readers.entity.formats.folder_entity_format import FolderEntityFormat
from toolbox.data.readers.entity.formats.json_entity_format import JsonEntityFormat
from toolbox.data.readers.entity.formats.xml_entity_format import XmlEntityFormat
from toolbox.data.readers.entity.supported_entity_formats import SupportedEntityFormats
from toolbox_test.base.paths.format_paths import toolbox_TEST_FORMAT_CSV_PATH, toolbox_TEST_FORMAT_FOLDER_PATH, \
    toolbox_TEST_FORMAT_JSON_PATH, toolbox_TEST_FORMAT_XML_PATH

from toolbox_test.base.tests.base_test import BaseTest


class TestSupportedEntityFormat(BaseTest):
    """
    Test entity format's ability to retrieve correct format.
    """
    DATUM = {
        CsvEntityFormat: toolbox_TEST_FORMAT_CSV_PATH,
        FolderEntityFormat: toolbox_TEST_FORMAT_FOLDER_PATH,
        XmlEntityFormat: toolbox_TEST_FORMAT_XML_PATH,
        JsonEntityFormat: toolbox_TEST_FORMAT_JSON_PATH
    }

    def test_get_entity_parser(self):
        """
        Tests ability to retrieve parser using implied methods. Includes testing all
        types of extensions.
        """
        for entity_format, data_path in self.DATUM.items():
            retrieved_parser = SupportedEntityFormats.get_parser(data_path)
            self.assertEqual(retrieved_parser, entity_format)

    def test_get_entity_parser_explicit(self):
        """
        Tests ability to retrieve parser when it is explicitly set.
        """
        for entity_format_name, entity_format in SupportedEntityFormats.FORMATS.items():
            data_path = self.DATUM[entity_format]
            retrieved_parser = SupportedEntityFormats.get_parser(data_path, {StructuredKeys.PARSER: entity_format_name})
            self.assertEqual(retrieved_parser, entity_format)
