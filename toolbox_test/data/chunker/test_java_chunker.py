from toolbox.data.chunkers.token_limit_chunkers.java_chunker import JavaChunker
from toolbox_test.base.paths.project_paths import toolbox_TEST_PROJECT_CHUNK_TESTJAVA_PATH
from toolbox_test.base.tests.base_test import BaseTest
from toolbox_test.data.chunker.base_code_chunker_test import BaseCodeChunkerTest


class TestJavaChunker(BaseTest):
    DATA_PATH = toolbox_TEST_PROJECT_CHUNK_TESTJAVA_PATH
    MODEL = "code-cushman-001"

    def test_chunk(self):
        BaseCodeChunkerTest.verify_chunk(self, self.get_chunker(), is_line_2_ignore=lambda line: line.startswith("import")
                                                                                                 or line.startswith("package"),
                                         line_overrides=lambda line: line == '}')

    def test_common_methods(self):
        BaseCodeChunkerTest.verify_common_methods(self, self.get_chunker())

    def get_chunker(self):
        chunker = JavaChunker(self.MODEL, 1000)
        return chunker
