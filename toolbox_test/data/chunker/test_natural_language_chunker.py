from toolbox.constants.open_ai_constants import TOKENS_2_WORDS_CONVERSION

from toolbox.data.chunkers.token_limit_chunkers.natural_language_chunker import NaturalLanguageChunker
from toolbox_test.base.tests.base_test import BaseTest


class TestNaturalLanguageChunker(BaseTest):

    def test_chunk(self):
        n_times_limit = 3
        chunker = NaturalLanguageChunker("ada", max_prompt_tokens=10)
        long_text = "Hello " * (n_times_limit * round(chunker.max_prompt_tokens * TOKENS_2_WORDS_CONVERSION))
        chunks = chunker.chunk(long_text)
        self.assertEqual(len(chunks), n_times_limit)
        self.assertEqual(" ".join(chunks) + " ", long_text)
