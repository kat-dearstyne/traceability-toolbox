from toolbox.llm.anthropic_manager import AnthropicManager
from toolbox.llm.open_ai_manager import OpenAIManager
from toolbox.util.supported_enum import SupportedEnum


class SupportedLLMManager(SupportedEnum):
    """
    Enumerates all the AI utility methods available to SAFA.
    """
    OPENAI = OpenAIManager
    ANTHROPIC = AnthropicManager
