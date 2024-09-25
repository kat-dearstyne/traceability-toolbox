from toolbox.llm.args.anthropic_args import AnthropicArgs
from toolbox.llm.args.open_ai_args import OpenAIArgs
from toolbox.util.supported_enum import SupportedEnum


class SupportedLLMArgs(SupportedEnum):
    """
    Enumerates supported language models arguments
    """
    OPENAI = OpenAIArgs
    ANTHROPIC = AnthropicArgs
