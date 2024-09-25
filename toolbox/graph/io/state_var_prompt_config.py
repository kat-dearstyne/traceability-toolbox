from dataclasses import dataclass
from typing import Type

from toolbox.graph.io.state_to_prompt_converters import iConverter


@dataclass
class StateVarPromptConfig:
    title: str = None
    include_in_message_prompt: bool = True
    prompt_converter: Type[iConverter] = None
