from enum import Enum

from toolbox.llm.model_manager import ModelManager


class SupportedModelManager(Enum):
    HF = ModelManager
