from typing import Dict

from toolbox.constants.anthropic_constants import ANTHROPIC_MODEL_DEFAULT
from toolbox.constants.model_constants import PREDICT_TASK, TRAIN_TASK
from toolbox.constants.open_ai_constants import MAX_TOKENS_DEFAULT
from toolbox.llm.args.abstract_llm_args import AbstractLLMArgs
from toolbox.util.dataclass_util import DataclassUtil
from toolbox.util.dict_util import DictUtil


class HGParams:
    """
    Contains allowed parameters to HG API.
    """
    PROMPTS = "prompt"
    MESSAGES = "messages"
    MODEL = "model"  # claude-v1, claude-v1.2, claude-v1.3, claude-instant-v1, claude-instant-v1.0
    MAX_TOKENS = "max_tokens"
    FREQUENCY_PENALTY = "frequency_penalty"
    LOGPROBS = "logprobs"
    PRESENCE_PENALTY = "presence_penalty"
    RESPONSE_FORMAT = "response_format"
    SEED = "seed"
    STOP = "stop"  # List of strings that will stop prediction when encountered.
    STREAM = "stream"
    STREAM_OPTIONS = "stream_options"  # NOT SUPPORTED.
    TEMPERATURE = "temperature"
    TOOL_CHOICE = "tool_choice"
    TOOL_PROMPT = "tool_prompt"
    TOOLS = "tools"
    TOP_LOGPROBS = "top_logprobs"
    TOP_P = "top_p"  # Nucleus sampling selects top probability tokens, alter temperature or top_p.


class HGArgs(AbstractLLMArgs):
    """
    Defines allowable arguments to HG API.
    """

    max_tokens: int = MAX_TOKENS_DEFAULT
    _EXPECTED_TASK_PARAMS = {TRAIN_TASK: [],
                             PREDICT_TASK: [HGParams.MODEL, HGParams.TEMPERATURE, HGParams.MAX_TOKENS]}

    def __init__(self, **kwargs):
        """
        Sets all necessary args for Anthropic
        :param kwargs: Contains all necessary arg name to value mappings
        """
        super_args = DataclassUtil.set_unique_args(self, AbstractLLMArgs, **kwargs)
        DictUtil.update_kwarg_values(super_args, replace_existing=False)
        super().__init__(expected_task_params=self._EXPECTED_TASK_PARAMS, llm_params=HGParams, **super_args)

    def _add_library_params(self, task: str, params: Dict, instructions: Dict) -> Dict:
        """
        Allows the usage of custom params defined in instructions. Currently unused for Anthropic
        :param task: The task being performed.
        :param params: The parameters current being constructed.
        :param instructions: Any custom instruction flags.
        :return: Parameters with customizations added.
        """
        return params

    def set_max_tokens(self, max_tokens: int) -> None:
        """
        Sets the max tokens of anthropic params.
        :param max_tokens: The max tokens to set it to.
        :return: None
        """
        self.max_tokens = max_tokens

    def get_max_tokens(self) -> int:
        """
        :return: Returns the max tokens of args.
        """
        return self.max_tokens
