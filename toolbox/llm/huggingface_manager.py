from typing import Dict, List, Optional, Set, Union

from huggingface_hub.inference._client import InferenceClient
from tqdm import tqdm

from toolbox.constants import environment_constants
from toolbox.constants.env_var_name_constants import HG_API_KEY_PARAM
from toolbox.constants.environment_constants import HUGGING_FACE_KEY
from toolbox.constants.hugging_face_constants import HG_MAX_TRIES
from toolbox.constants.symbol_constants import EMPTY_STRING
from toolbox.infra.t_logging.logger_manager import logger
from toolbox.llm.abstract_llm_manager import AbstractLLMManager
from toolbox.llm.args.hg_args import HGParams, HGArgs
from toolbox.llm.huggingface_response import HuggingFaceResponse, ToolCall
from toolbox.llm.llm_responses import GenerationResponse, \
    SupportedLLMResponses
from toolbox.llm.llm_task import LLMCompletionType
from toolbox.llm.prompts.llm_prompt_build_args import LLMPromptBuildArgs
from toolbox.util.attr_dict import AttrDict
from toolbox.util.pythonisms_util import default_mutable


def response_with_defaults(**params) -> HuggingFaceResponse:
    """
    Creates an HG response using default values unless provided
    :param params: Params to use in place of defaults
    :return: The HG response with defaults filled in
    """
    defaults = {str: EMPTY_STRING, bool: False}
    all_params = {attr: params[attr] if attr in params else defaults[type_]
                  for attr, type_ in HuggingFaceResponse.__annotations__.items()}
    return HuggingFaceResponse(**all_params)


class HuggingFaceManager(AbstractLLMManager[HuggingFaceResponse]):
    """
    Defines AI interface for HG API.
    """

    NOT_IMPLEMENTED_ERROR = "This has not been implemented yet"
    prompt_args = LLMPromptBuildArgs(prompt_prefix=EMPTY_STRING, prompt_suffix=EMPTY_STRING, completion_prefix=" ",
                                     completion_suffix="###", build_system_prompts=True)

    def __init__(self, llm_args: HGArgs):
        """
        Initializes with args used for the requests to HG model
        :param llm_args: args used for the requests to HG model
        """
        assert isinstance(llm_args, HGArgs), "Must use HG args with HG manager"
        super().__init__(llm_args=llm_args, prompt_args=self.prompt_args)

    @default_mutable()
    def make_completion_request_impl(self, raise_exception: bool = True, original_responses: List = None,
                                     retries: Set[int] = None, **params) -> List:
        """
        Makes a completion request to HG api.
        :param raise_exception: If True, raises an exception if the request has failed.
        :param original_responses: List of the original responses from the model if retrying.
        :param retries: Set of indices of responses that need retried because they failed the first time.
        :param params: Named parameters to HG API.
        :return: HG's response to completion request.
        """
        assert HGParams.PROMPTS in params, f"Expected {params} to include {HGParams.PROMPTS}"
        prompts = params.pop(HGParams.PROMPTS)

        if HGParams.MODEL not in params:
            params[HGParams.MODEL] = self.llm_args.model
        else:
            assert params[HGParams.MODEL] == self.llm_args.model, \
                f"Expected model to be {self.llm_args.model} but got {params[HGParams.MODEL]}"

        client = get_client()
        results = []
        for i, prompt in enumerate(tqdm(prompts, f"Starting Hugging Face inference ({len(prompts)}): {params['model']}")):
            res, success = None, False
            if i not in retries and len(original_responses) > i:
                res = original_responses[i]
                success = True

            for attempt in range(HG_MAX_TRIES):
                if success:
                    break

                try:
                    res = client.chat.completions.create(
                        messages=prompt,
                        stream=False,
                        **params
                    )
                    success = True
                except Exception as e:
                    res = e
                    logger.warning("Response failed: ", e)
                    if attempt < HG_MAX_TRIES - 1:
                        logger.info("Retrying...")

            if not success and raise_exception:
                raise res

            results.append(res)

        return results

    def translate_to_response(self, task: LLMCompletionType, res: List[HuggingFaceResponse],
                              **params) -> Optional[SupportedLLMResponses]:
        """
        Translates the LLM library response to task specific response.
        :param task: The task to translate to.
        :param res: The response from the LLM library.
        :param params: Any additional parameters to customize translation.
        :return: A task-specific response.
        """
        texts = [self._extract_response(r) for r in res]
        return GenerationResponse(texts)

    @classmethod
    def format_response(cls, response_text: str = None, exception: Exception = None) -> AttrDict:
        """
        Formats the text, exception and any other information in the same way as all other responses from OpenAI.
        :param response_text: The models generated text.
        :param exception: Any exception raised during the generation.
        :return: The formatted response
        """
        res = response_text if response_text else exception
        response = AttrDict({"message": AttrDict({"content": res})})
        return response

    @staticmethod
    def _extract_response(res: HuggingFaceResponse) -> Union[str, Dict, Exception]:
        """
        Gets the LLM response or the error msg if exception occurred.
        :param res: The response.
        :return: The LLM response or the error msg if exception occurred
        """
        messages = [choice["message"] for choice in res["choices"]]
        tool_use = [HuggingFaceManager._extract_tool_use(m["tool_calls"]) for m in messages if m.get("tool_calls", None)]
        texts = [m for m in messages if m.get("content", None)]
        assert tool_use or texts, f"Received a badly formatted response {messages}"
        content = texts[0] if len(tool_use) < 1 else tool_use[0]
        return content

    @staticmethod
    def _extract_tool_use(tool_calls: List[ToolCall]) -> Dict:
        """
        Converts tool use response to expected format.
        :param tool_calls: The model's tool call response.
        :return: Dict containing function name and arguments.
        """
        tool_call = tool_calls[0]
        function = tool_call["function"]
        arguments = {v: v for v in function["arguments"]} if isinstance(function["arguments"], list) else function["arguments"]
        return {"name": function["name"], **arguments}


def get_client():
    """
    Returns the current HG client.
    :return:  Returns the singleton HG client.
    """
    if environment_constants.IS_TEST:
        raise NotImplementedError("No Test Model Right now")
    else:
        assert HUGGING_FACE_KEY, f"Must supply value for {HG_API_KEY_PARAM} "
        return InferenceClient(api_key=HUGGING_FACE_KEY)
