from typing import TypedDict, List, Dict
from typing_extensions import NotRequired


class Usage(TypedDict):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class Function(TypedDict):
    arguments: dict | list
    description: str
    name: str


class ToolCall(TypedDict):
    function: Function
    id: str
    type: str


class Message(TypedDict):
    content: NotRequired[str]
    role: str
    tool_calls: NotRequired[List[ToolCall]]
    id: NotRequired[str]
    type: NotRequired[str]


class Choices(TypedDict):
    finish_reason: str
    index: int
    logprobs: Dict
    message: Message
    created: int
    id: str
    model: str
    system_fingerprint: str
    usage: Usage


class HuggingFaceResponse(TypedDict):
    choices: List[Choices]
