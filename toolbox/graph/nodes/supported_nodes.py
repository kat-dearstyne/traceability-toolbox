from types import DynamicClassAttribute

from extendableenum.extendableenum import inheritable_enum
from langchain_core.runnables.passthrough import RunnablePassthrough
from langgraph.constants import END

from toolbox.graph.nodes.explore_neighbors_node import ExploreNeighborsNode
from toolbox.graph.nodes.generate_node import GenerateNode
from toolbox.graph.nodes.retrieve_node import RetrieveNode
from toolbox.util.supported_enum import SupportedEnum


@inheritable_enum
class SupportedNodes(SupportedEnum):
    CONTINUE = RunnablePassthrough()
    END_COMMAND = END
    GENERATE = GenerateNode
    RETRIEVE = RetrieveNode
    EXPLORE_NEIGHBORS = ExploreNeighborsNode

    @DynamicClassAttribute
    def name(self) -> str:
        """Overrides getting the name of the Enum member to get version to use for langchain."""
        return self._name_.lower() if self._name_ != "END_COMMAND" else self._value_
