from toolbox.graph.io.graph_args import GraphArgs
from toolbox.graph.io.graph_state import GraphState
from toolbox.graph.llm_tools.vector_store_manager import VectorStoreManager
from toolbox.graph.nodes.abstract_node import AbstractNode


class RetrieveNode(AbstractNode):

    def __init__(self, graph_args: GraphArgs):
        """
        Responsible for retrieving context documents from vectorstore.
        :param graph_args: Arguments to the graph.
        """
        super().__init__(graph_args=graph_args)
        self.vector_store_manager = VectorStoreManager.from_args(self.graph_args)

    def perform_action(self, state: GraphState, run_async: bool = False):
        """
        Retrieve documents.
        :param state: The current state of the chat.
        :param run_async: Whether to run graph asynchronously.
        """
        documents = self.vector_store_manager.search(state["retrieval_query"])
        state["documents"].update(documents)
        return state
