from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    """
    An abstract base class for all agents in the system.

    This class ensures that all agents adhere to a common interface,
    requiring them to have an initializer that accepts an LLM client
    and an `execute` method for their core logic.
    """

    def __init__(self, llm_client: Any):
        """
        Initializes the agent with an LLM client.

        Args:
            llm_client: An instance of an LLM client that will be used
                        to make calls to the language model.
        """
        self.llm_client = llm_client

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        The main entry point for the agent's core logic.

        This method must be implemented by all concrete agent classes.
        """
        pass
