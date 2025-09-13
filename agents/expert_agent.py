import json
from typing import Any, List, Dict

from agents.base_agent import BaseAgent

class ExpertAgent(BaseAgent):
    """
    A generic expert agent whose behavior is defined by a system prompt.
    """

    def __init__(self, llm_client: Any, name: str, system_prompt: str):
        """
        Initializes the ExpertAgent.

        Args:
            llm_client: An instance of an LLM client.
            name: The name of the expert (e.g., "economist").
            system_prompt: The system prompt that defines the expert's persona and knowledge.
        """
        super().__init__(llm_client)
        self.name = name
        self.system_prompt = system_prompt

    def execute(self, task_description: str, context_data: List[Dict]) -> str:
        """
        Executes a task from the expert's point of view.

        Args:
            task_description: A description of the task to be performed.
            context_data: A list of dictionaries, typically retrieved documents, to provide context.

        Returns:
            A string containing the expert's insight, analysis, or contribution.
        """
        print(f"Expert Agent '{self.name}': Executing task...")

        # Format the context data into a readable string
        context_str = "\n\n".join([f"Source: {doc.get('metadata', {}).get('source', 'N/A')}\nContent: {doc.get('content', '')}" for doc in context_data])

        # Formulate the prompt using the expert's unique system prompt
        # This is where the "persona" of the agent comes to life.
        prompt = (
            f"{self.system_prompt}\n\n"
            f"You have been assigned the following task:\n"
            f"**Task:** {task_description}\n\n"
            f"To help you, here is some relevant data that has been retrieved:\n"
            f"**Contextual Data:**\n{context_str}\n\n"
            f"Based on your specific expertise and the provided data, please provide a comprehensive analysis and your key insights. "
            f"Present your response as a clear, well-structured block of text."
        )

        # Query the LLM
        response = self.llm_client.query(prompt)

        print(f"Expert Agent '{self.name}': Successfully generated response.")
        return response
