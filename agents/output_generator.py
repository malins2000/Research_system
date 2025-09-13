from typing import Any, List
from agents.base_agent import BaseAgent

class OutputGenerationAgent(BaseAgent):
    """
    An agent that synthesizes expert insights into a coherent text section.
    """

    def __init__(self, llm_client: Any):
        """
        Initializes the OutputGenerationAgent with an LLM client.

        Args:
            llm_client: An instance of an LLM client.
        """
        super().__init__(llm_client)

    def execute(self, topic_description: str, insights: List[str]) -> str:
        """
        Synthesizes a list of expert insights into a well-written text section.

        Args:
            topic_description: The description of the current research topic.
            insights: A list of strings, where each string is an insight from an expert agent.

        Returns:
            A single string containing the synthesized, coherent text.
        """
        print("Output Generation Agent: Synthesizing expert insights...")

        insights_str = "\n\n---\n\n".join(insights)

        prompt = (
            "You are a lead author and technical writer. Your task is to synthesize the following "
            "insights from various experts into a single, coherent, and well-written section of text. "
            "The section should be written in a clear and informative style, suitable for a research report.\n\n"
            f"**Topic to Address:**\n{topic_description}\n\n"
            "**Expert Insights to Synthesize:**\n"
            f"{insights_str}\n\n"
            "Please produce the final, synthesized text. Do not include any headers or introductory "
            "phrases like 'Here is the synthesized text'. Just provide the final output."
        )

        # Query the LLM
        synthesized_text = self.llm_client.query(prompt)

        print("Output Generation Agent: Successfully synthesized text.")
        return synthesized_text
