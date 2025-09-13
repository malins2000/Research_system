from typing import Any
from agents.base_agent import BaseAgent

class SummaryAgent(BaseAgent):
    """
    An agent responsible for creating a final summary of the research.
    """

    def __init__(self, llm_client: Any):
        """
        Initializes the SummaryAgent with an LLM client.

        Args:
            llm_client: An instance of an LLM client.
        """
        super().__init__(llm_client)

    def execute(self, full_text: str) -> str:
        """
        Generates a concise, executive-level summary of the provided text.

        Args:
            full_text: The entire concatenated text of the research.

        Returns:
            A string containing the summary.
        """
        print("Summary Agent: Generating final summary...")

        # Formulate the prompt for the LLM
        prompt = (
            "You are a professional technical writer. Your task is to generate a concise, "
            "executive-level summary of the following research text. Focus on the key findings, "
            "conclusions, and major supporting points. The summary should be clear, "
            "to the point, and suitable for a busy executive.\n\n"
            "**Full Research Text:**\n"
            f"{full_text}"
        )

        # Query the LLM
        summary = self.llm_client.query(prompt)

        print("Summary Agent: Successfully generated summary.")
        return summary
