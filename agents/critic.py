import json
from typing import Any, Union, Dict

from agents.base_agent import BaseAgent

class CriticAgent(BaseAgent):
    """
    An agent responsible for evaluating content and providing feedback.
    """

    def __init__(self, llm_client: Any):
        """
        Initializes the CriticAgent with an LLM client.

        Args:
            llm_client: An instance of an LLM client.
        """
        super().__init__(llm_client)

    def execute(self, content_to_review: Union[Dict, str], evaluation_criteria: str) -> Dict[str, Any]:
        """
        Evaluates a piece of content based on specific criteria.

        Args:
            content_to_review: The content to be evaluated (e.g., a plan dict or generated text).
            evaluation_criteria: A string describing what to check for (e.g., "Check for logical consistency.").

        Returns:
            A dictionary containing the evaluation results (e.g., {'approved': bool, 'feedback': str, 'rating': float}).
        """
        print("Critic Agent: Evaluating content...")

        # Ensure the content is in a string format for the prompt
        if isinstance(content_to_review, dict):
            content_str = json.dumps(content_to_review, indent=2)
        else:
            content_str = content_to_review

        # Formulate the prompt for the LLM
        prompt = (
            f"You are a meticulous critic. Your task is to evaluate the following content based on the given criteria. "
            f"Provide your feedback in a structured JSON format with three keys: 'approved' (boolean), 'feedback' (string), and 'rating' (a float from 0.0 to 5.0). "
            f"\n\n**Evaluation Criteria:**\n{evaluation_criteria}"
            f"\n\n**Content to Review:**\n{content_str}"
        )

        # Query the LLM
        response_str = self.llm_client.query(prompt)

        try:
            # Parse the JSON response
            evaluation_result = json.loads(response_str)
            print("Critic Agent: Successfully evaluated content.")
            return evaluation_result
        except json.JSONDecodeError as e:
            print(f"Critic Agent: Error decoding JSON from LLM response: {e}")
            return {"approved": False, "feedback": "Failed to parse LLM response.", "rating": 0.0}
        except Exception as e:
            print(f"Critic Agent: An unexpected error occurred: {e}")
            return {"approved": False, "feedback": "An unexpected error occurred during evaluation.", "rating": 0.0}
