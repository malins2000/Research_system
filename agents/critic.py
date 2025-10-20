import json
import re
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

        # NEW PROMPT: Ask the model to wrap the JSON in <critic_json> tags.
        prompt = (
            f"You are a meticulous critic. Your task is to evaluate the following content based on the given criteria. "
            f"Think step-by-step. First, analyze the content. Second, create a JSON object with your feedback. "
            f"Finally, wrap this JSON object in <critic_json> tags. "
            f"Do not include any other text after the closing </critic_json> tag.\n\n"
            f"The JSON must have three keys: 'approved' (boolean), 'feedback' (string), and 'rating' (a float from 0.0 to 5.0).\n\n"
            f"**Evaluation Criteria:**\n{evaluation_criteria}\n\n"
            f"**Content to Review:**\n{content_str}\n\n"
            f"YOUR RESPONSE:"
        )

        # Query the LLM
        response_str = self.llm_client.query(prompt)
        json_str = "" # Initialize for use in the except block

        try:
            # NEW PARSING LOGIC: Use regex to find the content inside our tags.
            match = re.search(r"<critic_json>(.*?)</critic_json>", response_str, re.DOTALL)

            if not match:
                print("Critic Agent: Error - Could not find <critic_json> tags in the LLM response.")
                print(f"Raw LLM Response:\n{response_str}")
                return {"approved": False, "feedback": "Failed to find <critic_json> tags in response.", "rating": 0.0}

            # Extract the clean JSON string
            json_str = match.group(1).strip()

            # Parse the extracted JSON
            evaluation_result = json.loads(json_str)
            print("Critic Agent: Successfully evaluated content.")
            return evaluation_result
        except json.JSONDecodeError as e:
            print(f"Critic Agent: Error decoding JSON from the extracted block: {e}")
            print(f"Extracted JSON String that failed parsing:\n{json_str}")
            return {"approved": False, "feedback": "Failed to parse LLM response.", "rating": 0.0}
        except Exception as e:
            print(f"Critic Agent: An unexpected error occurred: {e}")
            return {"approved": False, "feedback": "An unexpected error occurred during evaluation.", "rating": 0.0}
