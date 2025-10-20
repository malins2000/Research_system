import json
import re
from typing import Any, Union, Dict, Optional
from utils import parse_llm_json_output

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

    def execute(self, content_to_review: Union[Dict, str], evaluation_criteria: str, previous_feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluates a piece of content based on specific criteria.

        Args:
            content_to_review: The content to be evaluated (e.g., a plan dict or generated text).
            evaluation_criteria: A string describing what to check for.
            previous_feedback: Optional. The feedback from the last failed attempt.

        Returns:
            A dictionary containing the evaluation results (e.g., {'rating': int, 'feedback': str}).
        """
        print("Critic Agent: Evaluating content...")

        if isinstance(content_to_review, dict):
            content_str = json.dumps(content_to_review, indent=2)
        else:
            content_str = content_to_review

        feedback_prompt = "This is the first review of this content."
        if previous_feedback:
            feedback_prompt = f"The previous version was rejected with this feedback: '{previous_feedback}'. Please check if the new content has addressed these issues."

        # New prompt demanding a 0-100 rating and actionable feedback
        prompt = (
            f"You are a meticulous critic. First, think step-by-step in <think> tags. "
            f"Second, provide your final evaluation as a JSON object wrapped in <critic_json> tags. "
            f"Your entire response *must* end with the closing </critic_json> tag.\n\n"
            f"The JSON object must have two keys:\n"
            f"1. 'rating' (an integer from 0 to 100).\n"
            f"2. 'feedback' (a string providing *actionable suggestions* for improvement. If the rating is low, explain what is missing. If the rating is high, confirm it's good.).\n\n"
            f"**Evaluation Criteria:**\n{evaluation_criteria}\n\n"
            f"**Previous Feedback:**\n{feedback_prompt}\n\n"
            f"**Content to Review:**\n{content_str}"
        )

        response_str = self.llm_client.query(prompt)
        parsed_result = parse_llm_json_output(response_str, "critic_json") # Use the correct tag

        if parsed_result:
            evaluation_result = parsed_result
            print("Critic Agent: Successfully evaluated content.")
            return evaluation_result
        else:
            # Handle parsing failure robustly
            print(f"Critic Agent: CRITICAL - Failed to parse valid JSON from LLM after all fallbacks.")
            return {"rating": 0, "feedback": "CRITICAL PARSING FAILURE: LLM did not return usable JSON critique."}
            # raise ValueError("Failed to parse a valid plan from the LLM. Stopping workflow.")