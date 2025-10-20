import json
import re
from typing import Any, Union, Dict, Optional

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

        json_str = ""
        try:
            # Stage 1: Try parsing JSON wrapped in XML tags
            match = re.search(r"<critic_json>(.*?)</critic_json>", response_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                evaluation_result = json.loads(json_str)
            else:
                # Stage 2 (Fallback): Use regex to find the first JSON object or array
                print("Critic Agent: Could not find <critic_json> tags. Falling back to regex search.")
                json_match = re.search(r"\{.*\}|\[.*\]", response_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0).strip()
                    evaluation_result = json.loads(json_str)
                else:
                    print(f"Critic Agent: Error - No JSON found in the LLM response.")
                    print(f"Raw LLM Response:\n{response_str}")
                    return {"rating": 0, "feedback": "No valid JSON block found in response."}

            # Ensure keys exist
            if 'rating' not in evaluation_result or 'feedback' not in evaluation_result:
                raise KeyError("Missing 'rating' or 'feedback' key in JSON.")

            print("Critic Agent: Successfully evaluated content.")
            return evaluation_result

        except Exception as e:
            print(f"Critic Agent: Error parsing response: {e}")
            print(f"Raw String that failed parsing:\n{json_str}")
            return {"rating": 0, "feedback": f"An unexpected error occurred: {e}"}
