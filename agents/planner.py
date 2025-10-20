import json
import re  # <-- ADD THIS IMPORT
from typing import Any, Optional
from agents.base_agent import BaseAgent
from tools.plan_manager import PlanManager
from utils import parse_llm_json_output

class PlannerAgent(BaseAgent):
    """
    An agent responsible for creating the initial research plan.
    """

    def __init__(self, llm_client: Any):
        """
        Initializes the PlannerAgent with an LLM client.

        Args:
            llm_client: An instance of an LLM client.
        """
        super().__init__(llm_client)

    def execute(self, user_prompt: str, plan_manager: PlanManager, previous_feedback: Optional[str] = None) -> None:
        """
        Generates or refines a research plan based on user prompt and critic feedback.

        Args:
            user_prompt: The user's research topic or question.
            plan_manager: An instance of the PlanManager to save the plan.
            previous_feedback: Optional. The feedback from the critic on the last attempt.
        """
        print("Planner Agent: Generating research plan...")

        feedback_prompt = "This is the first attempt. Please generate a plan."
        if previous_feedback:
            feedback_prompt = (
                f"Your last plan was rejected. You *must* create an improved plan that addresses this feedback:\n"
                f"**Critic's Feedback:** {previous_feedback}\n\n"
                f"Please generate a new, complete plan that incorporates these suggestions."
            )

        prompt = (
            f"You are a helpful planning agent. Your task is to generate a structured research plan. "
            f"Think step-by-step. First, analyze the user's request and any feedback. "
            f"Second, create a JSON object for the plan. "
            f"Finally, wrap this JSON object in <plan_json> tags. "
            f"Do not include any other text after the closing </plan_json> tag.\n\n"
            f"**User's Main Prompt:** '{user_prompt}'\n\n"
            f"**Instructions:**\n{feedback_prompt}\n\n"
            f"The JSON must have a key 'children', which is a list of dictionaries. "
            f"Each dictionary must have 'title', 'description', and 'experts_needed' (as a list of strings) keys.\n\n"
            f"YOUR RESPONSE:"
        )

        response_str = self.llm_client.query(prompt)
        parsed_result = parse_llm_json_output(response_str, "plan_json") # Use the correct tag

        if parsed_result:
            initial_structure = parsed_result
            plan_manager.create_plan(prompt=user_prompt, initial_structure=initial_structure)
            print("Planner Agent: Successfully parsed plan and saved.")
        else:
            # Handle parsing failure robustly
            print(f"Planner Agent: CRITICAL - Failed to parse valid JSON from LLM after all fallbacks.")
            raise ValueError("Failed to parse a valid plan from the LLM. Stopping workflow.")
