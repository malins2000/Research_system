import json
import re  # <-- ADD THIS IMPORT
from typing import Any
from agents.base_agent import BaseAgent
from tools.plan_manager import PlanManager

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

    def execute(self, user_prompt: str, plan_manager: PlanManager) -> None:
        """
        Generates a research plan based on the user's prompt.

        Args:
            user_prompt: The user's research topic or question.
            plan_manager: An instance of the PlanManager to save the plan.
        """
        print("Planner Agent: Generating research plan...")

        # NEW PROMPT: Inspired by lang_graph_with_tool.py.
        # We now ask the model to wrap the plan in <plan_json> tags.
        prompt = (
            f"You are a helpful planning agent. Your task is to generate a structured research plan. "
            f"Think step-by-step. First, analyze the user's request. Second, create a "
            f"JSON object for the plan. Finally, wrap this JSON object in <plan_json> tags. "
            f"Do not include any other text after the closing </plan_json> tag.\n\n"
            f"The JSON must have a key 'children', which is a list of dictionaries. "
            f"Each dictionary must have 'title', 'description', and 'experts_needed' (as a list of strings) keys.\n\n"
            f"USER PROMPT: '{user_prompt}'\n\n"
            f"YOUR RESPONSE:"
        )

        # Query the LLM
        response_str = self.llm_client.query(prompt)

        try:
            # NEW PARSING LOGIC: Inspired by lang_graph_with_tool.py
            # Use regex to find the content *inside* our tags.
            # re.DOTALL makes '.' match newlines, which is crucial.
            match = re.search(r"<plan_json>(.*?)</plan_json>", response_str, re.DOTALL)

            if not match:
                print(f"Planner Agent: Error - Could not find <plan_json> tags in the LLM response.")
                print(f"Raw LLM Response:\n{response_str}")
                raise ValueError("No valid <plan_json> block found.")

            # Extract the clean JSON string
            json_str = match.group(1).strip()

            # Parse the extracted JSON
            initial_structure = json.loads(json_str)

            # Use the PlanManager to create and save the plan
            plan_manager.create_plan(prompt=user_prompt, initial_structure=initial_structure)

            print("Planner Agent: Successfully parsed plan and saved.")

        except json.JSONDecodeError as e:
            print(f"Planner Agent: Error decoding JSON from the extracted block: {e}")
            print(f"Extracted JSON String that failed parsing:\n{json_str}")
            raise ValueError("Failed to parse a valid plan from the LLM. Stopping workflow.")
        except Exception as e:
            print(f"Planner Agent: An unexpected error occurred: {e}")
            raise
