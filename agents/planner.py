import json
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

        # Formulate a prompt to the LLM to generate a structured plan
        prompt = (
            f"Based on the following user prompt, please generate a structured research plan. "
            f"The plan should be a table of contents that breaks down the topic into logical sections. "
            f"Return the plan as a JSON object with a key 'children', which is a list of dictionaries. "
            f"Each dictionary must have 'title', 'description', and 'experts_needed' (as a list of strings) keys. "
            f"User Prompt: '{user_prompt}'"
        )

        # Query the LLM
        response_str = self.llm_client.query(prompt)

        try:
            # Parse the LLM's response
            initial_structure = json.loads(response_str)

            # Use the PlanManager to create and save the plan
            plan_manager.create_plan(prompt=user_prompt, initial_structure=initial_structure)

            print("Planner Agent: Successfully created and saved the research plan.")

        except json.JSONDecodeError as e:
            print(f"Planner Agent: Error decoding JSON from LLM response: {e}")
            # Here, a more robust system might try to re-prompt the LLM or use a parsing-fixer agent.
        except Exception as e:
            print(f"Planner Agent: An unexpected error occurred: {e}")
