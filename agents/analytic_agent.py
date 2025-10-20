import json
import re
from typing import Any, List
from agents.base_agent import BaseAgent
from tools.persona_loader import PersonaLoader
from utils import parse_llm_json_output

class AnalyticAgent(BaseAgent):
    """
    An agent that analyzes a task to determine the required expertise.
    """

    def __init__(self, llm_client: Any, persona_loader: PersonaLoader):
        """
        Initializes the AnalyticAgent with an LLM client and a PersonaLoader.

        Args:
            llm_client: An instance of an LLM client.
            persona_loader: An instance of the PersonaLoader tool.
        """
        super().__init__(llm_client)
        self.persona_loader = persona_loader

    def execute(self, context: str) -> List[str]:
        """
        Determines the most relevant expert roles for a given task context.

        Args:
            context: A string describing the task, such as a plan point's description.

        Returns:
            A list of the most relevant expert role names.
        """
        print("Analytic Agent: Determining required expertise...")

        available_roles = self.persona_loader.list_personas()
        if not available_roles:
            print("Analytic Agent: No personas found. Cannot determine expertise.")
            return []

        prompt = (
            f"You are a project manager. Based on the task description, select the 2-3 most relevant expert roles "
            f"from the available list. Think step-by-step. First, analyze the task. Second, create a JSON list "
            f"of strings for the roles. Finally, wrap this JSON object in <roles_json> tags. "
            f"Do not include any other text after the closing </roles_json> tag.\n\n"
            f"**Task Description:**\n{context}\n\n"
            f"**Available Roles:**\n{', '.join(available_roles)}\n\n"
            f"YOUR RESPONSE:"
        )

        response_str = self.llm_client.query(prompt)
        parsed_result = parse_llm_json_output(response_str, "roles_json") # Use the correct tag

        if parsed_result:
            selected_roles = parsed_result
            if isinstance(selected_roles, list) and all(isinstance(role, str) for role in selected_roles):
                valid_roles = [role for role in selected_roles if role in available_roles]
                if len(valid_roles) != len(selected_roles):
                    print(f"Analytic Agent: Warning - LLM suggested roles that do not exist: {set(selected_roles) - set(valid_roles)}")
                print(f"Analytic Agent: Selected roles - {valid_roles}")
                return valid_roles
            else:
                print("Analytic Agent: LLM response was not a list of strings.")
                return []
        else:
            # Handle parsing failure robustly
            print(f"Analytic Agent: CRITICAL - Failed to parse valid JSON from LLM after all fallbacks.")
            return []
            raise ValueError("Failed to parse a valid plan from the LLM. Stopping workflow.")