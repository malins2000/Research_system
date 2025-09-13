import json
from typing import Any, List
from agents.base_agent import BaseAgent
from tools.persona_loader import PersonaLoader

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

        # Formulate the prompt for the LLM
        prompt = (
            f"You are a project manager responsible for assigning tasks to specialists. "
            f"Based on the following task description, select the 2-3 most relevant expert roles from the available list. "
            f"Return your selection as a JSON list of strings.\n\n"
            f"**Task Description:**\n{context}\n\n"
            f"**Available Roles:**\n{', '.join(available_roles)}"
        )

        # Query the LLM
        response_str = self.llm_client.query(prompt)

        try:
            # Parse the JSON response
            selected_roles = json.loads(response_str)
            if isinstance(selected_roles, list) and all(isinstance(role, str) for role in selected_roles):
                # Filter to ensure only available roles are returned
                valid_roles = [role for role in selected_roles if role in available_roles]
                if len(valid_roles) != len(selected_roles):
                    print(f"Analytic Agent: Warning - LLM suggested roles that do not exist: {set(selected_roles) - set(valid_roles)}")

                print(f"Analytic Agent: Selected roles - {valid_roles}")
                return valid_roles
            else:
                print("Analytic Agent: LLM response was not a list of strings.")
                return []
        except json.JSONDecodeError as e:
            print(f"Analytic Agent: Error decoding JSON from LLM response: {e}")
            return []
        except Exception as e:
            print(f"Analytic Agent: An unexpected error occurred: {e}")
            return []
