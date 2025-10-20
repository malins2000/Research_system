import json
import re
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

        # NEW PROMPT: Ask the model to wrap the JSON in <experts_json> tags.
        prompt = (
            f"You are a project manager. Based on the task description, select the 2-3 most relevant expert roles "
            f"from the available list. Think step-by-step. First, analyze the task. Second, create a JSON list "
            f"of strings for the roles. Finally, wrap this JSON object in <experts_json> tags. "
            f"Do not include any other text after the closing </experts_json> tag.\n\n"
            f"**Task Description:**\n{context}\n\n"
            f"**Available Roles:**\n{', '.join(available_roles)}\n\n"
            f"YOUR RESPONSE:"
        )

        # Query the LLM
        response_str = self.llm_client.query(prompt)
        json_str = "" # Initialize json_str to be available in the except block

        try:
            # NEW PARSING LOGIC: Use regex to find the content inside our tags.
            match = re.search(r"<experts_json>(.*?)</experts_json>", response_str, re.DOTALL)

            if not match:
                print("Analytic Agent: Error - Could not find <experts_json> tags in the LLM response.")
                print(f"Raw LLM Response:\n{response_str}")
                return [] # Return empty list if tags are not found

            # Extract the clean JSON string
            json_str = match.group(1).strip()

            # Parse the extracted JSON
            selected_roles = json.loads(json_str)

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
            print(f"Analytic Agent: Error decoding JSON from the extracted block: {e}")
            print(f"Extracted JSON String that failed parsing:\n{json_str}")
            return []
        except Exception as e:
            print(f"Analytic Agent: An unexpected error occurred: {e}")
            return []
