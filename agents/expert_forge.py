from typing import Any, List
from agents.expert_agent import ExpertAgent
from tools.persona_loader import PersonaLoader

class ExpertForge:
    """
    A factory class for creating ExpertAgent instances.

    This class is responsible for instantiating experts based on a list of
    required roles. It uses the PersonaLoader to fetch the appropriate
    system prompts that define each expert's behavior.
    """

    def __init__(self, llm_client: Any, persona_loader: PersonaLoader):
        """
        Initializes the ExpertForge.

        Args:
            llm_client: An instance of an LLM client to be passed to the created experts.
            persona_loader: An instance of the PersonaLoader to get system prompts.
        """
        self.llm_client = llm_client
        self.persona_loader = persona_loader

    def create_experts(self, roles: List[str]) -> List[ExpertAgent]:
        """
        Creates a list of ExpertAgent instances for the given roles.

        Args:
            roles: A list of role names (e.g., ['economist', 'marine_biologist']).

        Returns:
            A list of fully configured ExpertAgent instances.
        """
        print(f"Expert Forge: Creating experts for roles: {roles}")
        experts = []
        for role in roles:
            try:
                system_prompt = self.persona_loader.get_persona(role)
                expert = ExpertAgent(
                    llm_client=self.llm_client,
                    name=role,
                    system_prompt=system_prompt
                )
                experts.append(expert)
                print(f"Expert Forge: Successfully created '{role}' expert.")
            except FileNotFoundError:
                print(f"Expert Forge: Warning - Persona file for role '{role}' not found. Skipping.")
            except Exception as e:
                print(f"Expert Forge: Error creating expert for role '{role}': {e}")

        return experts
