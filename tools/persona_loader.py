import os
from typing import List

class PersonaLoader:
    """
    A utility to load agent personas from a directory of text files.
    """
    def __init__(self, persona_directory: str = "./personas"):
        """
        Initializes the PersonaLoader with the path to the persona directory.

        Args:
            persona_directory: The path to the directory containing persona files.

        Raises:
            ValueError: If the specified directory does not exist.
        """
        if not os.path.isdir(persona_directory):
            raise ValueError(f"Persona directory not found at: {persona_directory}")
        self.persona_directory = persona_directory

    def list_personas(self) -> List[str]:
        """
        Lists the names of all available personas.

        The persona name is derived from the filename (without the .txt extension).

        Returns:
            A list of available persona names.
        """
        return [f.replace('.txt', '') for f in os.listdir(self.persona_directory) if f.endswith('.txt')]

    def get_persona(self, role_name: str) -> str:
        """
        Retrieves the system prompt for a specific persona.

        Args:
            role_name: The name of the role (e.g., 'economist').

        Returns:
            The content of the persona file as a string.

        Raises:
            FileNotFoundError: If the persona file for the given role name does not exist.
        """
        filepath = os.path.join(self.persona_directory, f"{role_name}.txt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Persona file not found for role: {role_name}")

        with open(filepath, 'r') as f:
            return f.read()
