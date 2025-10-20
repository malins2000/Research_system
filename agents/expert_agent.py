import json
from typing import Any, List, Dict, Optional

from agents.base_agent import BaseAgent

class ExpertAgent(BaseAgent):
    """
    A generic expert agent whose behavior is defined by a system prompt.
    """

    def __init__(self, llm_client: Any, name: str, system_prompt: str):
        """
        Initializes the ExpertAgent.

        Args:
            llm_client: An instance of an LLM client.
            name: The name of the expert (e.g., "economist").
            system_prompt: The system prompt that defines the expert's persona and knowledge.
        """
        super().__init__(llm_client)
        self.name = name
        self.system_prompt = system_prompt

    def execute(self, task_description: str, context_data: List[Dict], discussion_history: List[str], project_summary_so_far: Optional[str], critic_feedback: Optional[str]) -> str:
        """
        Executes a task from the expert's point of view, considering the ongoing discussion.

        Args:
            task_description: A description of the task to be performed.
            context_data: A list of dictionaries, typically retrieved documents, to provide context.
            discussion_history: A list of strings representing the conversation from previous rounds.
            project_summary_so_far: A summary of the work completed so far in the project.

        Returns:
            A string containing the expert's insight, analysis, or contribution for this round.
        """
        print(f"Expert Agent '{self.name}': Executing task (considering discussion)...")

        # Format the context data into a readable string
        context_str = "\n\n".join([f"Source: {doc.get('metadata', {}).get('source', 'N/A')}\nContent: {doc.get('content', '')}" for doc in context_data])

        # Format the discussion history
        if not discussion_history:
            history_str = "No discussion has taken place yet. You are providing the first set of insights."
        else:
            history_str = "\n\n".join(discussion_history)

        # --- ADD THIS ---
        summary_context = "No overall project summary is available yet."
        if project_summary_so_far:
            summary_context = project_summary_so_far
        # --- END ADDITION ---

        # --- ADD THIS ---
        feedback_context = "No specific feedback from the critic on the previous attempt."
        if critic_feedback:
            feedback_context = f"IMPORTANT: The previous attempt at this section was rejected by the critic with the following feedback: '{critic_feedback}'. Ensure your contribution helps address these points."
        # --- END ADDITION ---

        prompt = (
            f"{self.system_prompt}\n\n"
            f"You are part of an expert panel working on a larger research project.\n\n"
            f"**Overall Project Summary (Work Completed So Far):**\n{summary_context}\n\n" # <-- ADDED
            f"**Current Task:** {task_description}\n\n"
            f"**Critic Feedback on Last Attempt:**\n{feedback_context}\n\n" # <-- ADDED
            f"**Contextual Data (For Current Task):**\n{context_str}\n\n"
            f"**Ongoing Discussion (For Current Task):**\n{history_str}\n\n"
            f"**Your Instructions:**\n"
            f"1. Review the Main Task, Contextual Data, and the Ongoing Discussion History.\n"
            f"2. Based on your unique expertise, provide your analysis. \n"
            f"3. You can *build on* others' points, *critique* them, or *introduce* a new perspective your colleagues may have missed.\n"
            f"4. Format your response as a clear, well-structured block of text. *Do not* prefix with your name (e.g., 'Economist:'). Just provide your thoughts.\n\n"
            f"Your Response:"
        )

        # Query the LLM
        response = self.llm_client.query(prompt)

        print(f"Expert Agent '{self.name}': Successfully generated response for this round.")
        # Return a formatted string that includes the agent's name for the discussion history
        return f"**{self.name}:**\n{response}"
