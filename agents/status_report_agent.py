import json
from typing import Any, Dict
from agents.base_agent import BaseAgent
from tools.plan_manager import PlanNode  # We need this for type hinting

class StatusReportAgent(BaseAgent):
    """
    An agent that reads the system's state files and generates a
    human-readable summary.
    """
    def __init__(self, llm_client: Any):
        super().__init__(llm_client)

    def _read_json_file(self, filepath: str) -> Dict:
        """Safely reads a JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _get_plan_stats(self, plan_data: Dict) -> Dict:
        """Recursively counts node statuses."""
        stats = {'pending': 0, 'in-progress': 0, 'completed': 0, 'current_task': 'None'}
        if not plan_data:
            return stats

        def traverse(node):
            status = node.get('status', 'pending')
            stats[status] += 1
            if status == 'in-progress':
                stats['current_task'] = node.get('title', 'Unknown')

            for child in node.get('children', []):
                traverse(child)

        traverse(plan_data)
        return stats

    def execute(self) -> str:
        """
        Reads the plan and blackboard and generates a status summary.

        Returns:
            A string (Markdown format) summarizing the current system status.
        """
        print("Status Report Agent: Generating status...")

        # Read the state files
        plan_data = self._read_json_file("research_plan.json")
        blackboard_data = self._read_json_file("blackboard.json")

        # Analyze the plan
        plan_stats = self._get_plan_stats(plan_data)

        # Get a snapshot of the blackboard
        current_discussion = blackboard_data.get("expert_discussion", {}).get("transcript", [])
        current_retrieval = blackboard_data.get("retrieved_data", {}).get("docs", [])

        # Formulate a prompt for the LLM
        prompt = (
            f"You are a project manager. Summarize the following system status into a "
            f"concise, human-readable update in Markdown format.\n\n"
            f"**Plan Status:**\n"
            f"- Current Task: {plan_stats['current_task']}\n"
            f"- Completed Tasks: {plan_stats['completed']}\n"
            f"- In-Progress Tasks: {plan_stats['in-progress']}\n"
            f"- Pending Tasks: {plan_stats['pending']}\n\n"
            f"**Blackboard Snapshot:**\n"
            f"- Documents Retrieved for Current Task: {len(current_retrieval)} docs\n"
            f"- Expert Messages in Current Debate: {len(current_discussion)} messages\n"
            f"(Do not show the content of the blackboard, just the status.)"
        )

        # Query the LLM
        summary = self.llm_client.query(prompt)
        print("Status Report Agent: Summary generated.")
        return summary
