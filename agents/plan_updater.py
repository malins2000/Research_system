from typing import Any, List, Dict
from agents.base_agent import BaseAgent
from tools.plan_manager import PlanManager

class PlanUpdaterAgent(BaseAgent):
    """
    An agent responsible for updating the research plan with new topics.
    """

    def __init__(self, llm_client: Any):
        """
        Initializes the PlanUpdaterAgent with an LLM client.

        Args:
            llm_client: An instance of an LLM client.
        """
        super().__init__(llm_client)

    def execute(self, proposals: List[Dict], plan_manager: PlanManager, parent_node_id: str) -> None:
        """
        Evaluates topic proposals and adds approved ones to the research plan.

        Args:
            proposals: A list of proposal dictionaries from the TopicExplorerAgent.
            plan_manager: An instance of the PlanManager to modify the plan.
            parent_node_id: The ID of the parent node under which to add new topics.
        """
        print("Plan Updater Agent: Evaluating topic proposals...")

        if not proposals:
            print("Plan Updater Agent: No proposals to evaluate.")
            return

        for proposal in proposals:
            # Basic validation: ensure the proposal has the required keys.
            if not all(key in proposal for key in ['title', 'summary', 'justification']):
                print(f"Plan Updater Agent: Skipping invalid proposal due to missing keys: {proposal}")
                continue

            # In a more advanced system, an LLM call could be made here to
            # perform a final check for relevance and non-duplication against the existing plan.
            # For now, we'll use a simple rule-based approval.
            is_approved = True

            if is_approved:
                print(f"Plan Updater Agent: Approving and adding new sub-node: '{proposal['title']}'")

                # The PlanNode model expects 'description'. We map the 'summary' from the proposal to it.
                new_node_data = {
                    "title": proposal['title'],
                    "description": proposal['summary'],
                    # experts_needed can be determined by the AnalyticAgent in a subsequent step of the main loop.
                    "experts_needed": []
                }

                plan_manager.add_sub_node(
                    parent_id=parent_node_id,
                    new_node_data=new_node_data
                )
            else:
                print(f"Plan Updater Agent: Rejecting proposal: '{proposal['title']}'")
