import json
import re
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

    def execute(self, proposals: List[Dict], plan_manager: PlanManager, parent_node_id: str, main_prompt: str) -> None:
        """
        Evaluates topic proposals using an LLM and adds approved ones to the research plan.

        Args:
            proposals: A list of proposal dictionaries.
            plan_manager: An instance of the PlanManager.
            parent_node_id: The ID of the parent node under which to potentially add new topics.
            main_prompt: The original user prompt for the entire research project.
        """
        print("Plan Updater Agent: Evaluating topic proposals using LLM...")

        if not proposals:
            print("Plan Updater Agent: No proposals to evaluate.")
            return

        # Get the current plan structure for context (as JSON string)
        current_plan_str = json.dumps(plan_manager.plan.model_dump(), indent=2) if plan_manager.plan else "{}"

        # --- CONFIGURATION ---
        MAX_PLAN_DEPTH = 5  # Set a maximum depth limit
        # ---------------------

        # Find the parent node to check its current depth
        parent_node = plan_manager._find_node_by_id(plan_manager.plan, parent_node_id)
        if not parent_node:
            print(f"Plan Updater Agent: Error - Parent node with ID {parent_node_id} not found.")
            return  # Cannot add nodes without a valid parent

        # Calculate the depth of the parent node (root is depth 0)
        def get_depth(node_id, current_node=plan_manager.plan, depth=0):
            if not current_node: return -1
            if current_node.id == node_id: return depth
            for child in current_node.children:
                found_depth = get_depth(node_id, child, depth + 1)
                if found_depth != -1: return found_depth
            return -1

        parent_depth = get_depth(parent_node_id)

        if parent_depth >= MAX_PLAN_DEPTH:
            print(
                f"Plan Updater Agent: Skipping proposals under node {parent_node_id}. Parent is already at max depth ({MAX_PLAN_DEPTH}).")
            return

        for proposal in proposals:
            # Basic validation
            if not all(key in proposal for key in ['title', 'summary', 'justification']):
                print(f"Plan Updater Agent: Skipping invalid proposal due to missing keys: {proposal}")
                continue

            # --- LLM Evaluation Call ---
            proposal_str = json.dumps(proposal, indent=2)
            prompt = (
                f"You are a strict project manager evaluating a proposed subtopic. "
                f"First, think step-by-step in <think> tags. "
                f"Second, decide whether to APPROVE or REJECT the proposal based ONLY on the criteria below. "
                f"Finally, output your decision wrapped in <decision> tags (e.g., <decision>APPROVE: Relevant and novel.</decision> or <decision>REJECT: Too similar to existing topic 'X'.</decision>). "
                f"Your entire response MUST end with the closing </decision> tag.\n\n"
                f"**Main Project Goal:** {main_prompt}\n\n"
                f"**Parent Topic ID:** {parent_node_id}\n"
                f"**Parent Topic Depth:** {parent_depth} (Max allowed depth is {MAX_PLAN_DEPTH})\n\n"
                f"**Proposed Subtopic:**\n{proposal_str}\n\n"
                f"**Existing Plan Structure (Partial):**\n{current_plan_str[:2000]}...\n\n"  # Limit context
                f"**Evaluation Criteria (REJECT if ANY are not met):**\n"
                f"1. **Relevance:** Is the proposal directly relevant to the Parent Topic and Main Project Goal?\n"
                f"2. **Novelty:** Is it sufficiently distinct from existing topics in the plan? (Check Existing Plan Structure)\n"
                f"3. **Scope:** Is the scope narrow enough to be manageable as a single research point?\n"
                f"4. **Depth:** Will adding this node exceed the MAX_PLAN_DEPTH ({MAX_PLAN_DEPTH})?\n\n"
                f"YOUR RESPONSE:"
            )

            response_str = self.llm_client.query(prompt)

            # --- Parse Decision ---
            decision = "REJECT"  # Default to reject
            justification = "Parsing failure or default."
            try:
                # Use simplified regex for APPROVE/REJECT
                match = re.search(r"<decision>(APPROVE|REJECT):?\s*(.*)</decision>", response_str,
                                  re.IGNORECASE | re.DOTALL)
                if match:
                    decision = match.group(1).upper()
                    justification = match.group(2).strip()
                else:
                    print(f"Plan Updater Agent: Warning - Could not parse <decision> tag from LLM response.")
                    print(f"Raw response: {response_str}")
                    # Keep decision as REJECT

            except Exception as e:
                print(f"Plan Updater Agent: Error during decision parsing: {e}")
                # Keep decision as REJECT

            # --- Add Node if Approved ---
            if decision == "APPROVE":
                print(
                    f"Plan Updater Agent: Approving and adding sub-node: '{proposal['title']}' (Justification: {justification})")
                new_node_data = {
                    "title": proposal['title'],
                    "description": proposal['summary'],
                    "experts_needed": []  # To be filled later
                }
                plan_manager.add_sub_node(
                    parent_id=parent_node_id,
                    new_node_data=new_node_data
                )
            else:
                print(f"Plan Updater Agent: Rejecting proposal: '{proposal['title']}' (Reason: {justification})")
