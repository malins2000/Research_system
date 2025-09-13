import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid

class PlanNode(BaseModel):
    """Data model for a node in the research plan."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the plan node.")
    title: str = Field(..., description="The title of the research topic or sub-topic.")
    description: str = Field(..., description="A detailed description of the topic.")
    status: str = Field(default='pending', description="The current status of the node ('pending', 'in-progress', 'completed').")
    experts_needed: List[str] = Field(default_factory=list, description="List of expert roles needed for this topic.")
    children: List['PlanNode'] = Field(default_factory=list, description="A list of sub-topics or child nodes.")

# This is needed for Pydantic to handle the recursive nature of the PlanNode model.
PlanNode.model_rebuild()

class PlanManager:
    """
    Manages the research plan stored in a JSON file.

    This class provides CRUD (Create, Read, Update, Delete) operations for
    managing a hierarchical research plan. It handles the serialization
    and deserialization of the plan to and from a JSON file.
    """
    def __init__(self, filepath: str = "research_plan.json"):
        """
        Initializes the PlanManager with the path to the plan file.

        Args:
            filepath: The path to the JSON file where the plan is stored.
        """
        self.filepath = filepath
        self.plan = self._load_plan()

    def _load_plan(self) -> Optional[PlanNode]:
        """Loads the research plan from the JSON file."""
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                return PlanNode(**data)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _save_plan(self) -> None:
        """Saves the current research plan to the JSON file."""
        if self.plan:
            with open(self.filepath, 'w') as f:
                json.dump(self.plan.model_dump(), f, indent=4)

    def create_plan(self, prompt: str, initial_structure: Dict[str, Any]) -> None:
        """
        Initializes a new research plan.

        Args:
            prompt: The user's initial research prompt.
            initial_structure: A dictionary representing the initial high-level plan.
        """
        # The root node could represent the overall project
        self.plan = PlanNode(title="Research Plan", description=f"Plan for: {prompt}", children=[PlanNode(**child) for child in initial_structure.get('children', [])])
        self._save_plan()

    def _find_node_by_id(self, node: PlanNode, node_id: str) -> Optional[PlanNode]:
        """Recursively searches for a node by its ID."""
        if node.id == node_id:
            return node
        for child in node.children:
            found = self._find_node_by_id(child, node_id)
            if found:
                return found
        return None

    def get_next_pending_node(self) -> Optional[PlanNode]:
        """
        Finds and returns the next node with 'pending' status using a DFS traversal.

        Returns:
            The next pending PlanNode, or None if no pending nodes are found.
        """
        if not self.plan:
            return None

        # Using a stack for iterative DFS
        stack = [self.plan]
        while stack:
            current_node = stack.pop()
            if current_node.status == 'pending':
                return current_node

            # Add children to the stack in reverse order to visit them from left to right
            for child in reversed(current_node.children):
                stack.append(child)

        return None

    def update_node_status(self, node_id: str, new_status: str) -> bool:
        """
        Updates the status of a specific plan node.

        Args:
            node_id: The ID of the node to update.
            new_status: The new status ('pending', 'in-progress', 'completed').

        Returns:
            True if the node was found and updated, False otherwise.
        """
        if not self.plan:
            return False

        node_to_update = self._find_node_by_id(self.plan, node_id)
        if node_to_update:
            node_to_update.status = new_status
            self._save_plan()
            return True
        return False

    def add_sub_node(self, parent_id: str, new_node_data: Dict[str, Any]) -> Optional[PlanNode]:
        """
        Inserts a new sub-topic (node) into the plan under a specified parent.

        Args:
            parent_id: The ID of the parent node.
            new_node_data: A dictionary containing the data for the new node.

        Returns:
            The newly created PlanNode if the parent was found, otherwise None.
        """
        if not self.plan:
            return None

        parent_node = self._find_node_by_id(self.plan, parent_id)
        if parent_node:
            new_node = PlanNode(**new_node_data)
            parent_node.children.append(new_node)
            self._save_plan()
            return new_node
        return None
