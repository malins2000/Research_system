import json
import threading
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class RetrievedDoc(BaseModel):
    """Data model for a document retrieved by the RAG system."""
    doc_id: str = Field(..., description="Unique identifier for the document.")
    content: str = Field(..., description="The full content of the document.")
    source: str = Field(..., description="The origin of the document (e.g., 'arXiv', 'Google Search').")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the document.")

class ExpertInsight(BaseModel):
    """Data model for an insight provided by an expert agent."""
    expert_name: str = Field(..., description="The name of the expert providing the insight.")
    insight: str = Field(..., description="The core insight or analysis.")
    confidence_score: float = Field(..., ge=0, le=1, description="The expert's confidence in the insight.")
    supporting_doc_ids: List[str] = Field(default_factory=list, description="List of document IDs supporting this insight.")

class TopicProposal(BaseModel):
    """Data model for a new topic proposed by the Topic Explorer agent."""
    title: str = Field(..., description="The proposed title for the new topic.")
    description: str = Field(..., description="A brief description of the new topic and its relevance.")
    justification: str = Field(..., description="Why this topic should be added to the research plan.")

class Blackboard:
    """
    A thread-safe, file-based blackboard for inter-agent communication.

    This class provides a simple key-value store organized into sections,
    backed by a JSON file. It ensures that all read and write operations
    are atomic, preventing race conditions when multiple agents access it
    concurrently.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, filepath: str = "blackboard.json"):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    cls._instance = super(Blackboard, cls).__new__(cls)
                    cls._instance._filepath = filepath
                    cls._instance._data = cls._instance._load_data()
        return cls._instance

    def _load_data(self) -> Dict[str, Any]:
        """Loads the blackboard data from the JSON file."""
        try:
            with open(self._filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_data(self) -> None:
        """Saves the current blackboard data to the JSON file."""
        with open(self._filepath, 'w') as f:
            json.dump(self._data, f, indent=4)

    def post(self, section: str, key: str, value: Any) -> None:
        """
        Posts or updates a key-value pair within a specific section.

        This method is thread-safe.

        Args:
            section: The name of the section (e.g., 'retrieved_data').
            key: The key for the data point.
            value: The value to be stored. Can be a Pydantic model or any JSON-serializable type.
        """
        with self._lock:
            if section not in self._data:
                self._data[section] = {}

            if isinstance(value, BaseModel):
                self._data[section][key] = value.model_dump()
            else:
                self._data[section][key] = value

            self._save_data()

    def get(self, section: str, key: str) -> Optional[Any]:
        """
        Retrieves a value from a specific section and key.

        This method is thread-safe.

        Args:
            section: The name of the section.
            key: The key of the data to retrieve.

        Returns:
            The retrieved value, or None if the section or key does not exist.
        """
        with self._lock:
            return self._data.get(section, {}).get(key)

    def get_section(self, section: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves an entire section from the blackboard.

        This method is thread-safe.

        Args:
            section: The name of the section to retrieve.

        Returns:
            A dictionary representing the section, or None if the section does not exist.
        """
        with self._lock:
            # Return a copy to prevent modification of the internal state
            return self._data.get(section, {}).copy()

    def clear_section(self, section: str) -> None:
        """
        Clears all data from a specific section of the blackboard.

        This is crucial for resetting state between major workflow loops.
        This method is thread-safe.

        Args:
            section: The name of the section to clear.
        """
        with self._lock:
            if section in self._data:
                self._data[section] = {}
                self._save_data()
