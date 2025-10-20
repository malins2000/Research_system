import json
import re
from typing import Any, List, Dict
from agents.base_agent import BaseAgent
from utils import parse_llm_json_output

class TopicExplorerAgent(BaseAgent):
    """
    An agent responsible for discovering new avenues of research.
    """

    def __init__(self, llm_client: Any):
        """
        Initializes the TopicExplorerAgent with an LLM client.

        Args:
            llm_client: An instance of an LLM client.
        """
        super().__init__(llm_client)

    def execute(self, generated_text: str, retrieved_docs: List[Dict]) -> List[Dict[str, str]]:
        # ... (print statement) ...
        context_str = "\n\n".join([f"Source: {doc.get('metadata', {}).get('source', 'N/A')}\nContent: {doc.get('content', '')}" for doc in retrieved_docs])

        # Updated prompt demanding XML-wrapped JSON
        prompt = (
            "You are a curious research assistant. First, think step-by-step in <think> tags. "
            "Second, identify 1-3 potential new topics for further research based on the provided text and sources. "
            "Finally, return your findings as a JSON list of dictionaries, wrapped ONLY in <proposals_json> tags. "
            "Your entire response *must* end with the closing </proposals_json> tag.\n\n"
            "Each dictionary must have 'title', 'summary', and 'justification'.\n\n"
            "If you find no new topics, return an empty list: <proposals_json>[]</proposals_json>\n\n"
            "**Generated Text:**\n"
            f"{generated_text}\n\n"
            "**Source Documents:**\n"
            f"{context_str}\n\n"
            "YOUR RESPONSE:"
        )

        response_str = self.llm_client.query(prompt)
        parsed_result = parse_llm_json_output(response_str, "proposals_json") # Use the correct tag

        if parsed_result:
            proposals = parsed_result
            print(f"Topic Explorer Agent: Found {len(proposals)} new topic proposals.")
            return proposals
        else:
            # Handle parsing failure robustly
            print(f"Topic Explorer Agent: CRITICAL - Failed to parse valid JSON from LLM after all fallbacks.")
            return []
            # raise ValueError("Failed to parse a valid plan from the LLM. Stopping workflow.")