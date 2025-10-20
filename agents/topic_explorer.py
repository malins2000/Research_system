import json
import re
from typing import Any, List, Dict
from agents.base_agent import BaseAgent

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

        json_str = ""
        try:
            # Stage 1: Try parsing JSON wrapped in XML tags
            match = re.search(r"<proposals_json>(.*?)</proposals_json>", response_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                proposals = json.loads(json_str)
            else:
                # Stage 2 (Fallback): Use regex to find the first JSON object or array
                print("Topic Explorer Agent: Could not find <proposals_json> tags. Falling back to regex search.")
                json_match = re.search(r"\{.*\}|\[.*\]", response_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0).strip()
                    proposals = json.loads(json_str)
                else:
                    print(f"Topic Explorer Agent: Error - No JSON found in the LLM response.")
                    print(f"Raw LLM Response:\n{response_str}")
                    return []

            if isinstance(proposals, list):
                print(f"Topic Explorer Agent: Found {len(proposals)} new topic proposals.")
                return proposals
            else:
                print(f"Topic Explorer Agent: Parsed content was not a list. Content: {proposals}")
                return []
        except json.JSONDecodeError as e:
            print(f"Topic Explorer Agent: Error decoding JSON from extracted block: {e}")
            print(f"Extracted JSON String: {json_str}")
            return []
        except Exception as e:
            print(f"Topic Explorer Agent: An unexpected error occurred: {e}")
            return []
