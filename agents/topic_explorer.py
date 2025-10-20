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
        """
        Identifies new research topics based on the latest generated text and source documents.

        Args:
            generated_text: The recently generated text for a plan point.
            retrieved_docs: The list of source documents used to generate the text.

        Returns:
            A list of structured proposal dictionaries, e.g.,
            [{'title': '...', 'summary': '...', 'justification': '...'}].
        """
        print("Topic Explorer Agent: Searching for new research avenues...")

        context_str = "\n\n".join([f"Source: {doc.get('metadata', {}).get('source', 'N/A')}\nContent: {doc.get('content', '')}" for doc in retrieved_docs])

        # NEW PROMPT: Ask the model to wrap the JSON in <topics_json> tags.
        prompt = (
            "You are a curious research assistant. Your goal is to identify novel subtopics based on the "
            "provided text and source material. Think step-by-step. First, analyze the content. Second, create a "
            "JSON list of dictionaries for 1-3 potential new topics. Finally, wrap this JSON object in "
            "<topics_json> tags. Do not include any other text after the closing </topics_json> tag.\n\n"
            "Each dictionary in the JSON list must have three keys: 'title' (a concise topic title), "
            "'summary' (a brief description), and 'justification' (why it's a valuable addition).\n\n"
            f"**Generated Text:**\n{generated_text}\n\n"
            f"**Source Documents:**\n{context_str}\n\n"
            "YOUR RESPONSE:"
        )

        response_str = self.llm_client.query(prompt)
        json_str = "" # Initialize for use in the except block

        try:
            # NEW PARSING LOGIC: Use regex to find the content inside our tags.
            match = re.search(r"<topics_json>(.*?)</topics_json>", response_str, re.DOTALL)

            if not match:
                print("Topic Explorer Agent: Error - Could not find <topics_json> tags in the LLM response.")
                print(f"Raw LLM Response:\n{response_str}")
                return []

            # Extract the clean JSON string
            json_str = match.group(1).strip()

            proposals = json.loads(json_str)

            if isinstance(proposals, list) and all(isinstance(p, dict) for p in proposals):
                print(f"Topic Explorer Agent: Found {len(proposals)} new topic proposals.")
                return proposals
            else:
                print("Topic Explorer Agent: LLM response was not a list of dictionaries.")
                return []
        except json.JSONDecodeError as e:
            print(f"Topic Explorer Agent: Error decoding JSON from the extracted block: {e}")
            print(f"Extracted JSON String that failed parsing:\n{json_str}")
            return []
        except Exception as e:
            print(f"Topic Explorer Agent: An unexpected error occurred: {e}")
            return []
