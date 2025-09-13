import json
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

        prompt = (
            "You are a curious and insightful research assistant. Your goal is to identify novel, relevant, "
            "and interesting subtopics that are not explicitly covered in the main text but are suggested by the "
            "source material. Analyze the following generated text and the source documents it was based on.\n\n"
            "**Generated Text:**\n"
            f"{generated_text}\n\n"
            "**Source Documents:**\n"
            f"{context_str}\n\n"
            "Based on this information, identify 1-3 potential new topics for further research. "
            "These could be unanswered questions, interesting tangents, or logical next steps. "
            "Return your findings as a JSON list of dictionaries. Each dictionary must have three keys: "
            "'title' (a concise topic title), 'summary' (a brief description), and 'justification' "
            "(a sentence explaining why it's a valuable addition to the research plan)."
        )

        response_str = self.llm_client.query(prompt)

        try:
            proposals = json.loads(response_str)
            if isinstance(proposals, list) and all(isinstance(p, dict) for p in proposals):
                print(f"Topic Explorer Agent: Found {len(proposals)} new topic proposals.")
                return proposals
            else:
                print("Topic Explorer Agent: LLM response was not a list of dictionaries.")
                return []
        except json.JSONDecodeError as e:
            print(f"Topic Explorer Agent: Error decoding JSON from LLM response: {e}")
            return []
        except Exception as e:
            print(f"Topic Explorer Agent: An unexpected error occurred: {e}")
            return []
