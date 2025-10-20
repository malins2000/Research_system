import json
from typing import Any, List, Dict
from agents.base_agent import BaseAgent
from tools.rag_system import RAGSystem
from tools.arxiv_search import ArxivSearchTool

class RetrievalAgent(BaseAgent):
    """
    An agent responsible for information gathering from various sources.
    """

    def __init__(self, llm_client: Any, rag_system: RAGSystem, arxiv_tool: ArxivSearchTool):
        """
        Initializes the RetrievalAgent with an LLM client and a RAG system.

        Args:
            llm_client: An instance of an LLM client.
            rag_system: An instance of the RAGSystem tool for internal search.
            arxiv_tool: An instance of the ArxivSearchTool for external search.
        """
        super().__init__(llm_client)
        self.rag_system = rag_system
        self.arxiv_tool = arxiv_tool
        # In a real system, other search tools (e.g., Google Search API) would be passed here too.

    def execute(self, topic: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Gathers information on a given topic.

        Args:
            topic: The topic to research.
            num_results: The desired number of results per query.

        Returns:
            A consolidated and de-duplicated list of retrieved documents.
        """
        print(f"Retrieval Agent: Gathering information for topic: '{topic}'")

        # Step 1: Brainstorm search queries with the LLM
        prompt = (
            f"You are a research assistant. Brainstorm a list of 3-5 diverse and effective search queries "
            f"to gather information on the following topic: '{topic}'. "
            f"Return the queries as a JSON list of strings."
        )

        response_str = self.llm_client.query(prompt)

        try:
            search_queries = json.loads(response_str)
        except json.JSONDecodeError as e:
            print(f"Retrieval Agent: Error decoding search queries from LLM response: {e}")
            # Fallback to using the topic itself as the only query
            search_queries = [topic]

        # Step 2: Execute queries and retrieve documents
        all_retrieved_docs = []
        for query in search_queries:
            print(f"Retrieval Agent: Executing query: '{query}'")

            # Query the RAG system
            rag_results = self.rag_system.query(query_text=query, k=num_results)
            all_retrieved_docs.extend(rag_results)

            # Query the arXiv tool
            arxiv_results = self.arxiv_tool.search(query=query, max_results=num_results)
            all_retrieved_docs.extend(arxiv_results)

        # Step 3: Consolidate and de-duplicate results
        unique_docs = {}
        for doc in all_retrieved_docs:
            # Use 'doc_id' from metadata for de-duplication
            doc_id = doc.get('metadata', {}).get('doc_id')
            if doc_id:
                unique_docs[doc_id] = doc
            else:
                # Fallback to using content for de-duplication if no id is present
                unique_docs[doc['content']] = doc

        final_results = list(unique_docs.values())
        print(f"Retrieval Agent: Found {len(final_results)} unique documents.")

        return final_results
