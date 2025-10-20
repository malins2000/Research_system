import json
from typing import Any, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        Gathers information on a given topic from all available sources in parallel.

        Args:
            topic: The topic to research.
            num_results: The desired number of results per query.

        Returns:
            A consolidated and de-duplicated list of retrieved documents.
        """
        print(f"Retrieval Agent: Gathering information for topic: '{topic}'")

        # Step 1: Brainstorm search queries (this remains sequential)
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
            search_queries = [topic]

        # Step 2: Execute all queries and source searches in parallel
        all_retrieved_docs = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for query in search_queries:
                print(f"Retrieval Agent: Submitting jobs for query: '{query}'")
                # Submit a job for the RAG system
                futures.append(executor.submit(self.rag_system.query, query_text=query, k=num_results))
                # Submit a job for the Arxiv tool
                futures.append(executor.submit(self.arxiv_tool.search, query=query, max_results=num_results))

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    results = future.result()
                    if results:
                        all_retrieved_docs.extend(results)
                except Exception as e:
                    print(f"Retrieval Agent: A search job failed: {e}")

        print(f"Retrieval Agent: Collected {len(all_retrieved_docs)} raw results from all sources.")

        # Step 3: Consolidate and de-duplicate results
        unique_docs = {}
        for doc in all_retrieved_docs:
            doc_id = doc.get('metadata', {}).get('doc_id')
            if doc_id:
                unique_docs[doc_id] = doc
            else:
                unique_docs[doc['content']] = doc

        final_results = list(unique_docs.values())
        print(f"Retrieval Agent: Found {len(final_results)} unique documents.")

        # Step 4: Add new documents to RAG (can also be parallelized)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for doc in final_results:
                # We only need to add documents that came from external sources
                if doc.get('metadata', {}).get('source') == 'arXiv':
                    futures.append(executor.submit(self.rag_system.add_document, doc['content'], doc['metadata']))

            # Wait for all add operations to complete
            for future in as_completed(futures):
                try:
                    future.result() # We don't need the return value, just wait for it to finish
                except Exception as e:
                    print(f"Retrieval Agent: Failed to add document to RAG: {e}")

        print("Retrieval Agent: RAG system updated with new findings.")

        return final_results
