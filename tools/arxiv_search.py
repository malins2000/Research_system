import arxiv
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import deque
import logging


class ArxivSearchTool:
    """
    A synchronous tool for searching arXiv.
    This tool is designed to be called by the RetrievalAgent.
    """

    def __init__(self):
        self.client = arxiv.Client()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - ArxivTool - %(message)s"))
            self.logger.addHandler(handler)
        self.logger.info("ArxivSearchTool initialized.")

    def search(self, query: str, max_results: int = 5, days_limit: Optional[int] = None) -> deque[Dict[str, Any]]:
        """
        Performs a single, synchronous search on arXiv.

        Args:
            query: The search query string.
            max_results: The maximum number of results to return.
            days_limit: Optional. Limit results to the last X days.

        Returns:
            A deque object of formatted result dictionaries.
        """
        self.logger.info(f"Executing arXiv search for query: '{query}'")
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )

            results = deque()
            cutoff_date = None
            if days_limit:
                cutoff_date = datetime.now() - timedelta(days=days_limit)

            for paper in self.client.results(search):
                if cutoff_date:
                    # Note: The arxiv library handles timezone-naive comparison
                    if paper.published.replace(tzinfo=None) < cutoff_date:
                        continue

                paper_info = {
                    "content": f"Title: {paper.title}\nAuthors: {', '.join(author.name for author in paper.authors)}\nAbstract: {paper.summary}",
                    "metadata": {
                        "source": "arXiv",
                        "doc_id": paper.entry_id,
                        "published": str(paper.published),
                        "pdf_url": paper.pdf_url
                    }
                }
                results.append(paper_info)

            self.logger.info(f"Found {len(results)} results from arXiv.")
            return results
        except Exception as e:
            self.logger.error(f"Error in arXiv search: {str(e)}")
            return deque()