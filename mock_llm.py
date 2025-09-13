import json

class MockLLMClient:
    """
    A mock LLM client for development and testing purposes.

    This client simulates the behavior of a real LLM client by returning
    pre-defined, plausible responses based on keywords found in the prompt.
    It allows for testing agent logic without incurring API costs or
    requiring a live model server.
    """
    def query(self, prompt: str) -> str:
        """
        Simulates a query to an LLM.

        Args:
            prompt: The input prompt for the LLM.

        Returns:
            A string containing a simulated LLM response.
        """
        prompt = prompt.lower()

        if "plan" in prompt or "table of contents" in prompt:
            # Simulate response for the PlannerAgent
            # The prompt asks for a JSON object, so we return that.
            # The planner agent will then use this to create the plan.
            plan_structure = {
                "children": [
                    {"title": "Introduction", "description": "A brief introduction to the research topic.", "experts_needed": ["research_analyst"]},
                    {"title": "Historical Background", "description": "An overview of the topic's history.", "experts_needed": ["historian"]},
                    {"title": "Current Landscape", "description": "Analysis of the current state of the topic.", "experts_needed": ["market_analyst"]},
                    {"title": "Conclusion", "description": "A summary of the findings.", "experts_needed": ["technical_writer"]}
                ]
            }
            return json.dumps(plan_structure)

        elif "criticize" in prompt or "evaluate" in prompt:
            # Simulate response for the CriticAgent
            critic_response = {
                "approved": True,
                "feedback": "The plan is logical and covers the key areas. The 'Current Landscape' section could be more detailed.",
                "rating": 4.5
            }
            return json.dumps(critic_response)

        elif "search queries" in prompt or "brainstorm" in prompt:
            # Simulate response for the RetrievalAgent
            queries = [
                "What is the history of the topic?",
                "Recent developments in the topic",
                "Future trends of the topic",
                "Key challenges in the topic area"
            ]
            return json.dumps(queries)

        elif "summary" in prompt or "summarize" in prompt:
            # Simulate response for the SummaryAgent
            return "This is an executive summary of the document, highlighting the main points and conclusions."

        else:
            # Default fallback response
            return "This is a generic response from the mock LLM client."
