import json

class MockLLMClient:
    """
    A mock LLM client for development and testing purposes.

    This client simulates the behavior of a real LLM client by returning
    pre-defined, plausible responses based on more specific keywords found
    in the prompts from different agents.
    """
    def query(self, prompt: str) -> str:
        """
        Simulates a query to an LLM based on specific keywords.

        Args:
            prompt: The input prompt for the LLM.

        Returns:
            A string containing a simulated LLM response.
        """
        prompt = prompt.lower()

        # PlannerAgent prompt
        if "generate" in prompt and "research plan" in prompt:
            plan_structure = {
                "children": [
                    {"title": "Introduction", "description": "A brief introduction to the research topic.", "experts_needed": ["economist"]},
                    {"title": "Historical Background", "description": "An overview of the topic's history.", "experts_needed": ["marine_biologist"]},
                    {"title": "Current Landscape", "description": "Analysis of the current state of the topic.", "experts_needed": ["economist"]},
                ]
            }
            return json.dumps(plan_structure)

        # CriticAgent prompt for the plan
        elif "evaluate" in prompt and "research plan" in prompt:
            critic_response = {
                "approved": True,
                "feedback": "The plan is logical and covers key areas.",
                "rating": 4.5
            }
            return json.dumps(critic_response)

        # CriticAgent prompt for a draft
        elif "evaluate" in prompt and "generated text" in prompt:
            critic_response = {
                "approved": True,
                "feedback": "The text is well-written and addresses the topic effectively.",
                "rating": 4.0
            }
            return json.dumps(critic_response)

        # AnalyticAgent prompt
        elif "select" in prompt and "expert roles" in prompt:
            # A bit of logic to make it respond to the context
            if "history" in prompt:
                return json.dumps(["marine_biologist"])
            else:
                return json.dumps(["economist"])

        # RetrievalAgent prompt
        elif "brainstorm" in prompt and "search queries" in prompt:
            queries = [
                "history of the topic", "recent developments in the topic", "future trends"
            ]
            return json.dumps(queries)

        # OutputGenerationAgent prompt
        elif "synthesize" in prompt and "expert insights" in prompt:
            return "This is a synthesized text combining the insights from various experts, forming a coherent and well-structured narrative."

        # TopicExplorerAgent prompt
        elif "identify" in prompt and "new topics" in prompt:
            proposals = [
                {"title": "New Discovery", "summary": "A potential new area of research.", "justification": "This was hinted at in the source material."}
            ]
            return json.dumps(proposals)

        # SummaryAgent prompt
        elif "summary" in prompt or "summarize" in prompt:
            return "This is an executive summary of the document, highlighting the main points and conclusions."

        else:
            return "This is a generic response from the mock LLM client."
