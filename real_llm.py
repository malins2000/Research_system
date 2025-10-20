import openai
from typing import Any


class RealLLMClient:
    """
    A wrapper class for the OpenAI client that provides a .query() method,
    matching the interface expected by the agents (like the MockLLMClient).
    """

    def __init__(self, base_url="http://localhost:8000/v1", api_key="vllm"):
        try:
            self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
            models = self.client.models.list()
            self.model_name = models.data[0].id
            print(f"RealLLMClient connected to vLLM. Using model: {self.model_name}")
        except Exception as e:
            print(f"Error connecting RealLLMClient to vLLM server (port 8000). {e}")
            raise

    def query(self, prompt: str) -> str:
        """
        The query method that all agents will call.
        """
        try:
            # Note: Your agents expect a simple prompt (user message), not a full chat history.
            # We will format it as such.
            messages = [
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
            )

            # Extract the text content from the response
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(f"Error during vLLM query: {e}")
            # Return an empty string or error message to prevent a crash
            return f"Error: {e}"