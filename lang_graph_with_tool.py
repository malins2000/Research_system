import json
import re
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolCall, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import random
import openai

# --- 1. Define tools ---
@tool
def get_current_location_temperature(location_name: str) -> str:
    """Get a current temperature for given location using external API"""
    print("Tool checking for location temperature")
    # dummy logic
    base_temp = 15.3
    variable_temp = round(random.uniform(-10.0, 15.0), 1)
    location_temp = base_temp + variable_temp
    print(f"Selected temperature in {location_name} is {location_temp}°C.")
    return f"{location_temp}"

@tool
def take_note(filename: str, note_text: str) -> str:
    """Save note to file on disk"""
    print("Tool for taking notes was called.")
    print(f"Saving {note_text} to {filename} file")
    if not ".txt" in filename:
        filename += ".txt"
    try:
        with open(filename, "w") as f:
            f.write(note_text)
    except:
        return "failed to save note."
    else:
        return "note successfully saved!"

tools = [get_current_location_temperature, take_note]
tool_map = {t.name: t for t in tools}

# --- 2. Set up the model and Graph state ---
try:
    client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="vllm")
    models = client.models.list()
    MODEL_NAME = models.data[0].id
    print(f"Connected to the server. Using model: {MODEL_NAME}")
    llm = ChatOpenAI(base_url="http://localhost:8000/v1", api_key="vllm", model=MODEL_NAME, temperature=0)
    # Bind the tools directly to the LLM.
    # The model will now choose to call your functions by their actual names.
    llm_with_tools = llm.bind_tools(tools)
except Exception as e:
    print(f"Error connecting to vLLM server: {e}")
    llm_with_tools = None

assert llm_with_tools, "Failed to connect to VLLM server and bind tools."


# --- 3. NEW: A more robust parser ---
def parse_xml_tool_calls(ai_message: AIMessage) -> AIMessage:
    """A more robust parser that handles dirty JSON and creates a new message."""
    text_content = ai_message.content
    tool_calls = []

    # Find all <tool_call> blocks
    tool_call_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text_content, re.DOTALL)

    for block in tool_call_blocks:
        try:
            # Find the JSON object within the block
            json_match = re.search(r"\{.*\}", block, re.DOTALL)
            if not json_match:
                print(f"Warning: Could not find a JSON object in the tool call block: {block}")
                continue

            # Extract the clean JSON string
            json_str = json_match.group(0)
            tool_call_json = json.loads(json_str)

            # Check if the tool name is valid
            tool_name = tool_call_json.get("name")
            if tool_name not in [t.name for t in tools]:
                print(f"Warning: Model tried to call an invalid tool: {tool_name}")
                continue

            tool_calls.append(
                ToolCall(
                    name=tool_name,
                    args=tool_call_json.get("arguments", {}),
                    id=f"tool_{tool_name}"
                )
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse tool call JSON due to: {e}")
            print(f"Block content: {block}")

    # Return a new AIMessage with the original content and parsed tool_calls
    return AIMessage(
        content=text_content,
        tool_calls=tool_calls,
        id=ai_message.id,
    )


# --- 4. Define Graph Nodes (Unchanged logic) ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]


def call_model(state: AgentState):
    """Invokes the LLM and then parses the output for tool calls."""
    print("\n---AGENT: Deciding next action---")
    messages = state["messages"]
    response = llm.invoke(messages)
    parsed_response = parse_xml_tool_calls(response)
    return {"messages": [parsed_response]}


tool_node = ToolNode(tools)


# The router logic is also the same and now works!
def router(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print("---ROUTER: Tool call detected. Routing to executor.---")
        return "tools"
    else:
        print("---ROUTER: No tool call. Ending execution.---")
        return END


# --- 5. Define the Graph (Unchanged) ---
# ... (graph definition is unchanged) ...
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router)
workflow.add_edge("tools", "agent")
app = workflow.compile()

# --- 6. Run the Agent with a NEW, more forceful System Prompt ---
system_prompt_text = f"""You are a helpful assistant. Think step-by-step to plan a course of action.
You have access to the following tools ONLY:

1. `get_current_location_temperature`
   - Description: {get_current_location_temperature.description}
   - Arguments: `location_name: str`
   - Example: {{ "name": "get_current_location_temperature", "arguments": {{ "location_name": "New York" }} }}

2. `take_note`
   - Description: {take_note.description}
   - Arguments: `filename: str, note_text: str`
   - Example: {{ "name": "take_note", "arguments": {{ "filename": "meeting_summary.txt", "note_text": "Everyone agreed on new product except Alice." }} }}

When you need to use a tool, you MUST use one of the exact tool names listed above.
Plan your steps and call **one tool at a time**.
After you get the result from a tool, you can decide on the next step.
When you are ready to call a tool, output a `<tool_call>` block with the tool name and arguments in JSON format. Do not add any other text after the closing </tool_call> tag.
"""
system_prompt = SystemMessage(content=system_prompt_text)

initial_input = "What is the current temperature in London? And please save it to a file named london_temp.txt"
inputs = {"messages": [system_prompt, HumanMessage(content=initial_input)]}

# ... (the rest of the run code is unchanged) ...
final_state = app.invoke(inputs, {"recursion_limit": 10})
print("\n\n---FINAL ANSWER---")
if final_state['messages'][-1].content:
    print(final_state['messages'][-1].content)
else:
    print("Agent finished.")
