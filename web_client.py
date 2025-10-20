# web_client.py (Master Control Panel v3)
import gradio as gr
import openai
import time
import json
import requests
import os
from datetime import datetime

# --- OpenAI Client Configuration ---
API_KEY = "vllm"
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = ""
try:
    client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)
    models = client.models.list()
    MODEL_NAME = models.data[0].id
    print(f"Connected to vLLM server. Using model: {MODEL_NAME}")
except Exception as e:
    print(f"Error connecting to vLLM server (port 8000). Make sure it's running. Error: {e}")

# --- System API Configuration ---
SYSTEM_API_URL = "http://localhost:8001"
HISTORY_DIR = "conversation_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# === Chat History Functions ===

def save_history(history):
    if not history:
        return "History is empty, not saved.",
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_{timestamp}.json"
    filepath = os.path.join(HISTORY_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return f"Saved history as: {filename}", gr.Dropdown(choices=get_history_files())
    except Exception as e:
        return f"Error saving: {e}", gr.Dropdown(choices=get_history_files())

def load_history(filename):
    if not filename:
        return "No file selected.", []
    filepath = os.path.join(HISTORY_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            history = json.load(f)
        return f"Loaded history: {filename}", history
    except Exception as e:
        return f"Error loading: {e}", []

def get_history_files():
    try:
        files = [f for f in os.listdir(HISTORY_DIR) if f.endswith('.json')]
        return sorted(files, reverse=True)
    except Exception:
        return []

# === Control Panel Functions ===

def refresh_system_status():
    """Fetches the system status from the API and displays it."""
    try:
        response = requests.get(f"{SYSTEM_API_URL}/api/status")
        response.raise_for_status()
        data = response.json()

        is_running = data.get("is_running", False)
        current_prompt = data.get("current_prompt", "None")
        summary = data.get("summary", "No summary available.")

        status_md = (
            f"**System Status:** `{'RUNNING' if is_running else 'IDLE'}`\n\n"
            f"**Active Task:** `{current_prompt}`\n\n"
            f"**Agent Summary:**\n\n{summary}"
        )

        # Return the status and, importantly, the visibility of the controls
        if is_running:
            # If the system is running, disable the ability to start a new task
            return status_md, gr.Button(interactive=False), gr.Textbox(interactive=False)
        else:
            # If the system is idle, enable the controls
            return status_md, gr.Button(interactive=True), gr.Textbox(interactive=True)

    except requests.exceptions.ConnectionError:
        error_msg = "Error: Could not connect to the `status_server.py` server. Is it running on port 8001?"
        return error_msg, gr.Button(interactive=False), gr.Textbox(interactive=False)
    except Exception as e:
        return f"An error occurred: {e}", gr.Button(interactive=False), gr.Textbox(interactive=False)

def start_research_job(prompt):
    """Sends a command to start a new research job."""
    if not prompt:
        return "Error: A main research topic (prompt) is required."

    payload = {"prompt": prompt}
    try:
        response = requests.post(f"{SYSTEM_API_URL}/api/start_research", json=payload)
        response.raise_for_status()
        message = response.json().get("message", "Task started.")
        return f"Success: {message}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the `status_server.py` server."
    except requests.HTTPError as e:
        if e.response.status_code == 409:
            return "Info: The system is already processing another task."
        return f"HTTP Error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

def submit_new_task(title, description):
    """Sends a new sub-task to the active system."""
    if not title or not description:
        return "Task title and description are required."

    payload = {"title": title, "description": description}
    try:
        response = requests.post(f"{SYSTEM_API_URL}/api/add_task", json=payload)
        response.raise_for_status()
        message = response.json().get("message", "Task sent.")
        return f"Success: {message}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the `status_server.py` server."
    except requests.HTTPError as e:
        if e.response.status_code == 400:
            return "Error: Cannot add a task because no research is active."
        return f"HTTP Error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

# === Chat Function (unchanged) ===
def predict(message, history):
    if not MODEL_NAME:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Error: vLLM server (port 8000) is not available."})
        yield "", history, "", "", ""
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    start_time = time.monotonic()
    thinking_steps = ""
    streaming_tool_calls = {}

    yield "", history, "...", "...", "..."

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=history,
        temperature=0.7,
        stream=True,
    )

    token_cnt = 0
    for chunk in stream:
        token_added_in_chunk = False
        delta = chunk.choices[0].delta
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            thinking_steps += delta.reasoning_content
            if not token_added_in_chunk: token_cnt += 1; token_added_in_chunk = True
        if delta and delta.content:
            history[-1]["content"] += delta.content
            if not token_added_in_chunk: token_cnt += 1; token_added_in_chunk = True
        if delta and delta.tool_calls:
            if not token_added_in_chunk: token_cnt += 1; token_added_in_chunk = True
            for tool_chunk in delta.tool_calls:
                index = tool_chunk.index
                if index not in streaming_tool_calls:
                    streaming_tool_calls[index] = {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                if tool_chunk.id: streaming_tool_calls[index]['id'] = tool_chunk.id
                if tool_chunk.function.name: streaming_tool_calls[index]['function']['name'] = tool_chunk.function.name
                if tool_chunk.function.arguments: streaming_tool_calls[index]['function']['arguments'] += tool_chunk.function.arguments

        tools_json = json.dumps(list(streaming_tool_calls.values()), indent=2)
        yield "", history, thinking_steps, tools_json, "Generating..."

    end_time = time.monotonic()
    duration = end_time - start_time
    token_per_s = token_cnt / duration if duration > 0 else 0
    stats_text = (f"**Response Time:** {duration:.2f} s\n\n**Tokens:** {token_cnt}\n\n**Speed:** {token_per_s:.2f} tok/s")
    final_tools_json = json.dumps(list(streaming_tool_calls.values()), indent=2)
    yield "", history, thinking_steps, final_tools_json, stats_text

# === NEW Gradio Interface ===
with gr.Blocks(theme=gr.themes.Soft(), title="Research System Control Panel") as demo:
    gr.Markdown(f"<h1>Research System Control Panel (Model: {MODEL_NAME})</h1>")

    with gr.Tabs():
        # --- Chat Tab ---
        with gr.Tab(label="💬 Chat with Model"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.Tab(label="🤔 Reasoning"):
                            thinking_box = gr.Markdown(label="Thinking Path")
                        with gr.Tab(label="🛠️ Tools"):
                            tools_box = gr.Code(label="Detected Tools", language="json")
                        with gr.Tab(label="📊 Statistics"):
                            stats_box = gr.Markdown(label="Generation Statistics")

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Conversation", type="messages", height=600)
                    msg = gr.Textbox(label="Your Message", placeholder="Ask the model anything...", scale=4)

        # --- Control Panel Tab ---
        with gr.Tab(label="🚀 Main Research"):
            gr.Markdown("Start and monitor the main research system.")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 1. Start New Research")
                    start_prompt = gr.Textbox(label="Main Research Topic (Prompt)", placeholder="e.g., 'Effects of climate change on coastal ecosystems'")
                    start_btn = gr.Button("Start New Research", variant="primary")
                    start_status = gr.Markdown(label="Start Status")

                    gr.Markdown("### 2. Add Task (while running)")
                    task_title = gr.Textbox(label="Task Title")
                    task_desc = gr.Textbox(label="Task Description", lines=3)
                    submit_task_btn = gr.Button("Send Task to System")
                    task_status = gr.Markdown(label="Send Status")

                with gr.Column(scale=3):
                    gr.Markdown("### 3. Monitor System Status")
                    status_output = gr.Markdown(label="Current Status", value="Click 'Refresh' to get the status...")
                    refresh_btn = gr.Button("Refresh Status")

        # --- Chat History Tab ---
        with gr.Tab(label="🗂️ Conversation History (Chat)"):
            gr.Markdown("Save and load previous conversations with the model.")
            history_status = gr.Markdown()
            with gr.Row():
                history_dropdown = gr.Dropdown(label="Select history to load", choices=get_history_files())
                load_btn = gr.Button("Load")
            save_btn = gr.Button("Save Current Conversation")

    # --- Component-linking logic ---
    all_components = [msg, chatbot, thinking_box, tools_box, stats_box, start_prompt, start_status, task_title, task_desc, task_status, status_output, history_status]
    clear = gr.ClearButton(all_components)

    # Chat
    msg.submit(predict, [msg, chatbot], [msg, chatbot, thinking_box, tools_box, stats_box])

    # Control Panel
    start_btn.click(start_research_job, [start_prompt], [start_status])

    # The refresh button updates the status AND the state of the start buttons
    refresh_btn.click(refresh_system_status, [], [status_output, start_btn, start_prompt])

    submit_task_btn.click(submit_new_task, [task_title, task_desc], [task_status])

    # History
    save_btn.click(save_history, [chatbot], [history_status, history_dropdown])
    load_btn.click(load_history, [history_dropdown], [history_status, chatbot])

    # Run a status refresh once at startup to disable buttons if the system is already running
    demo.load(refresh_system_status, [], [status_output, start_btn, start_prompt])

# Launch the application
if __name__ == "__main__":
    demo.launch()