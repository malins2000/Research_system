# web_client.py (Master Control Panel)
import gradio as gr
import openai
import time
import json
import requests  # <-- NOWE
import os  # <-- NOWE
from datetime import datetime

# --- Konfiguracja Klienta OpenAI ---
API_KEY = "vllm"
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = ""
try:
    client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)
    models = client.models.list()
    MODEL_NAME = models.data[0].id
    print(f"Połączono z serwerem vLLM. Używany model: {MODEL_NAME}")
except Exception as e:
    print(f"Błąd łączenia z serwerem vLLM. Sprawdź, czy jest uruchomiony. Błąd: {e}")

# --- Konfiguracja API Systemu ---
SYSTEM_API_URL = "http://localhost:8001"  # <-- NOWE
HISTORY_DIR = "conversation_history"  # <-- NOWE
os.makedirs(HISTORY_DIR, exist_ok=True)  # <-- NOWE


# === NOWE FUNKCJE: HISTORIA CZATU ===

def save_history(history):
    """Zapisuje bieżącą historię czatu do pliku JSON."""
    if not history:
        return "Historia jest pusta, nie zapisano.",

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_{timestamp}.json"
    filepath = os.path.join(HISTORY_DIR, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # Zwracamy zaktualizowaną listę plików
        return f"Zapisano historię jako: {filename}", gr.Dropdown(choices=get_history_files())
    except Exception as e:
        return f"Błąd zapisu: {e}", gr.Dropdown(choices=get_history_files())


def load_history(filename):
    """Wczytuje historię czatu z wybranego pliku."""
    if not filename:
        return "Nie wybrano pliku.", []

    filepath = os.path.join(HISTORY_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            history = json.load(f)
        return f"Wczytano historię: {filename}", history
    except Exception as e:
        return f"Błąd wczytania: {e}", []


def get_history_files():
    """Pobiera listę zapisanych plików historii."""
    try:
        files = [f for f in os.listdir(HISTORY_DIR) if f.endswith('.json')]
        return sorted(files, reverse=True)
    except Exception:
        return []


# === NOWE FUNKCJE: PANEL KONTROLNY ===

def refresh_system_status():
    """Pobiera status systemu z API i go wyświetla."""
    try:
        response = requests.get(f"{SYSTEM_API_URL}/api/status")
        response.raise_for_status()  # Rzuci błąd dla statusów 4xx/5xx
        summary = response.json().get("summary", "Brak podsumowania.")
        return summary
    except requests.exceptions.ConnectionError:
        return "Błąd: Nie można połączyć się z serwerem `status_server.py`. Czy jest uruchomiony na porcie 8001?"
    except Exception as e:
        return f"Wystąpił błąd: {e}"


def submit_new_task(title, description):
    """Wysyła nowe zadanie do systemu przez API."""
    if not title or not description:
        return "Tytuł i opis zadania są wymagane."

    payload = {"title": title, "description": description}
    try:
        response = requests.post(f"{SYSTEM_API_URL}/api/add_task", json=payload)
        response.raise_for_status()
        message = response.json().get("message", "Zadanie wysłane.")
        return f"Sukces: {message}"
    except requests.exceptions.ConnectionError:
        return "Błąd: Nie można połączyć się z serwerem `status_server.py`."
    except Exception as e:
        return f"Wystąpił błąd: {e}"


# === GŁÓWNA FUNKCJA CZATU (bez zmian) ===
def predict(message, history):
    if not MODEL_NAME:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Błąd: Serwer vLLM (port 8000) nie jest dostępny."})
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
            if not token_added_in_chunk:
                token_cnt += 1
                token_added_in_chunk = True

        if delta and delta.content:
            history[-1]["content"] += delta.content
            if not token_added_in_chunk:
                token_cnt += 1
                token_added_in_chunk = True

        if delta and delta.tool_calls:
            if not token_added_in_chunk:
                token_cnt += 1
                token_added_in_chunk = True
            for tool_chunk in delta.tool_calls:
                index = tool_chunk.index
                if index not in streaming_tool_calls:
                    streaming_tool_calls[index] = {"id": None, "type": "function",
                                                   "function": {"name": "", "arguments": ""}}
                if tool_chunk.id:
                    streaming_tool_calls[index]['id'] = tool_chunk.id
                if tool_chunk.function.name:
                    streaming_tool_calls[index]['function']['name'] = tool_chunk.function.name
                if tool_chunk.function.arguments:
                    streaming_tool_calls[index]['function']['arguments'] += tool_chunk.function.arguments

        tools_json = json.dumps(list(streaming_tool_calls.values()), indent=2)
        yield "", history, thinking_steps, tools_json, "Generowanie..."

    end_time = time.monotonic()
    duration = end_time - start_time
    token_per_s = token_cnt / duration if duration > 0 else 0
    stats_text = (
        f"**Czas odpowiedzi:** {duration:.2f} s\n\n"
        f"**Tokeny:** {token_cnt}\n\n"
        f"**Prędkość:** {token_per_s:.2f} tok/s"
    )
    final_tools_json = json.dumps(list(streaming_tool_calls.values()), indent=2)
    yield "", history, thinking_steps, final_tools_json, stats_text


# === NOWY INTERFEJS GRADIO ===
with gr.Blocks(theme=gr.themes.Soft(), title="Panel Kontrolny Systemu Badawczego") as demo:
    gr.Markdown(f"<h1>Panel Kontrolny Systemu Badawczego (Model: {MODEL_NAME})</h1>")

    with gr.Tabs():
        # --- Zakładka Czat ---
        with gr.Tab(label="💬 Czat z Modelem"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.Tab(label="🤔 Rozumowanie"):
                            thinking_box = gr.Markdown(label="Ścieżka myślenia (Chain of Thought)")
                        with gr.Tab(label="🛠️ Narzędzia"):
                            tools_box = gr.Code(label="Wykryte narzędzia", language="json")
                        with gr.Tab(label="📊 Statystyki"):
                            stats_box = gr.Markdown(label="Statystyki ostatniej generacji")

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Rozmowa", type="messages", height=600)
                    msg = gr.Textbox(label="Twoja Wiadomość", placeholder="Zapytaj model o cokolwiek...", scale=4)

        # --- NOWA Zakładka: Panel Kontrolny ---
        with gr.Tab(label="🕹️ Panel Kontrolny Systemu"):
            gr.Markdown("Monitoruj i steruj głównym systemem badawczym (z `main.py`).")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 Status Systemu")
                    status_output = gr.Markdown(label="Aktualny status",
                                                value="Kliknij 'Odśwież', aby pobrać status...")
                    refresh_btn = gr.Button("Odśwież Status")

                with gr.Column(scale=1):
                    gr.Markdown("### ✍️ Dodaj Nowe Zadanie do Planu")
                    task_title = gr.Textbox(label="Tytuł zadania")
                    task_desc = gr.Textbox(label="Opis zadania", lines=3)
                    submit_task_btn = gr.Button("Wyślij Zadanie do Systemu")
                    task_status = gr.Markdown(label="Status wysyłania")

        # --- NOWA Zakładka: Historia Czatów ---
        with gr.Tab(label="🗂️ Historia Rozmów"):
            gr.Markdown("Zapisuj i wczytuj poprzednie rozmowy z modelem.")
            history_status = gr.Markdown()
            with gr.Row():
                history_dropdown = gr.Dropdown(label="Wybierz historię do wczytania", choices=get_history_files())
                load_btn = gr.Button("Wczytaj")
            save_btn = gr.Button("Zapisz bieżącą rozmowę")

    # --- Logika łączenia komponentów ---

    # Czat
    msg.submit(predict, [msg, chatbot], [msg, chatbot, thinking_box, tools_box, stats_box])

    # Przycisk "Clear" (domyślny z web_client.py)
    # Dodajemy go, aby wyczyścić wszystkie pola, w tym nowe
    all_components = [msg, chatbot, thinking_box, tools_box, stats_box, status_output, task_title, task_desc,
                      task_status, history_status]
    clear = gr.ClearButton(all_components)

    # Panel Kontrolny
    refresh_btn.click(refresh_system_status, [], [status_output])
    submit_task_btn.click(submit_new_task, [task_title, task_desc], [task_status])

    # Historia
    save_btn.click(save_history, [chatbot], [history_status, history_dropdown])
    load_btn.click(load_history, [history_dropdown], [history_status, chatbot])

# Uruchomienie aplikacji
if __name__ == "__main__":
    demo.launch()