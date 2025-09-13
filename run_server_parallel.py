# run_server_parallel.py
import subprocess
import time


def main():
    # --- Konfiguracja ---
    # Upewnij się, że ta ścieżka jest poprawna
    # Llama
    # LOCAL_MODEL_PATH = "/media/malin/AI Data/Models/LLM/Llama/Llama_3_1/8B_Instruct"
    # SERVED_MODEL_NAME = "llama-3-8b-instruct-parallel"
    # Qwen 2
    # LOCAL_MODEL_PATH = "/media/malin/AI Data/Models/LLM/Qwen2-7B-Instruct"
    # SERVED_MODEL_NAME = "qwen2-7b-instruct"
    # Qwen 3 8B
    # LOCAL_MODEL_PATH = "/media/malin/AI Data/Models/LLM/Qwen3-8B"
    # SERVED_MODEL_NAME = "qwen3-8b-reasoning"
    # Qwen 3 14B FP8
    # LOCAL_MODEL_PATH = "/media/malin/AI Data/Models/LLM/Qwen3-14B-FP8"
    # SERVED_MODEL_NAME = "qwen3-14b-fp8-reasoning"
    # Qwen 3 14B
    # LOCAL_MODEL_PATH = "/media/malin/AI Data/Models/LLM/Qwen3-14B"
    # SERVED_MODEL_NAME = "qwen3-14b-reasoning"
    # Qwen 3 30B A3B FP8
    LOCAL_MODEL_PATH = "/media/malin/AI Data/Models/LLM/Qwen3-30B-A3B-FP8"
    SERVED_MODEL_NAME = "qwen3-30b-a3b-fp8-reasoning-moe"


    TENSOR_PARALLEL_SIZE = 2
    PORT = 8000
    # --------------------

    # non reasoner
    # Budowanie komendy, która zostanie wywołana w terminalu
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", LOCAL_MODEL_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--port", str(PORT),
        "--trust-remote-code",
        # reasoning flags
        # "--enable-reasoning",
        # "--reasoning-parser", "deepseek_r1",
        "--gpu_memory_utilization", "0.8", # default is 90%
        # tools
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
    ]

    print("--- Uruchamianie serwera vLLM ---")
    print(f"Model: {LOCAL_MODEL_PATH}")
    print(f"Liczba GPU: {TENSOR_PARALLEL_SIZE}")
    print(f"Port: {PORT}")
    print("Komenda:", " ".join(command))
    print("\nSerwer startuje w tle. Może to potrwać kilka minut...")
    print("Aby zatrzymać serwer, naciśnij Ctrl+C w tym oknie.")

    # Uruchomienie serwera jako procesu w tle
    try:
        # Używamy Popen, aby proces działał w tle, a logi pojawiały się na konsoli
        server_process = subprocess.Popen(command)
        # Czekaj na zakończenie procesu (np. przez Ctrl+C)
        server_process.wait()
    except KeyboardInterrupt:
        print("\nOtrzymano polecenie zatrzymania (Ctrl+C). Zamykanie serwera...")
        server_process.terminate()
        # Poczekaj chwilę, aby dać procesowi czas na zamknięcie
        time.sleep(5)
        print("Serwer zatrzymany.")

if __name__ == "__main__":
    main()