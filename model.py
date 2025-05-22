import os
import subprocess
import threading
import requests

def is_ollama_running() -> bool:
    """Check if the Ollama server is running."""
    try:
        response = requests.get("http://0.0.0.0:11434/")  # Adjust to OLLAMA_HOST if needed
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def _serve_ollama():
    """Function to start the Ollama server."""
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(["ollama", "serve"])

def start_ollama_server():
    """Start the Ollama server in a separate thread if it's not already running."""
    if not is_ollama_running():
        ollama_thread = threading.Thread(target=_serve_ollama)
        ollama_thread.start()
    else:
        print("** Ollama server is already running. **")

def initialize_chat_ollama():
    """Initialize and return the ChatOllama model."""
    start_ollama_server()  # Ensure Ollama server is running before model initialization