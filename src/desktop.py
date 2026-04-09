"""Desktop launcher for LearnMate.

Starts the FastAPI server in a background thread, then opens a native
desktop window via pywebview. Closing the window shuts everything down.

Run: python -m src.desktop
"""

import threading
import time

import uvicorn
import webview

from src.api.app import app

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860


def _start_server() -> None:
    """Run uvicorn in a daemon thread."""
    config = uvicorn.Config(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="warning",  # quiet in desktop mode
    )
    server = uvicorn.Server(config)
    server.run()


def main() -> None:
    """Start server, wait for it to be ready, then open the window."""
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()

    # Wait until the server accepts connections (max 10s)
    import socket

    for _ in range(40):
        try:
            with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=0.25):
                break
        except OSError:
            time.sleep(0.25)

    url = f"http://{SERVER_HOST}:{SERVER_PORT}"

    webview.create_window(
        title="LearnMate",
        url=url,
        width=960,
        height=700,
        min_size=(720, 520),
        frameless=False,
    )
    webview.start()


if __name__ == "__main__":
    main()
