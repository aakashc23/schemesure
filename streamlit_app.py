"""
Entry point for Streamlit Community Cloud (and any single-process host).

WHY THIS FILE EXISTS
--------------------
The project's real deployment shape is two processes in one container: FastAPI on
an internal port, Streamlit on the public one (see `Dockerfile` and `start.sh`).
Hugging Face now requires a paid plan for Docker Spaces, so the live demo runs on
Streamlit Community Cloud instead — which gives you exactly one Python process
and runs one script.

Rather than collapse the architecture and have the UI import `core/` directly,
this starts the **real FastAPI app in a background thread** and points the UI at
it over HTTP on localhost. The boundary is unchanged: the UI still only knows how
to make HTTP calls, all logic still lives behind the API, and the evaluation
harness still measures the same code path users hit. Only the process topology
differs, and `Dockerfile`/`start.sh` remain the production path.

WHY A THREAD AND NOT A SUBPROCESS
Streamlit re-runs this script on every interaction. A subprocess would be
orphaned or re-spawned; a daemon thread starts once per process and dies with it.

WHY `@st.cache_resource` AND NOT A MODULE GLOBAL
Streamlit executes this script into a FRESH module on every rerun, so a
module-level flag or lock is re-created each time and guards nothing: a second
session, a widget click or a page reload during the slow cold start would launch
another uvicorn thread (each one loads the ~470 MB embedding model and builds the
index before failing to bind the port). `st.cache_resource` is shared by the whole
Python process across all sessions and reruns, so the thread is started exactly
once.
"""

from __future__ import annotations

import os
import threading
import time

import requests
import streamlit as st

# The API only ever listens on loopback — it is an internal detail of this
# process, never exposed. Streamlit itself serves the public port.
API_HOST = "127.0.0.1"
API_PORT = int(os.environ.get("INTERNAL_API_PORT", "8000"))
API_URL = f"http://{API_HOST}:{API_PORT}"

# The UI reads this; set it before the UI module runs.
os.environ["API_URL"] = API_URL

def _serve() -> None:
    """Run uvicorn in this thread. Blocks for the life of the process."""
    import uvicorn

    from app.main import app

    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="warning",
        # No reloader and no signal handlers: both require the main thread.
        access_log=False,
    )


def _api_is_up(timeout: float = 1.0) -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=timeout).status_code == 200
    except requests.exceptions.RequestException:
        return False


@st.cache_resource
def _start_api_thread() -> threading.Thread:
    """
    Start the API thread. Cached, so it runs once per PROCESS, not per rerun.

    Every session and every rerun that reaches this call gets back the same
    thread object instead of starting another server.
    """
    thread = threading.Thread(target=_serve, daemon=True, name="schemesure-api")
    thread.start()
    return thread


def ensure_api_running() -> bool:
    """
    Start the API thread once, then wait for it to answer /health.

    First start is slow: it loads the embedding model and, on a host with no
    build step, builds the Chroma index from data/schemes. Both happen once per
    process, so the wait is shown to the user instead of silently timing out.
    """
    if _api_is_up():
        return True

    # The cached thread may have DIED (port still held, index build raised).
    # Clearing the cache makes the next call start a fresh server instead of
    # waiting 120 s on a corpse.
    if not _start_api_thread().is_alive():
        _start_api_thread.clear()
        _start_api_thread()

    placeholder = st.empty()
    placeholder.info(
        "Starting SchemeSure — loading the embedding model and building the "
        "search index. This happens once and takes up to a minute."
    )
    progress = st.progress(0.0)

    # 120s: model load (~15s) plus a cold index build (~25s), with headroom for
    # a slow shared runner.
    deadline = 120
    for elapsed in range(deadline):
        if _api_is_up():
            progress.empty()
            placeholder.empty()
            return True
        progress.progress(min((elapsed + 1) / deadline, 1.0))
        time.sleep(1)

    progress.empty()
    placeholder.error(
        "The backend did not start within 120 seconds. Reload the page to retry — "
        "if it keeps failing, the app may have run out of memory."
    )
    return False


if ensure_api_running():
    # Execute the real UI in this process, now that API_URL is set and the
    # backend is answering. runpy runs it as a script, which is what Streamlit
    # expects, and keeps ui/streamlit_app.py usable on its own for local
    # development against a separately-run API.
    import runpy
    from pathlib import Path

    # Relative to THIS file, not the working directory, so it works however the
    # host launches Streamlit.
    runpy.run_path(
        str(Path(__file__).resolve().parent / "ui" / "streamlit_app.py"),
        run_name="__main__",
    )
