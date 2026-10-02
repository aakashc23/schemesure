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
orphaned or re-spawned; a module-level daemon thread guarded by a global starts
once per process and dies with it. `@st.cache_resource` would also work but is
re-evaluated per session, which is the wrong lifetime for a server.
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

_SERVER_LOCK = threading.Lock()
_SERVER_STARTED = False


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


def ensure_api_running() -> bool:
    """
    Start the API thread once, then wait for it to answer /health.

    First start is slow: it loads the embedding model and, on a host with no
    build step, builds the Chroma index from data/schemes. Both happen once per
    process, so the wait is shown to the user instead of silently timing out.
    """
    global _SERVER_STARTED

    if _api_is_up():
        return True

    with _SERVER_LOCK:
        # Re-check inside the lock: a second Streamlit session can arrive here
        # while the first is still starting the server.
        if not _SERVER_STARTED:
            threading.Thread(target=_serve, daemon=True, name="schemesure-api").start()
            _SERVER_STARTED = True

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

    runpy.run_path("ui/streamlit_app.py", run_name="__main__")
