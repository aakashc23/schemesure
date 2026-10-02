#!/usr/bin/env bash
# Start FastAPI (internal) and Streamlit (published), and keep them tied together.
#
# WHY A SCRIPT: a Hugging Face Space publishes one port. Streamlit takes it;
# FastAPI runs behind it on 127.0.0.1:8000, reachable only from inside the
# container. The UI talks to it over plain HTTP, exactly as it does locally, so
# there is no "deployed-only" code path to debug.
#
# NOTE: this file must have LF line endings. With CRLF, Linux tries to run
# "bash\r" and fails with a baffling "no such file or directory".
# .gitattributes enforces that.

set -euo pipefail

API_PORT="${API_PORT:-8000}"
UI_PORT="${PORT:-7860}"     # Spaces may inject PORT; default to the Space port.

echo "[start] SchemeSure starting"
echo "[start] API  -> 127.0.0.1:${API_PORT} (internal)"
echo "[start] UI   -> 0.0.0.0:${UI_PORT} (published)"

if [ -z "${LLM_API_KEY:-}" ]; then
  # Not fatal: the app boots, /health reports the missing key and the Schemes
  # tab still works. Better than a container that refuses to start.
  echo "[start] WARNING: LLM_API_KEY is not set — asking questions will be disabled."
fi

# --- FastAPI, in the background ---------------------------------------------
python -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port "${API_PORT}" \
  --workers 1 \
  --log-level info &
API_PID=$!

# If the container is told to stop, stop the API too rather than orphaning it.
cleanup() {
  echo "[start] shutting down (api pid ${API_PID})"
  kill "${API_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- Wait for the API to become healthy -------------------------------------
# Startup loads the embedding model and opens Chroma, which takes a few seconds.
# Launching Streamlit first would show the user a connection error on their very
# first page load, so we wait for a real 200 from /health.
echo "[start] waiting for the API to become healthy..."
READY=0
for attempt in $(seq 1 90); do
  if ! kill -0 "${API_PID}" 2>/dev/null; then
    echo "[start] FATAL: the API process exited during startup."
    wait "${API_PID}" || true
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    READY=1
    echo "[start] API healthy after ~$((attempt * 2))s"
    break
  fi
  sleep 2
done

if [ "${READY}" -ne 1 ]; then
  echo "[start] FATAL: the API did not become healthy within 180s."
  exit 1
fi

# Log what the API reports, so a bad deploy (empty index, missing key) is
# obvious in the Space's build logs instead of needing to be reproduced.
echo "[start] /health says:"
curl -s "http://127.0.0.1:${API_PORT}/health" || true
echo

# --- Streamlit, in the foreground -------------------------------------------
# exec: Streamlit becomes PID 1 so it receives the container's stop signals.
export API_URL="http://127.0.0.1:${API_PORT}"

exec python -m streamlit run ui/streamlit_app.py \
  --server.port "${UI_PORT}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.fileWatcherType none
