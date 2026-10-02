# SchemeSure — single container running both the API and the UI.
#
# WHY ONE CONTAINER: a Hugging Face Space exposes exactly one port. Streamlit
# takes that port (7860) and FastAPI runs beside it on an internal port that is
# never published. start.sh supervises both.
#
# WHY THE INDEX IS BUILT AT IMAGE-BUILD TIME: Spaces have no persistent disk —
# anything written at runtime is lost on restart. The corpus is tiny (15 schemes,
# 312 chunks), so embedding it during the build is both simpler and better: the
# container starts with a ready index, nothing is recomputed on a cold start, and
# the build fails loudly if the data is broken rather than the app starting up
# and answering nothing.

FROM python:3.12-slim

# curl is used by start.sh to wait for the API's health check before launching
# the UI. Everything else we need is pure Python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as UID 1000. Creating that user up front (and using
# --chown on every COPY) avoids the permission errors that otherwise appear at
# runtime when the app tries to write its SQLite log or read the model cache.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The corpus contains Devanagari and the rupee sign; without this, any
    # print of scheme text raises UnicodeEncodeError on some locales.
    PYTHONIOENCODING=utf-8 \
    # Keep the model cache inside the user's home so it is writable, and so the
    # weights downloaded during the build are the ones used at runtime.
    HF_HOME=/home/user/.cache/huggingface \
    # Chroma phones home by default; switch that off.
    ANONYMIZED_TELEMETRY=False

WORKDIR $HOME/app

# ---- Dependencies -----------------------------------------------------------
# Installed before the source is copied, so editing code does not invalidate the
# (slow) dependency layer.
COPY --chown=user requirements.txt ./

# CPU-only torch, explicitly. sentence-transformers pulls in torch, and the
# default Linux wheel is the CUDA build at ~2.5 GB — far too large for a free
# Space and entirely wasted, since there is no GPU. Installing from the CPU
# index first means the requirements.txt install finds torch already satisfied.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.14.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ---- Application ------------------------------------------------------------
COPY --chown=user app   ./app
COPY --chown=user core  ./core
COPY --chown=user ui    ./ui
COPY --chown=user data  ./data
COPY --chown=user start.sh ./start.sh

# ---- Build-time preparation -------------------------------------------------
# 1. Download the embedding model now, so a cold start does not spend a minute
#    pulling ~120 MB from the Hub (and does not fail if the Hub is unreachable).
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
print('embedding model cached, dim =', m.get_sentence_embedding_dimension())"

# 2. Build the Chroma index. This also acts as a build-time smoke test: if the
#    scheme data is malformed or a chunk exceeds the model's token limit,
#    core.ingest exits non-zero and the image never gets published.
RUN python -m core.ingest

# Streamlit serves here; this is the only port the Space exposes.
EXPOSE 7860

# FastAPI's port is internal only and never published.
ENV API_PORT=8000 \
    API_URL=http://127.0.0.1:8000

CMD ["bash", "start.sh"]
