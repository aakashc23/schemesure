"""
Deploy SchemeSure to a Hugging Face Docker Space.

WHAT IT DOES
  1. Authenticates with HF_TOKEN and resolves the target Space id.
  2. Creates the Space (sdk=docker) if it does not exist.
  3. Sets LLM_API_KEY / LLM_BASE_URL / LLM_MODEL as Space **secrets** via the API.
     They are never written into any uploaded file.
  4. Uploads the files the container needs — chosen by an explicit allow-list.
  5. Polls the build until it is RUNNING or has failed.

WHY AN ALLOW-LIST RATHER THAN AN IGNORE-LIST
A deny-list fails silently the day someone adds a new file that should not be
published — and on a public Space, publishing a secret is irreversible. The
allow-list means a new file is *not* uploaded until someone adds it here, which
is the safe direction to fail. The script also refuses outright if `.env` or any
database file somehow matches.

Run:
    python scripts/deploy_hf.py              # create/update, set secrets, wait
    python scripts/deploy_hf.py --status     # just report the current state
    python scripts/deploy_hf.py --no-wait    # push and return immediately
    python scripts/deploy_hf.py --verify     # check the live app responds
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

# Exactly the files the container needs. Everything else stays out.
ALLOW_PATTERNS = [
    "Dockerfile",
    "start.sh",
    "requirements.txt",
    "README.md",
    ".dockerignore",
    "app/*.py",
    "core/*.py",
    "ui/*.py",
    "data/schemes/*.md",
    "data/schemes/*.rules.json",
]

# Belt and braces: even if an allow-list pattern were widened by accident, a path
# matching any of these aborts the upload.
FORBIDDEN_NAMES = {".env", ".env.local"}
FORBIDDEN_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pem", ".key"}

# Stages reported by the Spaces runtime API.
TERMINAL_OK = {"RUNNING"}
TERMINAL_BAD = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "DELETING", "PAUSED"}


def fail(message: str) -> int:
    print(f"\nERROR: {message}")
    return 1


def resolve_space_id(api, settings) -> str:
    """Use HF_SPACE_ID if set, otherwise <username>/schemesure."""
    if settings.hf_space_id.strip():
        return settings.hf_space_id.strip()
    username = api.whoami()["name"]
    return f"{username}/schemesure"


def collect_files(root: Path) -> list[Path]:
    """Expand the allow-list and sanity-check every match."""
    matched: list[Path] = []
    for pattern in ALLOW_PATTERNS:
        matched.extend(sorted(root.glob(pattern)))

    files: list[Path] = []
    for path in matched:
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise RuntimeError(
                f"refusing to upload '{path.name}': it matches the forbidden list"
            )
        files.append(path)
    return files


def poll_build(api, space_id: str, timeout_seconds: int = 1800) -> str:
    """
    Poll the Space until the build settles.

    Generous timeout: the first build installs torch and embeds the corpus, which
    takes a while on free CPU hardware.
    """
    print(f"\nWaiting for the build (up to {timeout_seconds // 60} minutes)...")
    started = time.time()
    last_stage = ""

    while time.time() - started < timeout_seconds:
        try:
            runtime = api.get_space_runtime(repo_id=space_id)
            stage = str(getattr(runtime, "stage", "UNKNOWN"))
        except Exception as exc:  # noqa: BLE001 - transient API hiccup
            print(f"  (could not read status: {exc})")
            time.sleep(10)
            continue

        if stage != last_stage:
            elapsed = int(time.time() - started)
            print(f"  [{elapsed:4d}s] {stage}")
            last_stage = stage

        if stage in TERMINAL_OK:
            return stage
        if stage in TERMINAL_BAD:
            return stage

        time.sleep(10)

    return "TIMEOUT"


def verify_live(space_id: str, attempts: int = 10) -> bool:
    """
    Check the live Space actually serves the UI.

    Only the Streamlit port is public, so this confirms the page loads. The
    backend is verified indirectly: the UI calls /health on load and shows a
    loud error if the API is down.
    """
    import requests

    owner, name = space_id.split("/", 1)
    slug = f"{owner}-{name}".replace("_", "-").replace(".", "-").lower()
    url = f"https://{slug}.hf.space"

    print(f"\nVerifying the live app at {url} ...")
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                print(f"  OK: HTTP 200 ({len(response.content):,} bytes)")
                return True
            print(f"  attempt {attempt}: HTTP {response.status_code}")
        except Exception as exc:  # noqa: BLE001
            print(f"  attempt {attempt}: {type(exc).__name__}")
        time.sleep(15)

    print("  Could not confirm the app is serving.")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy to a Hugging Face Space.")
    parser.add_argument("--status", action="store_true", help="report state and exit")
    parser.add_argument("--no-wait", action="store_true", help="do not poll the build")
    parser.add_argument("--verify", action="store_true", help="check the live app and exit")
    parser.add_argument("--skip-secrets", action="store_true",
                        help="upload files but leave Space secrets untouched")
    args = parser.parse_args()

    settings = get_settings()

    if not settings.hf_token.strip():
        return fail(
            "HF_TOKEN is empty. Create a token with **write** permission at "
            "https://huggingface.co/settings/tokens and put it in .env"
        )

    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi(token=settings.hf_token)

    try:
        identity = api.whoami()
    except Exception as exc:  # noqa: BLE001
        return fail(f"HF_TOKEN was rejected: {exc}")

    # Never print the token itself.
    print(f"Authenticated as: {identity['name']}")

    space_id = resolve_space_id(api, settings)
    print(f"Target Space    : {space_id}")

    owner, name = space_id.split("/", 1)
    slug = f"{owner}-{name}".replace("_", "-").replace(".", "-").lower()
    space_url = f"https://huggingface.co/spaces/{space_id}"
    app_url = f"https://{slug}.hf.space"

    if args.verify:
        return 0 if verify_live(space_id) else 1

    if args.status:
        try:
            runtime = api.get_space_runtime(repo_id=space_id)
            print(f"Stage           : {getattr(runtime, 'stage', '?')}")
            print(f"Hardware        : {getattr(runtime, 'hardware', '?')}")
            print(f"Space page      : {space_url}")
            print(f"App URL         : {app_url}")
        except Exception as exc:  # noqa: BLE001
            return fail(f"could not read the Space: {exc}")
        return 0

    # ---- 1. create the Space if needed -------------------------------------
    try:
        api.repo_info(repo_id=space_id, repo_type="space")
        print("Space exists; updating it.")
    except HfHubHTTPError:
        print("Space does not exist; creating it (sdk=docker, public).")
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )

    # ---- 2. secrets (set BEFORE the build, so the first boot has them) -----
    if not args.skip_secrets:
        print("\nSetting Space secrets (values are never printed or uploaded):")
        secrets = {
            "LLM_API_KEY": settings.llm_api_key,
            "LLM_BASE_URL": settings.llm_base_url,
            "LLM_MODEL": settings.llm_model,
        }
        for key, value in secrets.items():
            if not str(value).strip():
                print(f"  - {key}: SKIPPED (empty locally)")
                continue
            try:
                api.add_space_secret(repo_id=space_id, key=key, value=str(value))
                print(f"  - {key}: set")
            except Exception as exc:  # noqa: BLE001
                print(f"  - {key}: FAILED ({type(exc).__name__})")
                return fail(f"could not set the Space secret {key}")

    # ---- 3. upload ---------------------------------------------------------
    try:
        files = collect_files(ROOT)
    except RuntimeError as exc:
        return fail(str(exc))

    print(f"\nUploading {len(files)} files:")
    for path in files:
        print(f"  {path.relative_to(ROOT).as_posix()}")

    try:
        api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=str(ROOT),
            allow_patterns=ALLOW_PATTERNS,
            commit_message="Deploy SchemeSure",
        )
    except Exception as exc:  # noqa: BLE001
        return fail(f"upload failed: {exc}")

    print("\nUpload complete. The Space rebuilds automatically.")
    print(f"  Space page : {space_url}")
    print(f"  Build logs : {space_url}?logs=build")
    print(f"  App URL    : {app_url}")

    if args.no_wait:
        return 0

    # ---- 4. wait for the build --------------------------------------------
    stage = poll_build(api, space_id)

    if stage in TERMINAL_OK:
        print(f"\nBuild finished: {stage}")
        verify_live(space_id)
        print(f"\nLive at: {app_url}")
        return 0

    print(f"\nBuild did not succeed (stage={stage}).")
    print(f"Read the logs at: {space_url}?logs=build")
    return 1


if __name__ == "__main__":
    sys.exit(main())
