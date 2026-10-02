"""
Pre-push secret scanner. Exits non-zero if anything that looks like a live
credential is about to enter git history.

WHY: once a secret is committed and pushed to a public repo it must be treated
as leaked forever — rewriting history does not un-publish it. So the check runs
*before* the push, on exactly the content git is about to send.

Usage:
    python scripts/scan_secrets.py            # scan files staged for commit
    python scripts/scan_secrets.py --tracked  # scan everything git tracks
    python scripts/scan_secrets.py --all      # scan the whole working tree

Designed to be loud and slightly paranoid: a false alarm costs a few seconds,
a missed key costs a rotated account.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files/dirs that never need scanning (binary, vendored, or generated).
SKIP_DIRS = {
    ".git", "venv", ".venv", "__pycache__", "node_modules",
    "chroma_db", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".zip",
    ".whl", ".so", ".pyd", ".dll", ".bin", ".pt", ".onnx", ".db", ".sqlite3",
}

# Each pattern targets a *shape* that only real credentials have.
# Keeping them specific avoids drowning the signal in false positives.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Groq keys look like gsk_<52 alphanumerics>
    ("Groq API key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}")),
    # OpenAI keys: sk-... / sk-proj-...
    ("OpenAI API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    # Hugging Face user access tokens: hf_<34+>
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    # GitHub tokens: ghp_/gho_/ghu_/ghs_/ghr_ and fine-grained github_pat_
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("Private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----")),
    # A populated secret-looking assignment, e.g. LLM_API_KEY=abcdef123456...
    # Empty values and obvious placeholders are filtered out below.
    (
        "Populated secret assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token"
            r"|password|passwd|client[_-]?secret|hf[_-]?token)\b\s*[:=]\s*"
            r"['\"]?([A-Za-z0-9_\-\.]{16,})['\"]?"
        ),
    ),
]

# Strings that mean "this is a placeholder, not a credential".
PLACEHOLDER_HINTS = re.compile(
    r"(?i)^(?:your|my|put|add|insert|fill|xxx+|changeme|replace|example|sample|dummy"
    r"|test|fake|placeholder|none|null|empty|todo|redacted|\.{3}|<.*>|\$\{.*\}"
    r"|os\.environ|getenv|settings\.|secrets\.|\*+)"
)


def is_placeholder(value: str) -> bool:
    """True if a captured value is clearly documentation, not a live secret."""
    if PLACEHOLDER_HINTS.match(value):
        return True
    # Things like "LLM_API_KEY" or "YOUR_TOKEN_HERE" used as a name, not a value.
    if value.isupper() and "_" in value:
        return True
    # A value with no digits and no mixed case is very unlikely to be a key.
    if not any(ch.isdigit() for ch in value) and value.islower():
        return True
    return False


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout


def files_to_scan(mode: str) -> list[Path]:
    if mode == "staged":
        names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR").split("\n")
    elif mode == "tracked":
        names = git("ls-files").split("\n")
    else:  # all
        names = [
            str(p.relative_to(ROOT))
            for p in ROOT.rglob("*")
            if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
        ]

    paths = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        paths.append(path)
    return paths


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return a list of (line_number, finding_label, redacted_evidence)."""
    try:
        text = io.open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []

    findings: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        if len(line) > 4000:  # minified/generated line; skip
            continue
        for label, pattern in PATTERNS:
            for match in pattern.finditer(line):
                # For the generic assignment rule, the value is group 1.
                value = match.group(1) if match.groups() else match.group(0)
                if match.groups() and is_placeholder(value):
                    continue
                # Redact: never print the user's actual secret.
                shown = f"{value[:4]}...{len(value)} chars" if len(value) > 8 else "***"
                findings.append((line_number, label, shown))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan for leaked credentials.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tracked", action="store_true", help="scan all git-tracked files")
    group.add_argument("--all", action="store_true", help="scan the whole working tree")
    args = parser.parse_args()

    mode = "tracked" if args.tracked else "all" if args.all else "staged"
    paths = files_to_scan(mode)

    print(f"[scan-secrets] mode={mode}, files={len(paths)}")

    # Guard 1: .env must never be tracked by git, whatever the mode.
    tracked = {name.strip() for name in git("ls-files").split("\n") if name.strip()}
    hard_failures: list[str] = []
    for forbidden in (".env", ".env.local"):
        if forbidden in tracked:
            hard_failures.append(f"{forbidden} is tracked by git — it must be ignored")

    # Guard 2: .gitignore must actually list .env.
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        hard_failures.append(".gitignore is missing")
    else:
        lines = {
            line.strip()
            for line in io.open(gitignore, encoding="utf-8").read().split("\n")
        }
        if ".env" not in lines:
            hard_failures.append(".gitignore does not contain a bare '.env' entry")

    # Guard 3: content scan.
    total = 0
    for path in paths:
        for line_number, label, shown in scan_file(path):
            rel = path.relative_to(ROOT)
            print(f"  !! {rel}:{line_number}  {label}  [{shown}]")
            total += 1

    for failure in hard_failures:
        print(f"  !! {failure}")

    if total or hard_failures:
        print(
            f"\n[scan-secrets] FAILED: {total} suspicious value(s), "
            f"{len(hard_failures)} policy violation(s). Push aborted."
        )
        return 1

    print("[scan-secrets] PASSED: no credentials detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
