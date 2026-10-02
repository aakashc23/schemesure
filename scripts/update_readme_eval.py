"""
Fill the README's evaluation table and the resume bullets from eval/raw_results.json.

WHY A SCRIPT: these are the numbers someone will quote in an interview, so they
must be the measured ones. Copying them by hand across three documents is exactly
how a figure ends up stale or transposed. This reads the machine-written results
and substitutes them, so the docs cannot drift from the run.

Run after `python eval/run_eval.py`:
    python scripts/update_readme_eval.py
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "eval" / "raw_results.json"
README = ROOT / "README.md"
NOTES = ROOT / "docs" / "INTERVIEW_NOTES.md"


def build_table(off: dict, on: dict) -> str:
    """The headline comparison, as markdown."""
    def row(label: str, key: str, unit: str = "%", better: str = "lower") -> str:
        before, after = off[key], on[key]
        delta = round(after - before, 1)
        if abs(delta) < 0.05:
            note = "no change"
        else:
            improved = (delta < 0) if better == "lower" else (delta > 0)
            note = f"{'better' if improved else 'worse'} ({delta:+}{unit})"
        return f"| {label} | {before}{unit} | {after}{unit} | {note} |"

    lines = [
        "| Metric | Guardrail OFF | Guardrail ON | Change |",
        "|---|---|---|---|",
        row("**Unsupported claim rate** (in delivered answers)",
            "unsupported_claim_rate", better="lower"),
        row("Answers containing any unsupported claim",
            "answers_with_unsupported_claim_rate", better="lower"),
        row("Correct refusal rate (unanswerable)",
            "correct_refusal_rate", better="higher"),
        row("Over-blocking rate (answerable refused)",
            "over_blocking_rate", better="lower"),
        row("Key fact coverage", "key_fact_coverage", better="higher"),
        row("Retrieval hit rate (top-5)", "retrieval_hit_rate", better="higher"),
        f"| Avg LLM calls / question | {off['avg_llm_calls']} | {on['avg_llm_calls']} "
        f"| +{round(on['avg_llm_calls'] - off['avg_llm_calls'], 2)} |",
        f"| Avg latency | {off['avg_latency_ms'] / 1000:.1f}s | "
        f"{on['avg_latency_ms'] / 1000:.1f}s | "
        f"{(on['avg_latency_ms'] - off['avg_latency_ms']) / 1000:+.1f}s |",
        "",
        f"*{off['questions']} questions per condition "
        f"({off['answerable']} answerable, {off['unanswerable']} unanswerable). "
        f"Full method and per-category breakdown in "
        f"[`eval/results.md`](eval/results.md).*",
    ]
    return "\n".join(lines)


def substitute(path: Path, start: str, end: str, body: str) -> bool:
    """Replace the content between two HTML-comment markers."""
    text = io.open(path, encoding="utf-8").read()
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end), re.S
    )
    if not pattern.search(text):
        print(f"  ! markers {start} / {end} not found in {path.name}")
        return False
    replaced = pattern.sub(f"{start}\n{body}\n{end}", text)
    io.open(path, "w", encoding="utf-8").write(replaced)
    return True


def fill_resume_bullets(on: dict, off: dict) -> bool:
    """Replace the <OFF>/<ON>/<X> placeholders in the resume bullets."""
    text = io.open(NOTES, encoding="utf-8").read()
    before = text

    text = text.replace("`<OFF>%`", f"**{off['unsupported_claim_rate']}%**")
    text = text.replace("`<ON>%` on a 50-question benchmark",
                        f"**{on['unsupported_claim_rate']}%** on a 50-question benchmark")
    text = text.replace("to `<ON>%` while", f"to **{on['correct_refusal_rate']}%** while")
    text = text.replace("at `<X>%`", f"at **{on['over_blocking_rate']}%**")

    if text == before:
        print("  ! no resume placeholders were substituted")
        return False
    io.open(NOTES, "w", encoding="utf-8").write(text)
    return True


def main() -> int:
    if not RAW.exists():
        print(f"ERROR: {RAW} not found. Run `python eval/run_eval.py` first.")
        return 1

    data = json.loads(io.open(RAW, encoding="utf-8").read())
    summaries = data.get("summaries", {})
    off, on = summaries.get("off"), summaries.get("on")

    if not (off and on):
        print("ERROR: raw_results.json is missing one of the two conditions. "
              "Run the evaluation with --condition both.")
        return 1

    print("Measured results:")
    print(f"  unsupported claim rate : {off['unsupported_claim_rate']}% -> "
          f"{on['unsupported_claim_rate']}%")
    print(f"  correct refusal rate   : {off['correct_refusal_rate']}% -> "
          f"{on['correct_refusal_rate']}%")
    print(f"  over-blocking rate     : {off['over_blocking_rate']}% -> "
          f"{on['over_blocking_rate']}%")
    print(f"  avg LLM calls/question : {off['avg_llm_calls']} -> {on['avg_llm_calls']}")
    print()

    ok = True
    if substitute(README, "<!--EVAL_TABLE_START-->", "<!--EVAL_TABLE_END-->",
                  build_table(off, on)):
        print(f"  updated {README.name}")
    else:
        ok = False

    if fill_resume_bullets(on, off):
        print(f"  updated {NOTES.name} (resume bullets)")
    else:
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
