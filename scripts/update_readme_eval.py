"""
Fill the README's evaluation table and the resume bullets from eval/raw_results.json.

WHY A SCRIPT: these are the numbers someone will quote in an interview, so they
must be the measured ones. Copying them by hand across three documents is exactly
how a figure ends up stale or transposed. This reads the machine-written results
and substitutes them, so the docs cannot drift from the run.

SAFEGUARDS (so a half-finished or mismatched run is never published)
  * Both conditions must be present AND share the same metadata (same questions,
    model, top-k, thresholds). A 6-question ON run next to a 50-question OFF run
    is refused.
  * The table always states n. A subset run is labelled PRELIMINARY, and the
    resume bullets are NOT filled from a preliminary run.

Run after `python eval/run_eval.py`:
    python scripts/update_readme_eval.py
    python scripts/update_readme_eval.py --raw eval/runs/<name>/raw_results.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.common import fmt_pct, meta_compatible  # noqa: E402

RAW = ROOT / "eval" / "raw_results.json"
README = ROOT / "README.md"
NOTES = ROOT / "docs" / "INTERVIEW_NOTES.md"


def _cell(summary: dict, rate_key: str, num_key: str, den_key: str) -> str:
    rate = summary[rate_key]
    if rate is None:
        return fmt_pct(None, summary[den_key])
    return f"{rate}% ({summary[num_key]}/{summary[den_key]})"


def _change(off: dict, on: dict, key: str, better: str) -> str:
    before, after = off[key], on[key]
    if before is None or after is None:
        return "n/a"
    delta = round(after - before, 1)
    if abs(delta) < 0.05:
        return "no change"
    improved = (delta < 0) if better == "lower" else (delta > 0)
    # Difference of two percentages = PERCENTAGE POINTS, not a relative change.
    return f"{'better' if improved else 'worse'} ({delta:+} pp)"


def check_publishable(data: dict) -> tuple[bool, str]:
    """Both conditions present and comparable? Returns (ok, reason if not)."""
    summaries = data.get("summaries", {})
    cmeta = data.get("condition_meta", {})
    if not (summaries.get("off") and summaries.get("on")):
        return False, ("raw_results.json is missing one of the two conditions. "
                       "Run the evaluation with --condition both (or the other half).")
    if "off" not in cmeta or "on" not in cmeta:
        return False, ("raw_results.json has no per-condition metadata (written by an "
                       "older harness); re-run the evaluation.")
    differing = meta_compatible(cmeta["off"], cmeta["on"])
    if differing:
        return False, ("the OFF and ON runs are not comparable (they differ in: "
                       f"{', '.join(differing)}). Re-run so both use the same setup.")
    # Same setup is not enough: both conditions must also have been SCORED on the
    # same questions, otherwise their rates are over different question sets.
    if summaries["off"].get("scored_ids") != summaries["on"].get("scored_ids"):
        return False, ("the OFF and ON runs were scored on different question sets "
                       "(infrastructure failures differ). Re-run with --resume so both "
                       "conditions finish the same questions.")
    return True, ""


def build_table(off: dict, on: dict, meta: dict) -> str:
    """The headline comparison, as markdown, with n and the preliminary label."""
    rows = [
        ("**Unsupported claim rate** (in delivered answers)", "unsupported_claim_rate",
         "delivered_unsupported", "delivered_claims", "lower"),
        ("Answers containing any unsupported claim", "answers_with_unsupported_claim_rate",
         "answers_with_unsupported_claim", "answers_audited", "lower"),
        ("Correct refusal rate (unanswerable)", "correct_refusal_rate",
         "correct_refusals", "refusal_unanswerable_n", "higher"),
        ("Over-blocking rate (answerable refused)", "over_blocking_rate",
         "over_blocked", "refusal_answerable_n", "lower"),
        ("End-to-end fact accuracy (refusal = 0)", "end_to_end_fact_accuracy",
         "end_to_end_facts_found", "end_to_end_facts_total", "higher"),
        ("Key-fact coverage (answered only)", "key_fact_coverage",
         "key_facts_found", "key_facts_total", "higher"),
        (f"Retrieval hit rate (top-{meta['top_k']})", "retrieval_hit_rate",
         "retrieval_hits", "retrieval_n", "higher"),
    ]
    lines = []
    if meta.get("subset"):
        lines.append(f"**PRELIMINARY: n={meta['n_questions']} of {meta['n_golden']} "
                     "golden-set questions (a subset). Not the full benchmark.**\n")
    lines += [
        "| Metric | Guardrail OFF | Guardrail ON | Change |",
        "|---|---|---|---|",
    ]
    for label, rate_key, num_key, den_key, better in rows:
        lines.append(f"| {label} | {_cell(off, rate_key, num_key, den_key)} | "
                     f"{_cell(on, rate_key, num_key, den_key)} | "
                     f"{_change(off, on, rate_key, better)} |")
    if off["avg_llm_calls"] is not None and on["avg_llm_calls"] is not None:
        lines.append(f"| Avg LLM calls / question | {off['avg_llm_calls']} | "
                     f"{on['avg_llm_calls']} | "
                     f"{on['avg_llm_calls'] - off['avg_llm_calls']:+.2f} |")
    if off["median_latency_ms"] is not None and on["median_latency_ms"] is not None:
        lines.append(f"| Median latency | {off['median_latency_ms'] / 1000:.1f}s | "
                     f"{on['median_latency_ms'] / 1000:.1f}s | "
                     f"{(on['median_latency_ms'] - off['median_latency_ms']) / 1000:+.1f}s |")
    lines += [
        "",
        f"*n={meta['n_questions']} questions per condition "
        f"({meta['answerable']} answerable, {meta['unanswerable']} unanswerable); "
        f"scored on the same {off['scored']} questions in both (infrastructure "
        f"failures excluded: OFF {off['infra_failures']}, ON {on['infra_failures']}; "
        "a question that failed in either condition is dropped from both). The ON "
        "unsupported-claim figure comes from the same judge that approves answers, so "
        f"it is not an independent measurement. Full method and per-category breakdown in "
        f"[`eval/results.md`](eval/results.md).*",
    ]
    return "\n".join(lines)


def substitute(path: Path, start: str, end: str, body: str) -> bool:
    """Replace the content between two HTML-comment markers."""
    text = io.open(path, encoding="utf-8").read()
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        print(f"  ! markers {start} / {end} not found in {path.name}")
        return False
    replaced = pattern.sub(lambda _m: f"{start}\n{body}\n{end}", text)
    io.open(path, "w", encoding="utf-8").write(replaced)
    return True


def fill_resume_bullets(on: dict, off: dict, meta: dict) -> bool:
    """Replace the <OFF>/<ON>/<X> placeholders in the resume bullets."""
    needed = (off["unsupported_claim_rate"], on["unsupported_claim_rate"],
              off["correct_refusal_rate"], on["correct_refusal_rate"],
              on["over_blocking_rate"], off["avg_llm_calls"], on["avg_llm_calls"])
    if any(value is None for value in needed):
        print("  ! a resume figure has an empty denominator (n/a); not filling bullets")
        return False
    # The bullets say "cut" and "raising". Never publish those words when the
    # measured change went the other way or did not happen.
    if not on["unsupported_claim_rate"] < off["unsupported_claim_rate"]:
        print("  ! the guardrail did not lower the unsupported-claim rate; the 'cut' "
              "bullet would be false, so the bullets are NOT filled")
        return False
    if not on["correct_refusal_rate"] > off["correct_refusal_rate"]:
        print("  ! the guardrail did not raise the correct-refusal rate; the 'raising' "
              "bullet would be false, so the bullets are NOT filled")
        return False

    text = io.open(NOTES, encoding="utf-8").read()
    before = text
    n = meta["n_questions"]

    text = text.replace("`<OFF>%`", f"**{off['unsupported_claim_rate']}%**")
    text = text.replace("`<ON>%` on a 50-question benchmark",
                        f"**{on['unsupported_claim_rate']}%** on a {n}-question benchmark "
                        "(measured with the same LLM judge that approves answers)")
    text = text.replace("questions to `<ON>%` while",
                        f"questions from **{off['correct_refusal_rate']}%** to "
                        f"**{on['correct_refusal_rate']}%** while")
    text = text.replace("at `<X>%`", f"at **{on['over_blocking_rate']}%**")
    # The call cost is measured, not a constant: fill it from the run.
    text = text.replace("at a measured cost of 4 LLM calls per answer",
                        f"at a measured cost of {on['avg_llm_calls']} LLM calls per "
                        f"question on average (vs {off['avg_llm_calls']} without it)")

    if text == before:
        print("  ! no resume placeholders were substituted")
        return False
    io.open(NOTES, "w", encoding="utf-8").write(text)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish eval numbers into the docs.")
    parser.add_argument("--raw", default=str(RAW), help="path to raw_results.json")
    args = parser.parse_args()
    raw_path = Path(args.raw)

    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run `python eval/run_eval.py` first.")
        return 1

    data = json.loads(io.open(raw_path, encoding="utf-8").read())
    ok, reason = check_publishable(data)
    if not ok:
        print(f"ERROR: not publishing: {reason}")
        return 1

    off, on = data["summaries"]["off"], data["summaries"]["on"]
    meta = data["meta"]
    preliminary = bool(meta.get("subset"))

    print("Measured results" + (" (PRELIMINARY subset)" if preliminary else "") + ":")
    print(f"  questions              : n={meta['n_questions']} of {meta['n_golden']}")
    print(f"  unsupported claim rate : {fmt_pct(off['unsupported_claim_rate'], off['delivered_claims'])}"
          f" -> {fmt_pct(on['unsupported_claim_rate'], on['delivered_claims'])}")
    print(f"  correct refusal rate   : {fmt_pct(off['correct_refusal_rate'], off['refusal_unanswerable_n'])}"
          f" -> {fmt_pct(on['correct_refusal_rate'], on['refusal_unanswerable_n'])}")
    print(f"  over-blocking rate     : {fmt_pct(off['over_blocking_rate'], off['refusal_answerable_n'])}"
          f" -> {fmt_pct(on['over_blocking_rate'], on['refusal_answerable_n'])}")
    print()

    ok_all = True
    if substitute(README, "<!--EVAL_TABLE_START-->", "<!--EVAL_TABLE_END-->",
                  build_table(off, on, meta)):
        print(f"  updated {README.name}")
    else:
        ok_all = False

    if preliminary:
        print("  skipped resume bullets: a preliminary subset run is not quotable")
    elif fill_resume_bullets(on, off, meta):
        print(f"  updated {NOTES.name} (resume bullets)")
    else:
        ok_all = False

    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
