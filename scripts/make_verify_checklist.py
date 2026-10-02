"""
Generate docs/VERIFY_CHECKLIST.md — a human spot-check sheet for the scheme data.

WHY GENERATED RATHER THAN HAND-WRITTEN: a checklist typed by hand drifts away
from the data it is supposed to check, and then it verifies nothing. This reads
the actual files, so the facts listed are exactly the facts the system will use.

What goes in, per scheme:
  - the official source URL and the date it was fetched;
  - the headline benefit sentence (the number people care about most);
  - every rule encoded in the .rules.json, paired with the official sentence it
    was derived from — these are the facts most worth checking, because they are
    the ones that decide eligibility answers.

Run:  python scripts/make_verify_checklist.py
"""

from __future__ import annotations

import io
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.ingest import parse_front_matter, split_sections  # noqa: E402

SCHEMES_DIR = ROOT / "data" / "schemes"
OUT_PATH = ROOT / "docs" / "VERIFY_CHECKLIST.md"

RULE_FIELDS = (
    "min_age", "max_age", "max_annual_income",
    "allowed_states", "allowed_occupations", "gender", "category",
)
RULE_LABELS = {
    "min_age": "Minimum age",
    "max_age": "Maximum age",
    "max_annual_income": "Maximum annual income",
    "allowed_states": "Allowed states",
    "allowed_occupations": "Allowed occupations",
    "gender": "Gender restriction",
    "category": "Category restriction",
}


def first_sentence(text: str, limit: int = 300) -> str:
    """Pull the first meaningful sentence out of a markdown section."""
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^>\s*", "", line)            # blockquote marker
        line = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", line)  # bullet or "1." marker
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)  # drop bold markers
        line = line.strip()
        # Skip lines that are really labels ("For Fresh Applications:") rather
        # than facts, so the checklist shows something worth checking.
        if len(line) < 25 or line.endswith(":"):
            continue
        return line[:limit] + ("..." if len(line) > limit else "")
    return "(no single summary sentence — check the section in the source)"


def format_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def main() -> int:
    md_paths = sorted(SCHEMES_DIR.glob("*.md"))
    if not md_paths:
        print("no scheme files found")
        return 1

    lines: list[str] = []
    lines.append("# Verification checklist\n")
    lines.append(
        "This file exists so the scheme data can be **spot-checked by a human**. "
        "Everything in SchemeSure is built on these 15 documents, so if a figure "
        "here is wrong, every answer about that scheme is wrong too.\n"
    )
    lines.append(
        "**How to use it:** open a scheme's source URL, find the facts listed below, "
        "and tick the box if they match. Each eligibility rule is shown next to the "
        "exact official sentence it was derived from, so you are comparing like with "
        "like rather than re-interpreting the page.\n"
    )
    lines.append(
        "> All data comes from the official myScheme portal "
        "(`myscheme.gov.in`, run by Digital India Corporation, Ministry of "
        "Electronics & IT), read through the public JSON API that the portal's own "
        "pages use. Raw API responses are kept in `data/raw/` so any fact can be "
        "traced back to the response it came from.\n"
    )
    lines.append(f"*Generated {date.today().isoformat()} by "
                 f"`scripts/make_verify_checklist.py`*\n")
    lines.append("---\n")

    # Two facts were cross-checked against a *different* official source than the
    # one the pipeline reads. That tests the whole chain (API -> parser ->
    # markdown -> rules), not just the parser, which is why it is recorded here.
    lines.append("## Independent cross-checks already done\n")
    lines.append(
        "Everything below comes from the myScheme API. To check that chain end to "
        "end — not just that the parser is faithful — two headline facts were "
        "verified against a **different** official source:\n"
    )
    lines.append("| Fact in this repo | Independent official source | Result |")
    lines.append("|---|---|---|")
    lines.append(
        "| PMJJBY: ₹2 lakh cover, ₹436/year premium, age 18–50 "
        "| `financialservices.gov.in/pmjjby` and the Jan Suraksha rules "
        "(Department of Financial Services) | ✅ all three match exactly |"
    )
    lines.append(
        "| PM-KISAN: ₹6,000/year in three equal instalments of ₹2,000 every four months "
        "| `services.india.gov.in` / PIB releases (Ministry of Agriculture) "
        "| ✅ matches exactly |"
    )
    lines.append("")
    lines.append(
        "That is 2 of 15 schemes. The remaining 13 are listed below for you to "
        "check the same way — the point of this file.\n"
    )
    lines.append("---\n")

    # Summary table first, so the whole set is visible at a glance.
    lines.append("## At a glance\n")
    lines.append("| # | Scheme | Ministry | Verified on |")
    lines.append("|---|--------|----------|-------------|")
    for index, path in enumerate(md_paths, start=1):
        front, _ = parse_front_matter(io.open(path, encoding="utf-8").read())
        lines.append(
            f"| {index} | [{front.get('name', path.stem)}]({front.get('source_url', '')}) "
            f"| {front.get('ministry', '?')} | {front.get('last_verified', '?')} |"
        )
    lines.append("")
    lines.append("---\n")

    for index, path in enumerate(md_paths, start=1):
        text = io.open(path, encoding="utf-8").read()
        front, body = parse_front_matter(text)
        sections = dict(split_sections(body))

        name = front.get("name", path.stem)
        short = front.get("short_name", "")
        heading = f"{index}. {name}" + (f" ({short})" if short else "")

        lines.append(f"## {heading}\n")
        lines.append(f"- **Scheme id:** `{front.get('scheme_id', path.stem)}`")
        lines.append(f"- **Ministry:** {front.get('ministry', '?')}")
        lines.append(f"- **Source URL:** <{front.get('source_url', '')}>")
        lines.append(f"- **Last verified:** {front.get('last_verified', '?')}")
        lines.append("")

        lines.append("**Check these facts:**\n")

        # Fact 1: the headline benefit.
        benefit = first_sentence(sections.get("Benefits", ""))
        lines.append(f"- [ ] **Benefit:** {benefit}")

        # Fact 2: the headline eligibility sentence.
        eligibility = first_sentence(sections.get("Eligibility", ""))
        lines.append(f"- [ ] **Eligibility:** {eligibility}")

        # Fact 3+: every encoded rule, with its official sentence.
        rules_path = SCHEMES_DIR / f"{path.stem}.rules.json"
        if rules_path.exists():
            payload = json.loads(io.open(rules_path, encoding="utf-8").read())
            sources = payload.get("rule_sources") or {}
            active = [field for field in RULE_FIELDS if payload.get(field) is not None]

            if active:
                for field in active:
                    label = RULE_LABELS[field]
                    value = format_value(payload[field])
                    official = str(sources.get(field, "")).strip()
                    lines.append(f"- [ ] **{label}: {value}**")
                    if official:
                        lines.append(f"      - official wording: *\"{official}\"*")
            else:
                lines.append(
                    "- [ ] **No age / income / state / occupation / gender / category "
                    "rule is encoded for this scheme** — confirm the official page "
                    "really states no such restriction."
                )

            conditions = payload.get("unmodelled_conditions") or []
            if conditions:
                lines.append("")
                lines.append(
                    f"<details><summary>Conditions that could not be encoded as "
                    f"rules ({len(conditions)}) — these are shown to users as "
                    f"'still to verify'</summary>\n"
                )
                for condition in conditions:
                    lines.append(f"- {condition}")
                lines.append("\n</details>")

        lines.append("")
        lines.append("---\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with io.open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"Wrote {OUT_PATH} ({len(md_paths)} schemes, {len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
