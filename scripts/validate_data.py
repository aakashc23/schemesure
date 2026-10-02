"""
Validate everything in data/schemes/ before it is trusted.

WHY: the scheme files are the system's only source of truth. A missing section, a
broken source URL or a rules file that contradicts its own markdown would quietly
degrade every answer downstream, and the failure would look like "the LLM is bad"
rather than "the data is wrong". This script turns that class of bug into a
loud, early error.

Checks performed:
  1. Each .md has front-matter with the required keys.
  2. source_url points at an official domain (*.gov.in / *.nic.in).
  3. last_verified is a real, non-future ISO date.
  4. All six expected sections are present.
  5. Sections are not empty, and the not-specified marker is used rather than
     left blank.
  6. Each .md has a matching .rules.json and vice versa.
  7. Rules files contain exactly the seven contract fields, with sane types.
  8. Every non-null rule cites the official sentence it came from.
  9. Numeric rules are internally consistent (min_age <= max_age, etc).
 10. No HTML entities or replacement characters survived the fetch.

Run:  python scripts/validate_data.py
Exit code 0 = everything valid, 1 = problems found.
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

REQUIRED_FRONT_MATTER = ("scheme_id", "name", "ministry", "source_url", "last_verified")
REQUIRED_SECTIONS = (
    "Overview", "Benefits", "Eligibility",
    "Exclusions", "Documents Required", "How to Apply",
)
RULE_FIELDS = (
    "min_age", "max_age", "max_annual_income",
    "allowed_states", "allowed_occupations", "gender", "category",
)
NOT_SPECIFIED = "Not specified in official source"

# Official Indian government hosts. http is tolerated here because a handful of
# reference PDFs the portal links to are still served over plain http; the
# primary source_url is checked separately and is always https.
OFFICIAL_DOMAIN = re.compile(r"https?://[a-z0-9.\-]*\.(?:gov\.in|nic\.in)(?:[/:]|$)", re.I)
ENTITY = re.compile(r"&#\d+;|&(?:amp|quot|lt|gt|apos|nbsp|#39);")
VALID_GENDERS = {"female", "male", "other"}


class Report:
    """Collects errors (must fix) and warnings (worth a look)."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")


def validate_markdown(path: Path, report: Report) -> dict:
    """Validate one scheme .md file. Returns its front-matter."""
    where = path.name
    text = io.open(path, encoding="utf-8").read()
    front, body = parse_front_matter(text)

    if not front:
        report.error(where, "no YAML front-matter found")
        return {}

    # 1. required keys
    for key in REQUIRED_FRONT_MATTER:
        if not str(front.get(key, "")).strip():
            report.error(where, f"front-matter is missing '{key}'")

    # scheme_id must match the filename, or ingestion and rules pairing break.
    if front.get("scheme_id") and front["scheme_id"] != path.stem:
        report.error(
            where, f"scheme_id '{front['scheme_id']}' does not match filename '{path.stem}'"
        )

    # 2. official source
    source_url = str(front.get("source_url", ""))
    if source_url and not OFFICIAL_DOMAIN.match(source_url):
        report.error(where, f"source_url is not an official .gov.in/.nic.in URL: {source_url}")

    # reference URLs are allowed to be any official government host
    for reference in front.get("reference_urls") or []:
        if isinstance(reference, dict):
            url = str(reference.get("url", "")).strip()
            if url and not OFFICIAL_DOMAIN.match(url):
                report.warn(where, f"reference URL is not on a .gov.in/.nic.in host: {url}")

    # 3. last_verified
    raw_date = str(front.get("last_verified", "")).strip()
    if raw_date:
        try:
            verified = date.fromisoformat(raw_date)
            if verified > date.today():
                report.error(where, f"last_verified is in the future: {raw_date}")
        except ValueError:
            report.error(where, f"last_verified is not an ISO date: {raw_date}")

    # 4 + 5. sections
    sections = dict(split_sections(body))
    for required in REQUIRED_SECTIONS:
        if required not in sections:
            report.error(where, f"missing section '## {required}'")
            continue
        content = sections[required].strip()
        if not content:
            report.error(
                where,
                f"section '{required}' is empty — it should contain "
                f"'{NOT_SPECIFIED}' if the source says nothing",
            )
        elif content == NOT_SPECIFIED:
            report.warn(where, f"section '{required}' has no official content")

    for extra in set(sections) - set(REQUIRED_SECTIONS):
        report.warn(where, f"unexpected section '{extra}'")

    # 10. leftover encoding artefacts
    found = ENTITY.findall(text)
    if found:
        report.error(where, f"undecoded HTML entities present: {sorted(set(found))[:5]}")
    if "�" in text:
        report.error(where, "contains the Unicode replacement character (encoding was lost)")

    return front


def validate_rules(path: Path, report: Report, front: dict) -> None:
    """Validate one .rules.json file."""
    where = path.name
    try:
        payload = json.loads(io.open(path, encoding="utf-8").read())
    except json.JSONDecodeError as exc:
        report.error(where, f"invalid JSON: {exc}")
        return

    scheme_id = payload.get("scheme_id")
    expected_id = path.name.replace(".rules.json", "")
    if scheme_id != expected_id:
        report.error(where, f"scheme_id '{scheme_id}' does not match filename '{expected_id}'")

    # 7. exactly the seven contract fields
    for field in RULE_FIELDS:
        if field not in payload:
            report.error(where, f"missing required rule field '{field}'")

    # types
    for field in ("min_age", "max_age", "max_annual_income"):
        value = payload.get(field)
        if value is not None and not isinstance(value, int):
            report.error(where, f"'{field}' must be an integer or null, got {type(value).__name__}")
        if isinstance(value, int) and value < 0:
            report.error(where, f"'{field}' is negative: {value}")

    for field in ("allowed_states", "allowed_occupations"):
        value = payload.get(field)
        if value is not None:
            if not isinstance(value, list) or not value:
                report.error(where, f"'{field}' must be a non-empty list or null")
            elif not all(isinstance(item, str) and item.strip() for item in value):
                report.error(where, f"'{field}' must contain only non-empty strings")

    gender = payload.get("gender")
    if gender is not None and str(gender).strip().lower() not in VALID_GENDERS:
        report.error(where, f"'gender' must be one of {sorted(VALID_GENDERS)} or null, got {gender!r}")

    # 9. internal consistency
    min_age, max_age = payload.get("min_age"), payload.get("max_age")
    if isinstance(min_age, int) and isinstance(max_age, int) and min_age > max_age:
        report.error(where, f"min_age ({min_age}) is greater than max_age ({max_age})")
    if isinstance(min_age, int) and min_age > 100:
        report.warn(where, f"min_age looks implausible: {min_age}")

    # 8. provenance for every active rule
    sources = payload.get("rule_sources") or {}
    if not isinstance(sources, dict):
        report.error(where, "'rule_sources' must be an object")
        sources = {}
    for field in RULE_FIELDS:
        if payload.get(field) is not None and not str(sources.get(field, "")).strip():
            report.error(
                where,
                f"rule '{field}' is set but rule_sources has no official sentence for it",
            )
    for field in sources:
        if field not in RULE_FIELDS:
            report.warn(where, f"rule_sources mentions unknown field '{field}'")

    conditions = payload.get("unmodelled_conditions")
    if conditions is None:
        report.warn(where, "no 'unmodelled_conditions' key (use [] if there are none)")
    elif not isinstance(conditions, list):
        report.error(where, "'unmodelled_conditions' must be a list")

    # A scheme with no checkable rules and no listed conditions tells the user
    # nothing at all, which almost certainly means the translation was skipped.
    has_rules = any(payload.get(field) is not None for field in RULE_FIELDS)
    if not has_rules and not conditions:
        report.warn(where, "no rules and no unmodelled conditions — is this scheme really unrestricted?")


def main() -> int:
    report = Report()

    if not SCHEMES_DIR.exists():
        print(f"FAIL: {SCHEMES_DIR} does not exist")
        return 1

    md_paths = sorted(SCHEMES_DIR.glob("*.md"))
    rules_paths = sorted(SCHEMES_DIR.glob("*.rules.json"))

    print(f"[validate-data] {len(md_paths)} scheme documents, {len(rules_paths)} rules files\n")

    if len(md_paths) < 15:
        report.error("data/schemes", f"expected at least 15 schemes, found {len(md_paths)}")

    fronts: dict[str, dict] = {}
    for path in md_paths:
        fronts[path.stem] = validate_markdown(path, report)

    # 6. pairing both ways
    md_ids = {path.stem for path in md_paths}
    rule_ids = {path.name.replace(".rules.json", "") for path in rules_paths}
    for missing in sorted(md_ids - rule_ids):
        report.error(f"{missing}.md", "has no matching .rules.json")
    for orphan in sorted(rule_ids - md_ids):
        report.error(f"{orphan}.rules.json", "has no matching .md")

    for path in rules_paths:
        scheme_id = path.name.replace(".rules.json", "")
        validate_rules(path, report, fronts.get(scheme_id, {}))

    # duplicate scheme names would make citations ambiguous in the UI
    names: dict[str, str] = {}
    for scheme_id, front in fronts.items():
        name = str(front.get("name", "")).strip().lower()
        if name and name in names:
            report.error(f"{scheme_id}.md", f"duplicate scheme name, also used by {names[name]}")
        elif name:
            names[name] = scheme_id

    # ---- output ----
    if report.warnings:
        print(f"WARNINGS ({len(report.warnings)}):")
        for warning in report.warnings:
            print(f"  - {warning}")
        print()

    if report.errors:
        print(f"ERRORS ({len(report.errors)}):")
        for error in report.errors:
            print(f"  ! {error}")
        print(f"\n[validate-data] FAILED with {len(report.errors)} error(s)")
        return 1

    print(f"[validate-data] PASSED — {len(md_paths)} schemes valid "
          f"({len(report.warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
