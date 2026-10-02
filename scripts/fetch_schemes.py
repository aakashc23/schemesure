"""
Fetch scheme data from the official myScheme portal (Government of India).

WHY THIS EXISTS
---------------
myscheme.gov.in is a JavaScript single-page app, so a plain HTTP GET of the
human-facing page returns an empty shell. The page itself is rendered from a
public JSON API on the same official domain:

    https://www.myscheme.gov.in/api/apisetu/schemes?slug=<slug>&lang=en
    https://www.myscheme.gov.in/api/apisetu/schemes/<id>/documents?lang=en

We read that API instead of scraping rendered HTML. Same official source,
far more reliable, and it hands us the editors' own markdown in `*_md` fields.

HARD RULE (see CLAUDE.md): every fact written to data/schemes/ must come from a
response this script actually received. Nothing is filled in from memory. When a
section is missing from the API payload we write the literal string
"Not specified in official source" so the gap is visible instead of invented.

Run:  python scripts/fetch_schemes.py
"""

from __future__ import annotations

import html
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

API_BASE = "https://www.myscheme.gov.in/api/apisetu"
PAGE_BASE = "https://www.myscheme.gov.in/schemes"

# Project paths. This file lives in scripts/, so the repo root is one level up.
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "schemes"
RAW_DIR = ROOT / "data" / "raw"  # untouched API responses, kept for provenance

# The 15 central schemes we cover, as slug -> our own stable scheme_id.
# Every slug here was confirmed to return level == "Central" from the API.
# Chosen to span ministries and benefit types (income support, health, housing,
# insurance, pension, credit, skilling, scholarship, maternity, crop insurance).
SCHEMES: dict[str, str] = {
    "pm-kisan": "pm_kisan",
    "pmay-u": "pmay_urban",
    "pmuy": "pm_ujjwala",
    "apy": "atal_pension_yojana",
    "pmmy": "pm_mudra",
    "pmjjby": "pmjjby",
    "pmsby": "pmsby",
    "sui": "stand_up_india",
    "pm-svanidhi": "pm_svanidhi",
    "nmmss": "nmmss",
    "pmmvy": "pm_matru_vandana",
    "pmkvy-stt": "pmkvy_stt",
    "kcc": "kisan_credit_card",
    "pmfby": "pm_fasal_bima",
    "pmv": "pm_vishwakarma",
}

NOT_SPECIFIED = "Not specified in official source"

HEADERS = {
    # The API is public but rejects requests without a browser-ish User-Agent.
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SchemeSure/1.0 (educational project)",
    "Accept": "application/json",
}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def get_json(url: str, retries: int = 3) -> dict:
    """GET a URL and parse JSON, with a few polite retries on transient errors."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=45) as resp:
                # Always decode explicitly as UTF-8. The payload contains
                # curly quotes, the rupee sign and Devanagari; relying on the
                # Windows default codepage corrupts all three.
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


# --------------------------------------------------------------------------
# Rich-text -> Markdown
# --------------------------------------------------------------------------
# Some fields (notably documents_required and applicationProcess.process) come
# back as a Slate/Plate-style node tree rather than markdown, so we flatten it
# ourselves. Node shapes observed in the live payloads:
#   {"text": "...", "bold": true}            -- a leaf
#   {"type": "paragraph", "children": [...]}
#   {"type": "heading_four", "children": [...]}
#   {"type": "ul_list" | "ol_list", "children": [...]}
#   {"type": "list_item", "children": [...]}
#   {"type": "a" | "link", "url": "...", "children": [...]}

_HEADING_LEVELS = {
    "heading_one": 3,  # demoted: our own section headings already use ##
    "heading_two": 3,
    "heading_three": 4,
    "heading_four": 4,
    "heading_five": 5,
    "heading_six": 6,
}


def _inline(node: dict) -> str:
    """Render a leaf or inline node to markdown text."""
    if "text" in node:
        text = str(node["text"])
        if not text.strip():
            return text
        if node.get("bold"):
            text = f"**{text.strip()}**"
        if node.get("italic"):
            text = f"*{text.strip()}*"
        return text

    node_type = node.get("type")
    inner = "".join(_inline(c) for c in node.get("children", []))

    if node_type in ("a", "link"):
        url = str(node.get("url", "")).strip()
        return f"[{inner.strip()}]({url})" if url else inner
    return inner


def richtext_to_markdown(nodes, _depth: int = 0) -> str:
    """Flatten a list of rich-text nodes into markdown. Returns "" if empty."""
    if not nodes:
        return ""
    if isinstance(nodes, str):
        return nodes.strip()

    lines: list[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_type = node.get("type")
        children = node.get("children", [])

        if node_type in _HEADING_LEVELS:
            text = _inline(node).strip()
            if text:
                lines.append(f"\n{'#' * _HEADING_LEVELS[node_type]} {text}\n")

        elif node_type in ("ul_list", "ol_list"):
            # Render each list_item child; nested lists recurse with indent.
            for index, item in enumerate(children, start=1):
                if not isinstance(item, dict):
                    continue
                if item.get("type") in ("ul_list", "ol_list"):
                    nested = richtext_to_markdown([item], _depth + 1)
                    if nested.strip():
                        lines.append(nested)
                    continue
                bullet = "-" if node_type == "ul_list" else f"{index}."
                text = _inline(item).strip()
                if text:
                    lines.append(f"{'    ' * _depth}{bullet} {text}")

        elif node_type == "list_item":
            text = _inline(node).strip()
            if text:
                lines.append(f"{'    ' * _depth}- {text}")

        elif node_type in ("table", "tbody", "thead", "tr"):
            # Tables are rare here and their markdown equivalent is noisy;
            # flatten the cells into readable lines instead of faking a grid.
            nested = richtext_to_markdown(children, _depth)
            if nested.strip():
                lines.append(nested)

        elif node_type in ("td", "th"):
            text = _inline(node).strip()
            if text:
                lines.append(f"- {text}")

        else:
            # paragraph, blockquote, or an unknown wrapper: treat as a block.
            text = _inline(node).strip()
            if text:
                lines.append(f"\n{text}\n")
            elif children and any(
                isinstance(c, dict) and c.get("type") for c in children
            ):
                nested = richtext_to_markdown(children, _depth)
                if nested.strip():
                    lines.append(nested)

    return "\n".join(lines)


def clean_markdown(text: str | None) -> str:
    """Tidy whitespace in a markdown block without changing its wording."""
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    # The portal stores some fields HTML-escaped, and a few are escaped twice
    # ("&amp;#39;" -> "&#39;" -> "'"), so unescape until it stops changing.
    # Without this, raw entity text ends up in the chunks the LLM reads.
    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    text = text.replace(" ", " ")  # non-breaking spaces -> normal spaces
    # Collapse 3+ blank lines down to one blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces on each line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def pick_markdown(container: dict, key: str) -> str:
    """
    Prefer the API's own `<key>_md` markdown field; fall back to flattening the
    rich-text `<key>` tree. Returns NOT_SPECIFIED when both are empty.
    """
    md = clean_markdown(container.get(f"{key}_md"))
    if md:
        return md
    md = clean_markdown(richtext_to_markdown(container.get(key)))
    return md or NOT_SPECIFIED


# --------------------------------------------------------------------------
# Building one scheme's markdown file
# --------------------------------------------------------------------------


def yaml_escape(value: str) -> str:
    """Quote a scalar for YAML front-matter."""
    value = str(value).replace("\n", " ").strip()
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_application_process(entries: list) -> str:
    """Render applicationProcess (a list of {mode, process, process_md, url})."""
    if not entries:
        return NOT_SPECIFIED

    blocks: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mode = clean_markdown(entry.get("mode")) or "Application process"
        body = pick_markdown(entry, "process")
        url = str(entry.get("url") or "").strip()

        block = f"### Mode: {mode}\n"
        if url:
            block += f"\nOfficial link: {url}\n"
        block += f"\n{body}\n"
        blocks.append(block)

    return "\n".join(blocks).strip() or NOT_SPECIFIED


def build_markdown(slug: str, scheme_id: str, payload: dict, docs_payload: dict) -> str:
    """Assemble the final .md file contents for one scheme."""
    data = payload["data"]
    en = data["en"]

    basic = en.get("basicDetails", {}) or {}
    content = en.get("schemeContent", {}) or {}
    eligibility = en.get("eligibilityCriteria", {}) or {}

    def label_of(field) -> str:
        """basicDetails fields are often {"label": ..., "value": ...}."""
        if isinstance(field, dict):
            return str(field.get("label", "")).strip()
        return str(field or "").strip()

    name = clean_markdown(basic.get("schemeName")) or slug
    short_name = clean_markdown(basic.get("schemeShortTitle"))
    ministry = label_of(basic.get("nodalMinistryName")) or NOT_SPECIFIED
    department = label_of(basic.get("nodalDepartmentName"))
    level = label_of(basic.get("level"))
    scheme_type = label_of(basic.get("schemeType"))
    beneficiaries = basic.get("targetBeneficiaries") or []
    if isinstance(beneficiaries, list):
        beneficiaries = [label_of(b) for b in beneficiaries]
        beneficiaries = [b for b in beneficiaries if b]
    tags = [str(t).strip() for t in (basic.get("tags") or []) if str(t).strip()]

    source_url = f"{PAGE_BASE}/{slug}"

    # Reference links the portal itself cites (guidelines PDFs, scheme portals).
    references = []
    for ref in content.get("references") or []:
        if isinstance(ref, dict):
            # A few reference URLs in the official payload carry stray leading
            # punctuation or padding (e.g. ": https://..." and " https://... ").
            # Trimming that is a formatting fix, not a change of fact: it is what
            # makes the link resolve. The raw value stays in data/raw/ either way.
            url = str(ref.get("url", "")).strip().lstrip(":,; \t").strip()
            title = clean_markdown(ref.get("title")) or url
            if url:
                references.append((title, url))

    overview = pick_markdown(content, "detailedDescription")
    if overview == NOT_SPECIFIED:
        overview = clean_markdown(content.get("briefDescription")) or NOT_SPECIFIED

    benefits = pick_markdown(content, "benefits")
    eligibility_md = pick_markdown(eligibility, "eligibilityDescription")
    exclusions = pick_markdown(content, "exclusions")

    docs_en = (docs_payload.get("data") or {}).get("en") or {}
    documents = pick_markdown(docs_en, "documents_required")

    how_to_apply = build_application_process(en.get("applicationProcess") or [])

    # ---- front matter ----
    fm: list[str] = ["---"]
    fm.append(f"scheme_id: {scheme_id}")
    fm.append(f"name: {yaml_escape(name)}")
    if short_name:
        fm.append(f"short_name: {yaml_escape(short_name)}")
    fm.append(f"ministry: {yaml_escape(ministry)}")
    if department:
        fm.append(f"department: {yaml_escape(department)}")
    if level:
        fm.append(f"level: {yaml_escape(level)}")
    if scheme_type:
        fm.append(f"scheme_type: {yaml_escape(scheme_type)}")
    fm.append(f"source_url: {yaml_escape(source_url)}")
    fm.append(f"source_api: {yaml_escape(f'{API_BASE}/schemes?slug={slug}&lang=en')}")
    fm.append(f"last_verified: {date.today().isoformat()}")
    if beneficiaries:
        fm.append("target_beneficiaries:")
        fm.extend(f"  - {yaml_escape(b)}" for b in beneficiaries)
    if tags:
        fm.append("tags:")
        fm.extend(f"  - {yaml_escape(t)}" for t in tags)
    if references:
        fm.append("reference_urls:")
        for title, url in references:
            fm.append(f"  - title: {yaml_escape(title)}")
            fm.append(f"    url: {yaml_escape(url)}")
    fm.append("---")

    # ---- body ----
    body = [
        f"\n# {name}\n",
        "## Overview\n",
        f"{overview}\n",
        "## Benefits\n",
        f"{benefits}\n",
        "## Eligibility\n",
        f"{eligibility_md}\n",
        "## Exclusions\n",
        f"{exclusions}\n",
        "## Documents Required\n",
        f"{documents}\n",
        "## How to Apply\n",
        f"{how_to_apply}\n",
    ]

    return "\n".join(fm) + "\n".join(body)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ok, failed = [], []

    for slug, scheme_id in SCHEMES.items():
        print(f"[fetch] {slug} ... ", end="", flush=True)
        try:
            payload = get_json(f"{API_BASE}/schemes?slug={slug}&lang=en")
            if not payload.get("data"):
                print("NO DATA (skipped)")
                failed.append(slug)
                continue

            internal_id = payload["data"]["_id"]
            try:
                docs_payload = get_json(
                    f"{API_BASE}/schemes/{internal_id}/documents?lang=en"
                )
            except RuntimeError:
                docs_payload = {}

            # Keep the raw responses so any fact can be traced back later.
            raw = {"scheme": payload, "documents": docs_payload}
            with io.open(RAW_DIR / f"{scheme_id}.json", "w", encoding="utf-8") as fh:
                json.dump(raw, fh, ensure_ascii=False, indent=2)

            markdown = build_markdown(slug, scheme_id, payload, docs_payload)
            with io.open(OUT_DIR / f"{scheme_id}.md", "w", encoding="utf-8") as fh:
                fh.write(markdown)

            print(f"OK ({len(markdown):,} chars)")
            ok.append(scheme_id)
            time.sleep(0.6)  # be gentle with a public government API

        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAILED: {exc}")
            failed.append(slug)

    print(f"\nFetched {len(ok)}/{len(SCHEMES)} schemes into {OUT_DIR}")
    if failed:
        print(f"Failed/skipped: {', '.join(failed)}")
    return 0 if len(ok) >= 15 else 1


if __name__ == "__main__":
    sys.exit(main())
