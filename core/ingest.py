"""
Turn data/schemes/*.md into an embedded, searchable Chroma collection.

THREE DESIGN DECISIONS WORTH EXPLAINING
---------------------------------------
1. SECTION-AWARE CHUNKING, NOT FIXED-SIZE
   Splitting every N characters cuts sentences in half and mixes two schemes'
   eligibility rules into one chunk. Our documents already have meaningful
   boundaries (## Benefits, ## Eligibility), so we split on those first. Every
   chunk therefore belongs to exactly one scheme and one section, which is what
   makes a precise citation ("PM-KISAN, Benefits") possible.

2. A TOKEN BUDGET, NOT A CHARACTER BUDGET
   The embedding model (paraphrase-multilingual-MiniLM-L12-v2) truncates at 128
   tokens. Anything past that is silently thrown away — text you think you
   indexed but never did, which is the worst kind of bug because retrieval just
   quietly gets worse. So we measure with the model's own tokenizer and pack
   blocks up to a budget, splitting long sections into several chunks.

3. DETERMINISTIC IDS
   A chunk's id is `<scheme_id>#<section>#<n>` (e.g. `pm_kisan#benefits#0`).
   Re-running ingestion upserts the same ids instead of appending duplicates.
   They are also human-readable, which matters: these ids are what the guardrail
   judge must cite, so a fabricated one is obvious on sight.

Run:  python -m core.ingest          (rebuild the index)
      python -m core.ingest --stats  (show what is indexed)
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings, get_settings

# Leave headroom under the model's 128-token limit: the header prefix and the
# tokenizer's special tokens also consume budget.
MAX_CHUNK_TOKENS = 110
# A chunk shorter than this carries almost no meaning on its own and only adds
# noise to the search results.
MIN_CHUNK_CHARS = 60

# Sections we never index. "How to Apply" is kept (people ask about it), but a
# section consisting only of our not-specified marker is dead weight.
NOT_SPECIFIED = "Not specified in official source"


# ==========================================================================
# Front-matter parsing
# ==========================================================================
# A minimal parser for exactly the YAML subset that scripts/fetch_schemes.py
# writes: scalars, string lists, and a list of {title, url} pairs. We control
# both the writer and the reader, so pulling in a YAML dependency would buy us
# nothing (and the project spec fixes the dependency list).


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split a document into (front_matter_dict, body). Returns ({}, text) if absent."""
    if not text.startswith("---"):
        return {}, text

    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    raw = text[3:end].strip("\n")
    body = text[end + 4 :].lstrip("\n")

    data: dict = {}
    current_list_key: str | None = None

    for line in raw.split("\n"):
        if not line.strip():
            continue

        # "  - title: ..." / "    url: ..." -> entry in a list of dicts
        indented = line.startswith((" ", "\t"))
        stripped = line.strip()

        if indented and stripped.startswith("- "):
            item = stripped[2:].strip()
            if current_list_key is None:
                continue
            if ":" in item and not item.startswith(("http", '"http')):
                key, _, value = item.partition(":")
                data[current_list_key].append({key.strip(): _unquote(value.strip())})
            else:
                data[current_list_key].append(_unquote(item))
            continue

        if indented and ":" in stripped:
            # Continuation of the last dict in the current list.
            if current_list_key and data.get(current_list_key):
                last = data[current_list_key][-1]
                if isinstance(last, dict):
                    key, _, value = stripped.partition(":")
                    last[key.strip()] = _unquote(value.strip())
            continue

        # Top-level "key: value" or "key:" starting a list.
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            if value:
                data[key] = _unquote(value)
                current_list_key = None
            else:
                data[key] = []
                current_list_key = key

    return data, body


def _unquote(value: str) -> str:
    """Strip surrounding quotes and unescape the few sequences we emit."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
        value = value.replace('\\"', '"').replace("\\\\", "\\")
    return value


# ==========================================================================
# Chunking
# ==========================================================================


@dataclass
class Chunk:
    """One indexable passage, with everything needed to cite it."""

    chunk_id: str
    text: str          # what gets embedded and shown to the LLM as evidence
    scheme_id: str
    scheme_name: str
    section: str
    source_url: str
    last_verified: str
    ministry: str


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split a document body into [(section_title, section_text)] on '## ' headings."""
    sections: list[tuple[str, str]] = []
    current_title = "Overview"
    current_lines: list[str] = []

    for line in body.split("\n"):
        # Match '## Title' but not '### Title' — only our own top-level sections.
        match = re.match(r"^##\s+(?!#)(.+?)\s*$", line)
        if match:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = match.group(1).strip()
            current_lines = []
        elif line.startswith("# "):
            continue  # the document's H1 title; the metadata already has the name
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return [(title, text) for title, text in sections if text.strip()]


def split_blocks(section_text: str) -> list[str]:
    """
    Break a section into the smallest units we are willing to keep together:
    a paragraph, a sub-heading, or a single list item.

    Splitting at list-item granularity matters because eligibility sections are
    mostly lists, and one bullet is usually one checkable fact.
    """
    blocks: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            text = " ".join(line.strip() for line in buffer if line.strip()).strip()
            if text:
                blocks.append(text)
            buffer.clear()

    for line in section_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        # Sub-headings start a new block, with the '####' markers stripped.
        if re.match(r"^#{3,6}\s+", stripped):
            flush()
            blocks.append(re.sub(r"^#{3,6}\s+", "", stripped))
            continue

        # List items each start a new block, normalised to a '- ' bullet.
        # Normalising matters: the official API numbers every ordered item "1.",
        # so keeping the original markers would pack chunks reading
        # "1. first thing 1. second thing 1. third thing", which looks like a
        # transcription error to anyone reading the evidence. The bullet is kept
        # (rather than dropped) so the list structure survives into the chunk.
        list_item = re.match(r"^(?:[-*]|\d+[.)])\s+(.*)$", stripped)
        if list_item:
            flush()
            text = list_item.group(1).strip()
            if text:
                blocks.append(f"- {text}")
            continue
        buffer.append(stripped)

    flush()
    return blocks


def split_long_block(block: str, count_tokens, budget: int) -> list[str]:
    """
    Split a single over-long block on sentence boundaries.

    Only reached for a few unusually long paragraphs (PM-KISAN's exclusions
    list, for instance). Splitting mid-sentence would produce evidence that
    reads as nonsense, so we cut at '. ' and only fall back to a hard word-level
    cut if one sentence alone blows the budget.
    """
    sentences = re.split(r"(?<=[.;:!?])\s+", block)
    pieces: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()
        if current and count_tokens(candidate) > budget:
            pieces.append(" ".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        pieces.append(" ".join(current).strip())

    # Any single sentence still over budget gets a hard word-level split.
    final: list[str] = []
    for piece in pieces:
        if count_tokens(piece) <= budget:
            final.append(piece)
            continue
        words = piece.split()
        buffer: list[str] = []
        for word in words:
            buffer.append(word)
            if count_tokens(" ".join(buffer)) > budget:
                buffer.pop()
                if buffer:
                    final.append(" ".join(buffer))
                buffer = [word]
        if buffer:
            final.append(" ".join(buffer))

    return [piece for piece in final if piece.strip()]


def slugify_section(title: str) -> str:
    """'Documents Required' -> 'documents_required', for use inside a chunk id."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug or "section"


def chunk_document(path: Path, count_tokens) -> list[Chunk]:
    """Read one scheme .md file and produce its chunks."""
    text = io.open(path, encoding="utf-8").read()
    front, body = parse_front_matter(text)

    scheme_id = front.get("scheme_id") or path.stem
    scheme_name = front.get("name") or scheme_id
    source_url = front.get("source_url", "")
    last_verified = str(front.get("last_verified", ""))
    ministry = front.get("ministry", "")

    chunks: list[Chunk] = []

    for section_title, section_text in split_sections(body):
        if section_text.strip() == NOT_SPECIFIED:
            continue  # nothing official to index

        blocks = split_blocks(section_text)
        if not blocks:
            continue

        # Every chunk carries this prefix so the scheme name and section are part
        # of what gets embedded. Without it, a chunk reading "Rs 2000 every four
        # months" matches no scheme name at all and is unfindable by name.
        prefix = f"{scheme_name} ({section_title}): "
        budget = MAX_CHUNK_TOKENS - count_tokens(prefix)
        budget = max(budget, 30)

        # Greedily pack blocks into chunks up to the token budget.
        packed: list[str] = []
        buffer: list[str] = []

        for block in blocks:
            if count_tokens(block) > budget:
                # Flush what we have, then split the oversized block alone.
                if buffer:
                    packed.append(" ".join(buffer))
                    buffer = []
                packed.extend(split_long_block(block, count_tokens, budget))
                continue

            candidate = " ".join(buffer + [block])
            if buffer and count_tokens(candidate) > budget:
                packed.append(" ".join(buffer))
                buffer = [block]
            else:
                buffer.append(block)

        if buffer:
            packed.append(" ".join(buffer))

        section_slug = slugify_section(section_title)
        for index, piece in enumerate(packed):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{scheme_id}#{section_slug}#{index}",
                    text=prefix + piece,
                    scheme_id=scheme_id,
                    scheme_name=scheme_name,
                    section=section_title,
                    source_url=source_url,
                    last_verified=last_verified,
                    ministry=ministry,
                )
            )

    return chunks


# ==========================================================================
# Embedding + Chroma
# ==========================================================================

# Module-level cache: loading the model takes a few seconds and ~120 MB, and
# FastAPI must do it exactly once at startup, not per request.
_MODEL = None


def get_embedding_model(settings: Settings | None = None):
    """Load (once) and return the sentence-transformers model."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        settings = settings or get_settings()
        _MODEL = SentenceTransformer(
            settings.embedding_model, device=settings.embedding_device
        )
    return _MODEL


def make_token_counter(model):
    """Return a function counting tokens the way this model will."""

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(model.tokenizer.encode(text, add_special_tokens=False))

    return count_tokens


def get_chroma_client(settings: Settings | None = None) -> chromadb.ClientAPI:
    """Open the persistent Chroma client at the configured path."""
    settings = settings or get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection(settings: Settings | None = None):
    """
    Get (or create) the collection.

    cosine, explicitly: Chroma's default is squared L2. Our embeddings are
    normalised, so cosine similarity is what the threshold in config.py means.
    Getting this wrong makes every score meaningless.
    """
    settings = settings or get_settings()
    client = get_chroma_client(settings)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(settings: Settings | None = None, rebuild: bool = True) -> dict:
    """
    Embed every scheme document and upsert it into Chroma.

    Returns a small summary dict so callers (and the Docker build) can assert
    that a sensible number of chunks was actually indexed.
    """
    settings = settings or get_settings()

    if rebuild and settings.chroma_dir.exists():
        # Cleanest way to drop a stale schema or a changed embedding dimension.
        shutil.rmtree(settings.chroma_dir, ignore_errors=True)

    model = get_embedding_model(settings)
    count_tokens = make_token_counter(model)
    collection = get_collection(settings)

    paths = sorted(settings.data_dir.glob("*.md"))
    if not paths:
        raise RuntimeError(f"no scheme .md files found in {settings.data_dir}")

    all_chunks: list[Chunk] = []
    per_scheme: dict[str, int] = {}
    for path in paths:
        chunks = chunk_document(path, count_tokens)
        per_scheme[path.stem] = len(chunks)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("chunking produced no chunks — check the data files")

    # Guard against duplicate ids, which would silently drop content on upsert.
    ids = [chunk.chunk_id for chunk in all_chunks]
    if len(ids) != len(set(ids)):
        duplicates = {i for i in ids if ids.count(i) > 1}
        raise RuntimeError(f"duplicate chunk ids: {sorted(duplicates)[:5]}")

    texts = [chunk.text for chunk in all_chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,  # required for cosine scores to be comparable
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=[vector.tolist() for vector in embeddings],
        metadatas=[
            {
                "scheme_id": chunk.scheme_id,
                "scheme_name": chunk.scheme_name,
                "section": chunk.section,
                "source_url": chunk.source_url,
                "last_verified": chunk.last_verified,
                "ministry": chunk.ministry,
            }
            for chunk in all_chunks
        ],
    )

    token_counts = [count_tokens(text) for text in texts]
    summary = {
        "schemes": len(paths),
        "chunks": len(all_chunks),
        "per_scheme": per_scheme,
        "max_tokens": max(token_counts),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
        "over_budget": sum(1 for count in token_counts if count > 128),
        "collection": settings.chroma_collection,
        "path": str(settings.chroma_dir),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Chroma index.")
    parser.add_argument("--stats", action="store_true", help="show the index, do not rebuild")
    parser.add_argument(
        "--no-rebuild", action="store_true", help="upsert into the existing index"
    )
    args = parser.parse_args()

    settings = get_settings()

    if args.stats:
        collection = get_collection(settings)
        print(f"collection : {settings.chroma_collection}")
        print(f"path       : {settings.chroma_dir}")
        print(f"chunks     : {collection.count()}")
        return 0

    summary = build_index(settings, rebuild=not args.no_rebuild)
    print(f"Indexed {summary['chunks']} chunks from {summary['schemes']} schemes")
    print(f"  tokens/chunk: avg {summary['avg_tokens']}, max {summary['max_tokens']}")
    print(f"  over 128-token limit: {summary['over_budget']} (must be 0)")
    print(f"  stored at: {summary['path']}")
    print("\n  chunks per scheme:")
    for scheme, count in sorted(summary["per_scheme"].items()):
        print(f"    {count:3d}  {scheme}")

    if summary["over_budget"]:
        print("\nWARNING: some chunks exceed the embedding model's limit and "
              "will be silently truncated.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
