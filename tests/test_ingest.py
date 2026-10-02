"""
Tests for front-matter parsing, section-aware chunking and deterministic ids.

These guard the invariants the whole retrieval layer rests on:
  - a chunk never mixes two schemes or two sections;
  - chunk ids are stable, so re-ingesting upserts instead of duplicating;
  - no chunk exceeds the embedding model's token limit (anything past it is
    silently truncated, i.e. indexed in name only).
"""

from __future__ import annotations

from core.ingest import (
    MAX_CHUNK_TOKENS,
    chunk_document,
    parse_front_matter,
    slugify_section,
    split_blocks,
    split_sections,
)

# A token counter that needs no model: ~1 token per 4 characters. Good enough to
# exercise the packing logic without loading 120 MB of weights.
def fake_count(text: str) -> int:
    return max(1, len(text) // 4)


# ==========================================================================
# Front matter
# ==========================================================================


class TestFrontMatter:
    def test_scalars(self):
        document = '---\nscheme_id: pm_kisan\nname: "PM Kisan"\n---\n# Title\nbody'
        front, body = parse_front_matter(document)
        assert front["scheme_id"] == "pm_kisan"
        assert front["name"] == "PM Kisan"
        assert body.startswith("# Title")

    def test_string_list(self):
        document = '---\ntags:\n  - "Farmers"\n  - "Income Support"\n---\nbody'
        front, _ = parse_front_matter(document)
        assert front["tags"] == ["Farmers", "Income Support"]

    def test_list_of_dicts(self):
        document = (
            '---\nreference_urls:\n  - title: "Guidelines"\n'
            '    url: "https://example.gov.in/a.pdf"\n---\nbody'
        )
        front, _ = parse_front_matter(document)
        assert front["reference_urls"] == [
            {"title": "Guidelines", "url": "https://example.gov.in/a.pdf"}
        ]

    def test_no_front_matter_returns_body_unchanged(self):
        front, body = parse_front_matter("# Just a title\ntext")
        assert front == {}
        assert body == "# Just a title\ntext"

    def test_escaped_quote_in_value(self):
        front, _ = parse_front_matter('---\nname: "A \\"quoted\\" name"\n---\nx')
        assert front["name"] == 'A "quoted" name'

    def test_unterminated_front_matter_is_not_parsed(self):
        front, body = parse_front_matter("---\nscheme_id: x\nno end marker")
        assert front == {}
        assert "scheme_id" in body


# ==========================================================================
# Sections
# ==========================================================================


class TestSectionSplitting:
    DOCUMENT = (
        "# PM-KISAN\n\n"
        "## Overview\nIncome support scheme.\n\n"
        "## Benefits\nRs 6000 per year.\n\n"
        "## Eligibility\nLandholding farmers.\n"
    )

    def test_splits_on_h2(self):
        sections = split_sections(self.DOCUMENT)
        assert [title for title, _ in sections] == ["Overview", "Benefits", "Eligibility"]

    def test_h1_is_dropped(self):
        for _, text in split_sections(self.DOCUMENT):
            assert "PM-KISAN" not in text

    def test_h3_does_not_start_a_new_section(self):
        """Sub-headings belong to their parent section; only '## ' splits."""
        document = "## Benefits\n#### Upon exit\nPension of Rs 1000.\n#### On death\nSpouse gets it.\n"
        sections = split_sections(document)
        assert len(sections) == 1
        assert "Upon exit" in sections[0][1] and "On death" in sections[0][1]

    def test_empty_sections_are_dropped(self):
        sections = split_sections("## Empty\n\n## Real\nSome text.\n")
        assert [title for title, _ in sections] == ["Real"]

    def test_slugify(self):
        assert slugify_section("Documents Required") == "documents_required"
        assert slugify_section("How to Apply") == "how_to_apply"
        assert slugify_section("Overview") == "overview"


# ==========================================================================
# Blocks
# ==========================================================================


class TestBlockSplitting:
    def test_bullets_become_separate_blocks(self):
        """One bullet is usually one checkable fact, so bullets must not be
        merged into a paragraph. The '- ' marker is kept so the list structure
        survives into the chunk the LLM reads as evidence."""
        blocks = split_blocks("- Aadhaar Card\n- Landholding papers\n- Bank account\n")
        assert blocks == ["- Aadhaar Card", "- Landholding papers", "- Bank account"]

    def test_numbered_items_are_normalised_to_bullets(self):
        blocks = split_blocks("1. First item\n2. Second item\n")
        assert blocks == ["- First item", "- Second item"]

    def test_repeated_one_dot_numbering_is_normalised(self):
        """Exactly what the official API payloads contain: every ordered item is
        numbered "1.". Keeping those markers would pack chunks reading
        "1. x 1. y 1. z", which reads like a transcription error."""
        blocks = split_blocks("1. All Institutional Land holders.\n1. Income tax payers.\n")
        assert blocks == ["- All Institutional Land holders.", "- Income tax payers."]

    def test_paragraph_lines_are_joined(self):
        blocks = split_blocks("This is one\nparagraph split over lines.\n")
        assert blocks == ["This is one paragraph split over lines."]

    def test_blank_line_separates_paragraphs(self):
        blocks = split_blocks("First para.\n\nSecond para.\n")
        assert blocks == ["First para.", "Second para."]

    def test_subheading_markers_are_stripped(self):
        blocks = split_blocks("#### Objective\nTo support farmers.\n")
        assert blocks == ["Objective", "To support farmers."]


# ==========================================================================
# Chunking a real file
# ==========================================================================


class TestChunkRealDocuments:
    def test_pm_kisan_chunks_are_well_formed(self, tmp_path=None):
        from app.config import get_settings

        path = get_settings().data_dir / "pm_kisan.md"
        chunks = chunk_document(path, fake_count)

        assert chunks, "pm_kisan.md produced no chunks"
        for chunk in chunks:
            # Every chunk is attributable and citable.
            assert chunk.scheme_id == "pm_kisan"
            assert chunk.scheme_name
            assert chunk.section
            assert chunk.source_url.startswith("https://")
            assert chunk.last_verified
            # The scheme name and section are embedded in the text itself, so a
            # chunk can be found by scheme name.
            assert chunk.text.startswith(f"{chunk.scheme_name} ({chunk.section}): ")

    def test_chunk_ids_are_unique_and_deterministic(self):
        from app.config import get_settings

        path = get_settings().data_dir / "pm_kisan.md"
        first = chunk_document(path, fake_count)
        second = chunk_document(path, fake_count)

        ids = [chunk.chunk_id for chunk in first]
        assert len(ids) == len(set(ids)), "duplicate chunk ids in one document"
        # Re-running must produce byte-identical ids, or re-ingestion appends
        # duplicates instead of upserting.
        assert ids == [chunk.chunk_id for chunk in second]

    def test_chunk_id_format_is_readable(self):
        from app.config import get_settings

        chunks = chunk_document(get_settings().data_dir / "pm_kisan.md", fake_count)
        for chunk in chunks:
            scheme, section, index = chunk.chunk_id.split("#")
            assert scheme == "pm_kisan"
            assert section and section.islower()
            assert index.isdigit()

    def test_every_scheme_produces_chunks_within_the_token_budget(self):
        """
        The real check, with the real tokenizer: nothing may exceed the model's
        limit. A chunk over the limit is truncated at embedding time, so it would
        look indexed while part of it was never searchable.
        """
        from app.config import get_settings
        from core.ingest import get_embedding_model, make_token_counter

        settings = get_settings()
        count_tokens = make_token_counter(get_embedding_model(settings))

        paths = sorted(settings.data_dir.glob("*.md"))
        assert len(paths) == 15

        all_ids: set[str] = set()
        for path in paths:
            chunks = chunk_document(path, count_tokens)
            assert chunks, f"{path.name} produced no chunks"
            for chunk in chunks:
                assert count_tokens(chunk.text) <= MAX_CHUNK_TOKENS, (
                    f"{chunk.chunk_id} exceeds the {MAX_CHUNK_TOKENS}-token budget"
                )
                # Ids must be globally unique, not just per document.
                assert chunk.chunk_id not in all_ids
                all_ids.add(chunk.chunk_id)
