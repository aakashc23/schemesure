"""
`Retriever.embed` reuses the vectors Chroma already stores for retrieved chunks.

Why it matters: the guardrail's similarity signal embeds every claim AND every
retrieved passage. Re-embedding five ~100-token passages took ~2 s per answer on
one CPU thread — longer than generating the answer. Ingest stores exactly the
embedded text, so the stored vector is the same vector; these tests pin that
down with the real model and the real index (no network, no LLM).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.retriever import Retriever


@pytest.fixture(scope="module")
def retriever():
    r = Retriever()
    if r.count() == 0:
        pytest.skip("index not built (run python -m core.ingest)")
    return r


def test_stored_vectors_match_a_fresh_encoding(retriever):
    result = retriever.search("How much money does PM-KISAN give per year?")
    texts = [chunk.text for chunk in result.chunks]

    reused = retriever.embed(texts)
    fresh = retriever.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    assert reused.shape == fresh.shape
    assert np.allclose(reused, fresh, atol=1e-5)


def test_only_unseen_texts_are_encoded(retriever, monkeypatch):
    result = retriever.search("What is the age limit for PMJJBY?")
    chunk_texts = [chunk.text for chunk in result.chunks]
    claims = ["The minimum age is 18 years.", "The maximum age is 50 years."]

    encoded: list[str] = []
    real_encode = retriever.model.encode

    def counting_encode(texts, **kwargs):
        encoded.extend(texts)
        return real_encode(texts, **kwargs)

    monkeypatch.setattr(retriever.model, "encode", counting_encode)
    vectors = retriever.embed(claims + chunk_texts)

    assert encoded == claims                      # the passages were not re-embedded
    assert vectors.shape == (len(claims) + len(chunk_texts), 384)


def test_texts_never_retrieved_are_still_embedded(retriever):
    vectors = retriever.embed(["A sentence that is not a stored chunk."])
    assert vectors.shape == (1, 384)
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-4)
