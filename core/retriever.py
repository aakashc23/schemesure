"""
Vector search over the indexed scheme chunks, with an evidence-strength verdict.

THE ONE IDEA HERE
-----------------
A vector search always returns something. Ask "what is the best pizza recipe?"
and you still get five scheme chunks back, just with poor scores. If you hand
those to an LLM and ask it to answer, it will write something plausible — that is
how RAG systems hallucinate even with a correct index.

So the retriever does not just return chunks, it returns a judgement:
`low_evidence=True` when even the best match is below the configured threshold.
The generator refuses outright in that case, without spending an LLM call.

WHAT THE THRESHOLD CANNOT DO (measured, see docs/DECISIONS.md)
-------------------------------------------------------------
It catches off-topic questions. It does NOT catch a question about a real but
un-indexed scheme: "what is the interest rate on Sukanya Samriddhi Yojana?"
scores as high as a correct PM-KISAN question, because it looks exactly like a
scheme question. Catching that is the guardrail's job, not the threshold's.
Two different failure modes, two different defences.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.schemas import ChunkMetadata, RetrievalResult, RetrievedChunk
from core.ingest import get_collection, get_embedding_model


class Retriever:
    """
    Wraps the embedding model and the Chroma collection.

    Constructed once (at FastAPI startup) and reused: loading the model per
    request would add seconds of latency and hundreds of MB of churn.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = get_embedding_model(self.settings)
        self.collection = get_collection(self.settings)

    # ---- introspection used by /health and /schemes ----------------------

    def count(self) -> int:
        """Number of indexed chunks."""
        try:
            return self.collection.count()
        except Exception:  # noqa: BLE001 - /health must never raise
            return 0

    # ---- the search ------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        """
        Embed `query`, fetch the nearest chunks, and judge the evidence strength.

        `query` should already be the normalized English query — raw Hinglish
        retrieves badly (measured at ~0.17 similarity against the chunk that
        actually answers it, versus ~0.52 for the same question in English).
        """
        top_k = top_k or self.settings.retrieval_top_k
        threshold = (
            self.settings.retrieval_min_score if min_score is None else min_score
        )

        query = (query or "").strip()
        if not query:
            return RetrievalResult(
                query=query, chunks=[], best_score=0.0,
                low_evidence=True, threshold=threshold,
            )

        vector = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]

        raw = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns one list per query; we only ever send one.
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = metadata or {}
            # The collection is created with hnsw:space=cosine, where Chroma
            # reports distance = 1 - cosine_similarity. Convert back so the
            # number means "similarity" (higher = closer), matching the
            # threshold in config.py.
            score = 1.0 - float(distance)
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=document or "",
                    score=round(score, 4),
                    metadata=ChunkMetadata(
                        scheme_id=str(metadata.get("scheme_id", "")),
                        scheme_name=str(metadata.get("scheme_name", "")),
                        section=str(metadata.get("section", "")),
                        source_url=str(metadata.get("source_url", "")),
                        last_verified=str(metadata.get("last_verified", "")),
                        ministry=str(metadata.get("ministry", "")),
                    ),
                )
            )

        best_score = max((chunk.score for chunk in chunks), default=0.0)

        return RetrievalResult(
            query=query,
            chunks=chunks,
            best_score=round(best_score, 4),
            # The decision the generator acts on.
            low_evidence=(not chunks) or best_score < threshold,
            threshold=threshold,
        )


# ==========================================================================
# Evidence formatting
# ==========================================================================
# Shared by the generator and the guardrail judge so both see byte-identical
# evidence. If they disagreed on what the evidence was, the judge's verdicts
# would be meaningless.


def format_evidence(chunks: list[RetrievedChunk]) -> str:
    """
    Render chunks as a numbered evidence block for a prompt.

    Each entry shows both the [n] marker (what the answer cites) and the
    chunk_id (what the judge must cite). Keeping both visible is what lets us
    verify the judge's citation in Python afterwards.
    """
    if not chunks:
        return "(no evidence passages)"

    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        blocks.append(
            f"[{index}] chunk_id: {chunk.chunk_id}\n"
            f"    scheme: {metadata.scheme_name}\n"
            f"    section: {metadata.section}\n"
            f"    text: {chunk.text}"
        )
    return "\n\n".join(blocks)
