import numpy as np

from src import config
from src.chunking import split_text
from src.embeddings import build_cosine_index, embed_texts


class EvidenceError(Exception):
    """Raised when there is no readable text to build evidence from."""


class EvidenceStore:
    """All source text, chunked and indexed in an in-memory FAISS cosine index.

    Used three ways: retrieve passages per sub-question for the Writer (`gather`),
    and pull the passages that back a specific claim for the Checker (`support_for`).
    """

    def __init__(self, sources: list, embedder=None):
        self.sources = sources
        self.embedder = embedder
        self.records = []
        for source_idx, source in enumerate(sources):
            for piece in split_text(source["content"], config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                self.records.append({"id": len(self.records), "text": piece, "source_idx": source_idx})
        if not self.records:
            raise EvidenceError("The sources contained no readable text.")

        self.vectors = embed_texts([r["text"] for r in self.records], embedder)
        self.index = build_cosine_index(self.vectors)
        self.numbering = {}   # source_idx -> citation number, assigned by gather()
        self.selected = []

    def _search(self, query: str, k: int):
        query_vector = embed_texts([query], self.embedder)
        scores, ids = self.index.search(query_vector, min(len(self.records), k))
        return [(self.records[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def gather(self, questions: list, chunks_per_question: int) -> tuple:
        """Selects the best passages for each sub-question; returns (selected_chunks, numbered_sources)."""
        chosen, chosen_ids, per_source_total = [], set(), {}

        for q_idx, question in enumerate(questions):
            kept, per_source_q = 0, {}
            for record, score in self._search(question, chunks_per_question * 6):
                if kept >= chunks_per_question or len(chosen) >= config.MAX_EVIDENCE_CHUNKS:
                    break
                sid = record["source_idx"]
                if per_source_q.get(sid, 0) >= config.MAX_PER_SOURCE_PER_QUESTION:
                    continue
                if record["id"] in chosen_ids:
                    # Already picked for an earlier question: it still covers this one, so count it
                    # instead of padding this question with weaker passages.
                    if score >= config.MIN_SCORE:
                        per_source_q[sid] = per_source_q.get(sid, 0) + 1
                        kept += 1
                    continue
                if per_source_total.get(sid, 0) >= config.MAX_CHUNKS_PER_SOURCE:
                    continue
                if score < config.MIN_SCORE and kept >= config.MIN_KEEP_PER_QUESTION:
                    continue
                chosen_ids.add(record["id"])
                chosen.append({**record, "score": score, "question_idx": q_idx})
                per_source_q[sid] = per_source_q.get(sid, 0) + 1
                per_source_total[sid] = per_source_total.get(sid, 0) + 1
                kept += 1

        if not chosen:
            raise EvidenceError("No relevant passages were found in the sources.")

        # Number sources by first appearance so citations read [1], [2], [3]... in order
        self.numbering = {}
        for chunk in chosen:
            self.numbering.setdefault(chunk["source_idx"], len(self.numbering) + 1)
        for chunk in chosen:
            chunk["cite"] = self.numbering[chunk["source_idx"]]
        self.selected = chosen

        numbered = [
            {
                "n": n,
                "title": self.sources[sid]["title"],
                "url": self.sources[sid].get("url", ""),
                "kind": self.sources[sid].get("kind", "web"),
            }
            for sid, n in sorted(self.numbering.items(), key=lambda kv: kv[1])
        ]
        return chosen, numbered

    def support_for(self, claim: str, cites: list, k: int = 2) -> list:
        """The k passages from the cited sources that are closest to a claim (what the Checker reads)."""
        wanted = {sid for sid, n in self.numbering.items() if n in set(cites)}
        candidates = [r["id"] for r in self.records if r["source_idx"] in wanted]
        if not candidates:
            return []
        claim_vector = embed_texts([claim], self.embedder)[0]
        sims = self.vectors[candidates] @ claim_vector
        best = np.argsort(-sims)[:k]
        return [self.records[candidates[i]]["text"] for i in best]
