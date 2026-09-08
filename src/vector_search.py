"""
Vector Similarity Search for Context-Frames.

Retrieves top-k relevant Context-Frames based on semantic similarity to a query,
and converts retrieved frames into evidence and proposition nodes for the Beta factor graph.
"""

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.context_frames import FrameRecord


@dataclass
class SearchResult:
    frame_id: str
    record: FrameRecord
    similarity_score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "frame_id": self.frame_id,
            "similarity_score": round(self.similarity_score, 4),
            "status": self.record.status,
            "priority": self.record.priority,
            "deliverable": self.record.deliverable,
            "results": self.record.results,
        }


class FrameVectorSearchEngine:
    """
    Semantic vector search engine over Context-Frames.
    Uses TF-IDF + Character/Word N-Gram Semantic Embedding with Cosine Similarity.
    100% reproducible and offline-capable without external API keys.
    """

    def __init__(self, frames: List[FrameRecord], model_name: str = "TF-IDF N-Gram Vectorizer"):
        self.frames = frames
        self.model_name = model_name
        self.similarity_metric = "Cosine Similarity"

        # Corpus of documents
        self.documents = [f.full_text for f in frames]

        # Vectorizer: word and char-wb ngrams for rich semantic and lexical capture
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer="word",
            lowercase=True,
            stop_words="english",
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.documents)

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Perform vector similarity search for query and return top-k matches.
        """
        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.doc_matrix).flatten()

        # Sort indices descending
        sorted_indices = np.argsort(-sim_scores)

        results = []
        for rank, idx in enumerate(sorted_indices[:top_k], 1):
            score = float(sim_scores[idx])
            results.append(
                SearchResult(
                    frame_id=self.frames[idx].frame_id,
                    record=self.frames[idx],
                    similarity_score=score,
                    rank=rank,
                )
            )
        return results

    def convert_search_results_to_evidence(
        self, search_results: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        """
        Convert retrieved frames into concrete evidence for the Beta factor graph.
        - Semantic relevance score modulates the evidence weight.
        - Status provides success / failure counts.
        """
        evidence_list = []
        for res in search_results:
            rec = res.record
            # Base evidence from status
            if rec.status == "completed":
                s_base, f_base = 2.5, 0.2
            elif rec.status == "active":
                s_base, f_base = 1.0, 1.0
            else:  # pending
                s_base, f_base = 0.1, 2.0

            # Modulate by cosine similarity score
            effective_weight = max(0.1, res.similarity_score)
            s_eff = s_base * effective_weight
            f_eff = f_base * effective_weight

            evidence_list.append({
                "frame_id": rec.frame_id,
                "parent_id": rec.parent_id,
                "similarity": res.similarity_score,
                "s_evidence": s_eff,
                "f_evidence": f_eff,
                "effective_weight": effective_weight,
                "status": rec.status,
                "deliverable": rec.deliverable,
            })
        return evidence_list
