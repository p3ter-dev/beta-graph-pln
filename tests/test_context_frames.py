"""
Unit tests for Bonus Context-Frames parsing, hierarchy, and vector similarity search.
"""

from src.context_frames import FrameKnowledgeGraph, parse_context_frames
from src.pipeline import EvaluationPipeline
from src.vector_search import FrameVectorSearchEngine


def test_context_frames_parsing():
    frames = parse_context_frames()
    assert len(frames) == 20

    # Test F3 -> F4, F5
    kg = FrameKnowledgeGraph(frames)
    assert "F4" in kg.children["F3"]
    assert "F5" in kg.children["F3"]

    # Test F17 -> F18, F19 and F19 -> F20
    assert "F18" in kg.children["F17"]
    assert "F19" in kg.children["F17"]
    assert "F20" in kg.children["F19"]


def test_vector_similarity_search():
    frames = parse_context_frames()
    engine = FrameVectorSearchEngine(frames)

    # Query regarding website password validation
    results = engine.search("password validation website login", top_k=3)
    assert len(results) == 3

    # Top retrieved frames should include F20 or F19
    retrieved_ids = [r.frame_id for r in results]
    assert "F20" in retrieved_ids or "F19" in retrieved_ids
    assert results[0].similarity_score >= results[1].similarity_score


def test_bonus_pipeline_end_to_end():
    pipe = EvaluationPipeline()
    res = pipe.run_bonus_context_frames_pipeline(query="Python loop examples", top_k=2)

    assert res["query"] == "Python loop examples"
    assert len(res["search_results"]) == 2
    assert len(res["evidence_extracted"]) == 2
    assert len(res["beta_posteriors"]) == 2
