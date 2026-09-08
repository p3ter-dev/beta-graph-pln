"""
Unit tests for Candidate Program Evaluation against Parity-3.
"""

from src.correlation_weights import compute_mcc, compute_weighted_counts
from src.truth_table import (
    PARITY_3_DATA,
    evaluate_all_candidates,
)


def test_evaluate_all_candidates():
    results = evaluate_all_candidates()
    assert len(results) == 20

    targets = [r["O"] for r in PARITY_3_DATA]
    assert len(targets) == 8

    for res in results:
        assert len(res.predictions) == 8
        assert len(res.row_matches) == 8
        assert res.matches + res.mismatches == 8
        assert 0 <= res.match_rate <= 1.0
        assert res.match_rate == res.matches / 8.0


def test_correlation_and_weights():
    targets = [r["O"] for r in PARITY_3_DATA]
    results = evaluate_all_candidates()

    for res in results:
        mcc = compute_mcc(res.predictions, targets)
        assert -1.0 <= mcc <= 1.0

        w_plus, w_minus = compute_weighted_counts(res.predictions, targets)
        assert round(w_plus + w_minus, 4) == 8.0000
