"""
Unit test for Reproducibility requirement: running the program twice produces exact identical results.
"""

from src.pipeline import EvaluationPipeline


def test_pipeline_reproducibility():
    pipe1 = EvaluationPipeline()
    rankings1 = pipe1.compute_final_rankings()
    pln1 = pipe1.run_pln_demonstrations()

    pipe2 = EvaluationPipeline()
    rankings2 = pipe2.compute_final_rankings()
    pln2 = pipe2.run_pln_demonstrations()

    assert len(rankings1) == len(rankings2) == 20

    for s1, s2 in zip(rankings1, rankings2):
        assert s1.rank == s2.rank
        assert s1.candidate_id == s2.candidate_id
        assert s1.matches == s2.matches
        assert s1.mismatches == s2.mismatches
        assert s1.mcc == s2.mcc
        assert s1.weighted_matches == s2.weighted_matches
        assert s1.initial_alpha == s2.initial_alpha
        assert s1.initial_beta == s2.initial_beta
        assert s1.posterior_alpha == s2.posterior_alpha
        assert s1.posterior_beta == s2.posterior_beta
        assert s1.final_belief == s2.final_belief
        assert s1.stv.s == s2.stv.s
        assert s1.stv.c == s2.stv.c

    # Check PLN results
    assert pln1["deduction"]["result"].stv == pln2["deduction"]["result"].stv
    assert pln1["revision"]["result"].stv == pln2["revision"]["result"].stv
