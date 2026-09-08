"""
Unit tests for Discrete Factor Graph Sum-Product BP and Beta Factor Graph.
"""

from src.factor_graph import (
    BetaFactorGraph,
    ContinuousFactor,
    DiscreteFactorGraph,
)


def test_discrete_sum_product_bp():
    # Construct a simple binary tree factor graph
    # X1 (domain [0, 1]) -- f12 -- X2 (domain [0, 1])
    fg = DiscreteFactorGraph("SimpleTree")
    x1 = fg.add_variable("X1", [0, 1])
    x2 = fg.add_variable("X2", [0, 1])

    # Factor f1 (prior on X1): favors X1=1 (0.2, 0.8)
    fg.add_factor("f1", [x1], lambda v1: 0.2 if v1 == 0 else 0.8, "Prior on X1")

    # Factor f12 (compatibility): soft equality (0.9 if equal, 0.1 if different)
    fg.add_factor(
        "f12",
        [x1, x2],
        lambda v1, v2: 0.9 if v1 == v2 else 0.1,
        "Compatibility X1 and X2",
    )

    iters = fg.run_sum_product_bp(max_iterations=10)
    assert iters >= 1

    b_x1 = fg.get_belief("X1")
    b_x2 = fg.get_belief("X2")

    # Sum of probabilities must equal 1.0
    assert round(sum(b_x1.values()), 5) == 1.00000
    assert round(sum(b_x2.values()), 5) == 1.00000

    # Since f1 favors X1=1 and f12 encourages agreement, X2=1 should be favored
    assert b_x1[1] > b_x1[0]
    assert b_x2[1] > b_x2[0]


def test_beta_factor_graph_posterior():
    fg = BetaFactorGraph("TestBetaFG", alpha_0=1.0, beta_0=1.0)
    fg.add_candidate_variable("Hyp_1")

    # Add 3 observation factors
    fg.add_factor(ContinuousFactor("f1", "Hyp_1", "obs", 1.0, 0.0, "success 1"))
    fg.add_factor(ContinuousFactor("f2", "Hyp_1", "obs", 1.0, 0.0, "success 2"))
    fg.add_factor(ContinuousFactor("f3", "Hyp_1", "obs", 0.0, 1.0, "failure 1"))

    post = fg.compute_variable_posterior("Hyp_1")
    # alpha = 1.0 + 1.0 + 1.0 = 3.0, beta = 1.0 + 1.0 = 2.0
    assert post.alpha == 3.0
    assert post.beta == 2.0
    assert post.mean == 3.0 / 5.0
