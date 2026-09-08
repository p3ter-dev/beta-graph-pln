"""
Unit tests for Beta-Bernoulli conjugate model and PLN STV conversion.
"""

import pytest
from src.beta_model import BetaDistribution


def test_beta_conjugate_update():
    # Test Beta(alpha, beta) + S successes + F failures -> Beta(alpha+S, beta+F)
    prior = BetaDistribution(alpha=1.0, beta=1.0)
    assert prior.mean == 0.5

    # Update with 5 successes and 3 failures
    posterior = prior.update(s=5.0, f=3.0)
    assert posterior.alpha == 6.0
    assert posterior.beta == 4.0
    assert posterior.mean == 6.0 / 10.0
    assert round(posterior.mean, 4) == 0.6000
    assert posterior.variance < prior.variance


def test_beta_invalid_args():
    with pytest.raises(ValueError):
        BetaDistribution(alpha=-1.0, beta=1.0)

    with pytest.raises(ValueError):
        BetaDistribution(alpha=1.0, beta=0.0)

    prior = BetaDistribution(alpha=1.0, beta=1.0)
    with pytest.raises(ValueError):
        prior.update(s=-2.0, f=1.0)


def test_beta_stv_roundtrip():
    dist = BetaDistribution(alpha=5.0, beta=3.0)
    stv = dist.to_stv(k=1.0, prior_offset=2.0)
    assert 0.0 <= stv.s <= 1.0
    assert 0.0 <= stv.c <= 1.0

    # Convert back
    reconstructed = BetaDistribution.from_stv(stv, k=1.0, alpha_0=1.0, beta_0=1.0)
    assert round(reconstructed.mean, 3) == round(dist.mean, 3)
