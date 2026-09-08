"""
Beta-Bernoulli Uncertainty Model and PLN Simple Truth Value (STV) Conversion.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class STV:
    """
    PLN Simple Truth Value: (strength, confidence)
    - strength s in [0, 1]: probability estimate
    - confidence c in [0, 1]: certainty / amount of evidence
    """
    s: float
    c: float

    def __post_init__(self):
        self.s = max(0.0, min(1.0, float(self.s)))
        self.c = max(0.0, min(1.0, float(self.c)))

    def to_tuple(self) -> Tuple[float, float]:
        return (self.s, self.c)

    def __repr__(self) -> str:
        return f"STV(s={self.s:.4f}, c={self.c:.4f})"


@dataclass
class BetaDistribution:
    """
    Beta(alpha, beta) distribution over Bernoulli probability parameter p in [0, 1].
    - alpha: accumulated positive / success evidence (+ prior offset)
    - beta: accumulated negative / failure evidence (+ prior offset)
    """
    alpha: float
    beta: float

    def __post_init__(self):
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(f"Beta parameters must be strictly positive, got alpha={self.alpha}, beta={self.beta}")

    @property
    def mean(self) -> float:
        """Expected value E[p] = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance Var[p] = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / ((total ** 2) * (total + 1.0))

    @property
    def std_dev(self) -> float:
        """Standard deviation of the parameter p."""
        return math.sqrt(self.variance)

    @property
    def mode(self) -> Optional[float]:
        """Mode of the Beta distribution (if alpha > 1 and beta > 1)."""
        if self.alpha > 1.0 and self.beta > 1.0:
            return (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        return None

    @property
    def total_evidence(self) -> float:
        return self.alpha + self.beta

    def update(self, s: float, f: float) -> "BetaDistribution":
        """
        Conjugate Beta-Bernoulli update:
        alpha' = alpha + s
        beta'  = beta + f
        """
        if s < 0 or f < 0:
            raise ValueError(f"Evidence counts cannot be negative: s={s}, f={f}")
        return BetaDistribution(self.alpha + s, self.beta + f)

    def to_stv(self, k: float = 1.0, prior_offset: float = 2.0) -> STV:
        """
        Convert Beta distribution to PLN Simple Truth Value STV(s, c).
        w = max(0, (alpha + beta) - prior_offset)
        s = alpha / (alpha + beta)
        c = w / (w + k)
        """
        s = self.mean
        effective_w = max(0.0, (self.alpha + self.beta) - prior_offset)
        c = effective_w / (effective_w + k) if (effective_w + k) > 0 else 0.0
        return STV(s, c)

    @classmethod
    def from_stv(
        cls,
        stv: STV,
        k: float = 1.0,
        prior_offset: float = 2.0,
        alpha_0: Optional[float] = None,
        beta_0: Optional[float] = None,
    ) -> "BetaDistribution":
        """
        Convert PLN Simple Truth Value STV(s, c) to Beta(alpha, beta).
        w = c / (1.0 - c) * k
        total = w + prior_offset
        alpha = s * total
        beta = (1 - s) * total
        """
        c = min(0.999999, max(0.0, stv.c))
        w = (c / (1.0 - c)) * k
        if alpha_0 is not None and beta_0 is not None:
            prior_offset = alpha_0 + beta_0
        total = max(1e-6, w + prior_offset)
        alpha = max(1e-6, stv.s * total)
        beta = max(1e-6, (1.0 - stv.s) * total)
        return cls(alpha, beta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "mean": round(self.mean, 4),
            "variance": round(self.variance, 6),
            "std_dev": round(self.std_dev, 4),
            "mode": round(self.mode, 4) if self.mode is not None else None,
        }

    def __repr__(self) -> str:
        return f"Beta(α={self.alpha:.4f}, β={self.beta:.4f}, E[p]={self.mean:.4f})"


@dataclass
class UpdateRecord:
    """Auditable log of a Beta-Bernoulli update."""
    step_name: str
    target_hypothesis: str
    previous_alpha: float
    previous_beta: float
    new_positive_evidence: float
    new_negative_evidence: float
    new_alpha: float
    new_beta: float
    resulting_belief: float
    resulting_variance: float
    resulting_stv: STV
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "target_hypothesis": self.target_hypothesis,
            "previous_alpha": round(self.previous_alpha, 4),
            "previous_beta": round(self.previous_beta, 4),
            "new_positive_evidence": round(self.new_positive_evidence, 4),
            "new_negative_evidence": round(self.new_negative_evidence, 4),
            "new_alpha": round(self.new_alpha, 4),
            "new_beta": round(self.new_beta, 4),
            "resulting_belief": round(self.resulting_belief, 4),
            "resulting_variance": round(self.resulting_variance, 6),
            "resulting_stv": self.resulting_stv.to_tuple(),
            "description": self.description,
        }
