"""
Correlation and Weighted Count Calculation for Boolean Candidates.

Mathematical Definitions:
1. Correlation:
   Matthews Correlation Coefficient (MCC) / Pearson correlation for binary vectors:
   MCC = (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
   If denominator == 0, MCC = 0.0.

2. Weighted Count:
   Inverse stratum frequency weighting over Hamming weight strata k in {0, 1, 2, 3}:
   w_r = 2.0 / C(3, k_r)
   Sum of w_r across all 8 rows = 8.0.
   W_plus  = sum_{r: pred_r == target_r} w_r
   W_minus = sum_{r: pred_r != target_r} w_r

3. Transformation to Beta Parameters:
   kappa = (1.0 + MCC) / 2.0  in [0, 1]
   S = W_plus * kappa
   F = W_minus * (1.0 - kappa)
   alpha = alpha_0 + S
   beta  = beta_0 + F
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from src.truth_table import PARITY_3_DATA, CandidateEvaluation


@dataclass
class CandidateMetrics:
    candidate_id: int
    expression: str
    matches: int
    mismatches: int
    match_rate: float
    mcc: float
    weighted_matches: float
    weighted_mismatches: float
    kappa: float
    derived_success_evidence: float
    derived_failure_evidence: float
    initial_alpha: float
    initial_beta: float
    initial_belief: float  # E[p] = alpha / (alpha + beta)
    initial_variance: float

    def to_dict(self) -> Dict:
        return {
            "candidate_id": self.candidate_id,
            "expression": self.expression,
            "matches": self.matches,
            "mismatches": self.mismatches,
            "match_rate": round(self.match_rate, 4),
            "mcc": round(self.mcc, 4),
            "weighted_matches": round(self.weighted_matches, 4),
            "weighted_mismatches": round(self.weighted_mismatches, 4),
            "kappa": round(self.kappa, 4),
            "derived_S": round(self.derived_success_evidence, 4),
            "derived_F": round(self.derived_failure_evidence, 4),
            "initial_alpha": round(self.initial_alpha, 4),
            "initial_beta": round(self.initial_beta, 4),
            "initial_belief": round(self.initial_belief, 4),
            "initial_variance": round(self.initial_variance, 6),
        }


def compute_mcc(predictions: List[int], targets: List[int]) -> float:
    """
    Compute Matthews Correlation Coefficient between binary predictions and targets.
    """
    assert len(predictions) == len(targets), "Vectors must have same length"
    tp = sum(1 for p, t in zip(predictions, targets) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(predictions, targets) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(predictions, targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, targets) if p == 0 and t == 1)

    numerator = (tp * tn) - (fp * fn)
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    if denom_sq <= 0:
        return 0.0

    return numerator / math.sqrt(denom_sq)


def compute_row_weights() -> List[float]:
    """
    Compute inverse stratum frequency weights for Parity-3 truth table rows.
    Parity-3 has 3 inputs A, B, C.
    Hamming weights:
    k=0: 1 row  -> weight = 2.0 / 1 = 2.0
    k=1: 3 rows -> weight = 2.0 / 3 = 0.6666666667
    k=2: 3 rows -> weight = 2.0 / 3 = 0.6666666667
    k=3: 1 row  -> weight = 2.0 / 1 = 2.0
    Total sum = 1*2.0 + 3*(2/3) + 3*(2/3) + 1*2.0 = 8.0
    """
    weights = []
    for row in PARITY_3_DATA:
        k = row["A"] + row["B"] + row["C"]
        comb = math.comb(3, k)
        w = 2.0 / comb
        weights.append(w)
    return weights


ROW_WEIGHTS: List[float] = compute_row_weights()


def compute_weighted_counts(predictions: List[int], targets: List[int]) -> Tuple[float, float]:
    """
    Compute weighted matches (W_plus) and weighted mismatches (W_minus).
    """
    w_plus = 0.0
    w_minus = 0.0
    for pred, target, w in zip(predictions, targets, ROW_WEIGHTS):
        if pred == target:
            w_plus += w
        else:
            w_minus += w
    return w_plus, w_minus


def analyze_candidate(
    eval_result: CandidateEvaluation,
    alpha_0: float = 1.0,
    beta_0: float = 1.0,
) -> CandidateMetrics:
    """
    Transform candidate evaluation outputs via correlation and weighted counts
    into initial Beta parameters and beliefs.
    """
    targets = [row["O"] for row in PARITY_3_DATA]
    mcc = compute_mcc(eval_result.predictions, targets)
    w_plus, w_minus = compute_weighted_counts(eval_result.predictions, targets)

    # Correlation mapping factor kappa in [0, 1]
    kappa = (1.0 + mcc) / 2.0

    # Derived evidence
    s_evidence = w_plus * kappa
    f_evidence = w_minus * (1.0 - kappa)

    alpha = alpha_0 + s_evidence
    beta = beta_0 + f_evidence

    belief = alpha / (alpha + beta)
    variance = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))

    return CandidateMetrics(
        candidate_id=eval_result.candidate_id,
        expression=eval_result.expression,
        matches=eval_result.matches,
        mismatches=eval_result.mismatches,
        match_rate=eval_result.match_rate,
        mcc=mcc,
        weighted_matches=w_plus,
        weighted_mismatches=w_minus,
        kappa=kappa,
        derived_success_evidence=s_evidence,
        derived_failure_evidence=f_evidence,
        initial_alpha=alpha,
        initial_beta=beta,
        initial_belief=belief,
        initial_variance=variance,
    )


def analyze_all_candidates(
    eval_results: List[CandidateEvaluation],
    alpha_0: float = 1.0,
    beta_0: float = 1.0,
) -> List[CandidateMetrics]:
    """Analyze all candidate evaluations."""
    return [analyze_candidate(e, alpha_0, beta_0) for e in eval_results]
