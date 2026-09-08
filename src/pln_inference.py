"""
Probabilistic Logic Networks (PLN) Inference Operations and Evidence Provenance.

Reflected directly from Hyperon PeTTa lib_pln.metta:
- Truth_c2w (confidence to weight of evidence)
- Truth_w2c (weight of evidence to confidence)
- Truth_Revision (evidence pooling)
- Truth_Deduction (syllogism / transitive implication)
- Truth_ModusPonens (direct deduction A, A->B |- B)
- Truth_Induction (instance generalization B->A, B->C |- A->C)
- Truth_Abduction (explanatory inference A->B, C->B |- A->C and A->B, B |- A)
- Evidence Provenance tracking to avoid double counting.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
from src.beta_model import STV, BetaDistribution


def clamp(v: float, min_v: float = 0.0, max_v: float = 1.0) -> float:
    return max(min_v, min(max_v, float(v)))


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1e-12:
        return default
    return num / den


def truth_c2w(c: float) -> float:
    """Convert confidence to weight of evidence: w = c / (1 - c)."""
    c_clamped = clamp(c, 0.0, 0.999999)
    return safe_div(c_clamped, 1.0 - c_clamped, default=1e6)


def truth_w2c(w: float) -> float:
    """Convert weight of evidence to confidence: c = w / (w + 1)."""
    w_nonneg = max(0.0, float(w))
    return safe_div(w_nonneg, w_nonneg + 1.0, default=0.0)


@dataclass(frozen=True)
class PLNBelief:
    """
    PLN Uncertain Belief wrapping an STV, an underlying Beta distribution,
    and an immutable set of evidence source IDs (provenance).
    """
    stv: STV
    provenance: FrozenSet[str] = field(default_factory=frozenset)
    label: str = ""

    @property
    def strength(self) -> float:
        return self.stv.s

    @property
    def confidence(self) -> float:
        return self.stv.c

    @property
    def weight(self) -> float:
        return truth_c2w(self.confidence)

    def to_beta(self, alpha_0: float = 1.0, beta_0: float = 1.0) -> BetaDistribution:
        w = self.weight
        alpha = alpha_0 + self.strength * w
        beta = beta_0 + (1.0 - self.strength) * w
        return BetaDistribution(alpha, beta)

    @classmethod
    def from_beta(
        cls,
        beta_dist: BetaDistribution,
        provenance: Optional[FrozenSet[str]] = None,
        label: str = "",
        k: float = 1.0,
    ) -> "PLNBelief":
        s = beta_dist.mean
        effective_w = max(0.0, beta_dist.alpha + beta_dist.beta - 2.0)
        c = safe_div(effective_w, effective_w + k, 0.0)
        return cls(
            stv=STV(s, c),
            provenance=provenance or frozenset(),
            label=label,
        )

    def __repr__(self) -> str:
        prov_str = f", prov={sorted(list(self.provenance))}" if self.provenance else ""
        return f"PLNBelief({self.label}: s={self.strength:.4f}, c={self.confidence:.4f}{prov_str})"


@dataclass
class PLNInferenceTrace:
    """Auditable log entry for an executed PLN inference operation."""
    rule_name: str
    premises: List[Dict[str, Any]]
    conclusion: Dict[str, Any]
    formula_used: str
    double_counting_mitigation: str
    explanation: str


class PLNReasoner:
    """
    Execution engine for PLN operations with evidence tracking.
    """

    def __init__(self):
        self.traces: List[PLNInferenceTrace] = []

    def clear_traces(self):
        self.traces.clear()

    def revision(self, b1: PLNBelief, b2: PLNBelief, label: str = "Revised") -> Tuple[PLNBelief, PLNInferenceTrace]:
        """
        PLN Revision Rule: Combines two beliefs about the same proposition.
        Formula (lib_pln.metta):
            w1 = c2w(c1), w2 = c2w(c2)
            w = w1 + w2
            f = (w1*f1 + w2*f2) / w
            c = min(1.0, max(w2c(w), c1, c2))

        Double counting prevention:
        - If provenance sets are identical, returns b1 (idempotent, no evidence added).
        - If overlapping, only the novel fraction of b2 is weighted.
        """
        overlap = b1.provenance.intersection(b2.provenance)
        union_prov = b1.provenance.union(b2.provenance)

        if b1.provenance and b2.provenance and b1.provenance == b2.provenance:
            # 100% Duplicate evidence: strictly no-op
            trace = PLNInferenceTrace(
                rule_name="Revision",
                premises=[
                    {"label": b1.label, "stv": b1.stv.to_tuple(), "provenance": list(b1.provenance)},
                    {"label": b2.label, "stv": b2.stv.to_tuple(), "provenance": list(b2.provenance)},
                ],
                conclusion={"label": label, "stv": b1.stv.to_tuple(), "provenance": list(b1.provenance)},
                formula_used="Idempotent: Identical provenance detected, zero weight added.",
                double_counting_mitigation="Identical provenance sets; update ignored to prevent double counting.",
                explanation=f"Belief {b1.label} and {b2.label} share identical evidence sources. No revision performed.",
            )
            self.traces.append(trace)
            return b1, trace

        w1 = truth_c2w(b1.confidence)
        w2_raw = truth_c2w(b2.confidence)

        # Discount overlapping evidence proportion
        if b2.provenance and len(b2.provenance) > 0:
            novel_ratio = (len(b2.provenance) - len(overlap)) / len(b2.provenance)
            w2 = w2_raw * novel_ratio
            mitigation_msg = f"Evidence overlap detected ({len(overlap)} shared sources). b2 weight scaled by novel ratio {novel_ratio:.2f}."
        else:
            w2 = w2_raw
            mitigation_msg = "Disjoint provenance; full independent evidence pooling."

        w = w1 + w2
        if w > 0:
            f = (w1 * b1.strength + w2 * b2.strength) / w
            c = truth_w2c(w)
            c = min(1.0, max(c, b1.confidence, b2.confidence if w2 == w2_raw else 0.0))
        else:
            f = b1.strength
            c = b1.confidence

        res_belief = PLNBelief(
            stv=STV(clamp(f), clamp(c)),
            provenance=frozenset(union_prov),
            label=label,
        )

        trace = PLNInferenceTrace(
            rule_name="Revision",
            premises=[
                {"label": b1.label, "stv": b1.stv.to_tuple(), "provenance": list(b1.provenance)},
                {"label": b2.label, "stv": b2.stv.to_tuple(), "provenance": list(b2.provenance)},
            ],
            conclusion={"label": label, "stv": res_belief.stv.to_tuple(), "provenance": list(res_belief.provenance)},
            formula_used="f = (w1*f1 + w2*f2)/(w1+w2); c = max(w2c(w1+w2), c1, c2)",
            double_counting_mitigation=mitigation_msg,
            explanation=f"Revised belief from {b1.label} (w={w1:.2f}) and {b2.label} (effective w={w2:.2f}) into total weight {w:.2f}.",
        )
        self.traces.append(trace)
        return res_belief, trace

    def deduction_modus_ponens(
        self,
        premise_a: PLNBelief,
        implication_a_to_b: PLNBelief,
        label: str = "DeductionConclusion",
    ) -> Tuple[PLNBelief, PLNInferenceTrace]:
        """
        PLN Modus Ponens Deduction:
        Premises:
            A with STV(sA, cA)
            A -> B with STV(sAB, cAB)
        Formula (lib_pln.metta Truth_ModusPonens):
            s = f1 * f2 + 0.02 * (1 - f1)
            c = c1 * c2
        """
        f1 = premise_a.strength
        f2 = implication_a_to_b.strength
        c1 = premise_a.confidence
        c2 = implication_a_to_b.confidence

        s_out = (f1 * f2) + 0.02 * (1.0 - f1)
        c_out = c1 * c2

        union_prov = premise_a.provenance.union(implication_a_to_b.provenance)
        res_belief = PLNBelief(
            stv=STV(clamp(s_out), clamp(c_out)),
            provenance=frozenset(union_prov),
            label=label,
        )

        trace = PLNInferenceTrace(
            rule_name="Deduction (Modus Ponens)",
            premises=[
                {"label": premise_a.label, "stv": premise_a.stv.to_tuple(), "provenance": list(premise_a.provenance)},
                {"label": implication_a_to_b.label, "stv": implication_a_to_b.stv.to_tuple(), "provenance": list(implication_a_to_b.provenance)},
            ],
            conclusion={"label": label, "stv": res_belief.stv.to_tuple(), "provenance": list(res_belief.provenance)},
            formula_used="s = sA * sAB + 0.02 * (1 - sA); c = cA * cAB",
            double_counting_mitigation="Provenance unions input premises {A} and {A->B}. Downstream rules will not treat this as independent evidence.",
            explanation=f"Deduction applied: from '{premise_a.label}' and rule '{implication_a_to_b.label}', derived '{label}'.",
        )
        self.traces.append(trace)
        return res_belief, trace

    def deduction_syllogism(
        self,
        p_node: PLNBelief,
        q_node: PLNBelief,
        r_node: PLNBelief,
        pq_imp: PLNBelief,
        qr_imp: PLNBelief,
        label: str = "P_to_R_Deduction",
    ) -> Tuple[PLNBelief, PLNInferenceTrace]:
        """
        PLN Deduction Formula (lib_pln.metta Truth_Deduction):
        P -> Q, Q -> R |- P -> R
        Formula:
            if Qs >= 0.9999: s = Rs
            else: s = PQs * QRs + (1 - PQs) * (Rs - Qs * QRs) / (1 - Qs)
            c = min(Pc, Qc, Rc, PQc, QRc)
        """
        ps, qs, rs = p_node.strength, q_node.strength, r_node.strength
        pqs, qrs = pq_imp.strength, qr_imp.strength

        if qs >= 0.9999:
            s_out = rs
        else:
            num = (1.0 - pqs) * (rs - (qs * qrs))
            den = 1.0 - qs
            s_out = (pqs * qrs) + safe_div(num, den, 0.0)

        c_out = min(p_node.confidence, q_node.confidence, r_node.confidence, pq_imp.confidence, qr_imp.confidence)

        union_prov = frozenset().union(
            p_node.provenance, q_node.provenance, r_node.provenance, pq_imp.provenance, qr_imp.provenance
        )

        res_belief = PLNBelief(
            stv=STV(clamp(s_out), clamp(c_out)),
            provenance=union_prov,
            label=label,
        )

        trace = PLNInferenceTrace(
            rule_name="Deduction (Transitive Implication)",
            premises=[
                {"label": pq_imp.label, "stv": pq_imp.stv.to_tuple()},
                {"label": qr_imp.label, "stv": qr_imp.stv.to_tuple()},
            ],
            conclusion={"label": label, "stv": res_belief.stv.to_tuple(), "provenance": list(union_prov)},
            formula_used="s = PQs*QRs + ((1-PQs)*(Rs - Qs*QRs))/(1-Qs); c = min(Pc, Qc, Rc, PQc, QRc)",
            double_counting_mitigation="Provenance tracks all participating premises.",
            explanation=f"Transitive deduction from {pq_imp.label} and {qr_imp.label} yielding {label}.",
        )
        self.traces.append(trace)
        return res_belief, trace

    def induction(
        self,
        a_node: PLNBelief,
        b_node: PLNBelief,
        c_node: PLNBelief,
        ba_imp: PLNBelief,
        bc_imp: PLNBelief,
        label: str = "A_to_C_Induction",
    ) -> Tuple[PLNBelief, PLNInferenceTrace]:
        """
        PLN Induction Rule (lib_pln.metta Truth_Induction):
        Given evidence sample B: B -> A and B -> C, induce A -> C.
        Formula:
            s = (sBA * sBC * sB / sA) + (1 - (sBA * sB / sA)) * (sC - sB * sBC) / (1 - sB)
            c = w2c(min(cBA, cBC))
        """
        sa, sb, sc = a_node.strength, b_node.strength, c_node.strength
        sba, sbc = ba_imp.strength, bc_imp.strength

        term1 = safe_div(sba * sbc * sb, sa, 0.0)
        prob_b_given_a = safe_div(sba * sb, sa, 0.0)
        term2 = (1.0 - prob_b_given_a) * safe_div(sc - (sb * sbc), 1.0 - sb, 0.0)
        s_out = term1 + term2

        # Confidence based on minimum premise confidence converted through weight
        c_min = min(ba_imp.confidence, bc_imp.confidence)
        w_min = truth_c2w(c_min)
        c_out = truth_w2c(w_min * 0.5)  # Induction has higher epistemic uncertainty

        union_prov = frozenset().union(
            a_node.provenance, b_node.provenance, c_node.provenance, ba_imp.provenance, bc_imp.provenance
        )

        res_belief = PLNBelief(
            stv=STV(clamp(s_out), clamp(c_out)),
            provenance=union_prov,
            label=label,
        )

        trace = PLNInferenceTrace(
            rule_name="Induction",
            premises=[
                {"label": ba_imp.label, "stv": ba_imp.stv.to_tuple()},
                {"label": bc_imp.label, "stv": bc_imp.stv.to_tuple()},
            ],
            conclusion={"label": label, "stv": res_belief.stv.to_tuple(), "provenance": list(union_prov)},
            formula_used="s = (sBA*sBC*sB/sA) + (1 - sBA*sB/sA)*(sC - sB*sBC)/(1 - sB); c = w2c(0.5*c2w(min(cBA, cBC)))",
            double_counting_mitigation="Provenance records empirical sample B and implication links.",
            explanation=f"Induction from sample instances '{b_node.label}' exhibiting properties '{a_node.label}' and '{c_node.label}'.",
        )
        self.traces.append(trace)
        return res_belief, trace

    def abduction(
        self,
        a_node: PLNBelief,
        b_node: PLNBelief,
        c_node: PLNBelief,
        ab_imp: PLNBelief,
        cb_imp: PLNBelief,
        label: str = "A_to_C_Abduction",
    ) -> Tuple[PLNBelief, PLNInferenceTrace]:
        """
        PLN Abduction Rule (lib_pln.metta Truth_Abduction):
        Given A -> B and C -> B, abduce relationship A -> C.
        Formula:
            s = (sAB * sCB * sC / sB) + safe_div(sC * (1 - sAB) * (1 - sCB), 1 - sB)
            c = w2c(min(cAB, cCB))
        """
        sa, sb, sc = a_node.strength, b_node.strength, c_node.strength
        sab, scb = ab_imp.strength, cb_imp.strength

        term1 = safe_div(sab * scb * sc, sb, 0.0)
        term2 = safe_div(sc * (1.0 - sab) * (1.0 - scb), 1.0 - sb, 0.0)
        s_out = term1 + term2

        c_min = min(ab_imp.confidence, cb_imp.confidence)
        w_min = truth_c2w(c_min)
        c_out = truth_w2c(w_min * 0.4)  # Abduction represents hypothetical plausibility

        union_prov = frozenset().union(
            a_node.provenance, b_node.provenance, c_node.provenance, ab_imp.provenance, cb_imp.provenance
        )

        res_belief = PLNBelief(
            stv=STV(clamp(s_out), clamp(c_out)),
            provenance=union_prov,
            label=label,
        )

        trace = PLNInferenceTrace(
            rule_name="Abduction (Relational)",
            premises=[
                {"label": ab_imp.label, "stv": ab_imp.stv.to_tuple()},
                {"label": cb_imp.label, "stv": cb_imp.stv.to_tuple()},
            ],
            conclusion={"label": label, "stv": res_belief.stv.to_tuple(), "provenance": list(union_prov)},
            formula_used="s = (sAB*sCB*sC/sB) + (sC*(1-sAB)*(1-sCB))/(1-sB); c = w2c(0.4*c2w(min(cAB, cCB)))",
            double_counting_mitigation="Provenance tracks premises A->B, C->B and context B.",
            explanation=f"Abductive inference: {a_node.label} and {c_node.label} share consequence {b_node.label}, suggesting possible link {label}.",
        )
        self.traces.append(trace)
        return res_belief, trace

    def abduction_explanatory(
        self,
        hypothesis_prior: PLNBelief,
        observation: PLNBelief,
        hypothesis_implies_obs: PLNBelief,
        label: str = "AbductiveExplanation",
    ) -> Tuple[PLNBelief, PLNInferenceTrace]:
        """
        Abductive Explanatory Inference:
        Hypothesis H implies observation E (H -> E).
        We observe E.
        Therefore, H is supported as an explanation of E.
        By Bayes rule:
            P(H|E) = P(E|H) * P(H) / P(E)
        """
        p_h = hypothesis_prior.strength
        p_e_given_h = hypothesis_implies_obs.strength
        p_e = max(0.01, observation.strength)

        posterior_s = safe_div(p_e_given_h * p_h, p_e, p_h)
        posterior_c = min(0.95, hypothesis_prior.confidence * observation.confidence * hypothesis_implies_obs.confidence * 0.8)

        union_prov = frozenset().union(
            hypothesis_prior.provenance, observation.provenance, hypothesis_implies_obs.provenance
        )

        res_belief = PLNBelief(
            stv=STV(clamp(posterior_s), clamp(posterior_c)),
            provenance=union_prov,
            label=label,
        )

        trace = PLNInferenceTrace(
            rule_name="Abduction (Explanatory)",
            premises=[
                {"label": hypothesis_prior.label, "stv": hypothesis_prior.stv.to_tuple()},
                {"label": observation.label, "stv": observation.stv.to_tuple()},
                {"label": hypothesis_implies_obs.label, "stv": hypothesis_implies_obs.stv.to_tuple()},
            ],
            conclusion={"label": label, "stv": res_belief.stv.to_tuple(), "provenance": list(union_prov)},
            formula_used="P(H|E) = P(E|H)*P(H)/P(E); c = c_H * c_E * c_impl * 0.8",
            double_counting_mitigation="Provenance explicitly records observation IDs and hypothesis prior.",
            explanation=f"Observation '{observation.label}' provides abductive support for hypothesis '{hypothesis_prior.label}'.",
        )
        self.traces.append(trace)
        return res_belief, trace
