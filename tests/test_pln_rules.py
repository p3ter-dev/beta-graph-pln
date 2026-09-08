"""
Unit tests for the four PLN Inference Operations and Double Counting Prevention.
"""

from src.beta_model import STV
from src.pln_inference import PLNBelief, PLNReasoner


def test_pln_deduction():
    reasoner = PLNReasoner()
    a = PLNBelief(STV(0.80, 0.70), frozenset({"Hyp_A"}), "A")
    a_imp_b = PLNBelief(STV(0.90, 0.85), frozenset({"Rule_AB"}), "A->B")

    conclusion, trace = reasoner.deduction_modus_ponens(a, a_imp_b, "B")
    assert trace.rule_name == "Deduction (Modus Ponens)"
    assert 0.0 <= conclusion.strength <= 1.0
    assert 0.0 <= conclusion.confidence <= 1.0
    # Provenance contains union
    assert conclusion.provenance == frozenset({"Hyp_A", "Rule_AB"})


def test_pln_induction():
    reasoner = PLNReasoner()
    sample_b = PLNBelief(STV(0.50, 0.80), frozenset({"Sample_B"}), "Sample_B")
    prop_a = PLNBelief(STV(0.60, 0.70), frozenset({"Prop_A"}), "Prop_A")
    prop_c = PLNBelief(STV(0.40, 0.70), frozenset({"Prop_C"}), "Prop_C")
    ba_link = PLNBelief(STV(0.80, 0.75), frozenset({"B_to_A"}), "B->A")
    bc_link = PLNBelief(STV(0.70, 0.75), frozenset({"B_to_C"}), "B->C")

    ind_res, trace = reasoner.induction(prop_a, sample_b, prop_c, ba_link, bc_link, "A->C")
    assert trace.rule_name == "Induction"
    assert 0.0 <= ind_res.strength <= 1.0
    assert 0.0 <= ind_res.confidence <= 1.0


def test_pln_abduction():
    reasoner = PLNReasoner()
    hyp = PLNBelief(STV(0.50, 0.60), frozenset({"Hyp"}), "Hyp")
    obs = PLNBelief(STV(0.40, 0.90), frozenset({"Obs"}), "Obs")
    hyp_imp_obs = PLNBelief(STV(0.95, 0.85), frozenset({"Hyp_imp_Obs"}), "Hyp->Obs")

    abd_res, trace = reasoner.abduction_explanatory(hyp, obs, hyp_imp_obs, "HypothesisSupported")
    assert trace.rule_name == "Abduction (Explanatory)"
    assert abd_res.strength > hyp.strength  # Explanatory evidence boosts support
    assert 0.0 <= abd_res.confidence <= 1.0


def test_pln_revision_independent():
    reasoner = PLNReasoner()
    b1 = PLNBelief(STV(0.60, 0.70), frozenset({"Evidence_1"}), "Claim")
    b2 = PLNBelief(STV(0.80, 0.60), frozenset({"Evidence_2"}), "Claim")

    revised, trace = reasoner.revision(b1, b2, "Revised_Claim")
    assert trace.rule_name == "Revision"
    assert revised.confidence > b1.confidence  # Pooling increases confidence
    assert revised.confidence > b2.confidence
    assert b1.strength < revised.strength < b2.strength


def test_pln_avoid_double_counting():
    reasoner = PLNReasoner()
    b1 = PLNBelief(STV(0.60, 0.70), frozenset({"Evidence_Shared"}), "Claim")
    b2 = PLNBelief(STV(0.60, 0.70), frozenset({"Evidence_Shared"}), "Claim")

    # Both have identical provenance
    revised, trace = reasoner.revision(b1, b2, "Claim")
    # Must be idempotent: no new weight added
    assert revised.confidence == b1.confidence
    assert revised.strength == b1.strength
    assert "Identical provenance" in trace.double_counting_mitigation
