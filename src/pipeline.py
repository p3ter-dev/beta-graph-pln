"""
End-to-End Orchestration Pipeline for Factor Graph + Beta/PLN Evaluation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from src.beta_model import STV, BetaDistribution, UpdateRecord
from src.context_frames import parse_context_frames
from src.correlation_weights import CandidateMetrics, analyze_all_candidates
from src.factor_graph import BetaFactorGraph, ContinuousFactor
from src.pln_inference import PLNBelief, PLNReasoner
from src.truth_table import (
    PARITY_3_DATA,
    CandidateEvaluation,
    evaluate_all_candidates,
)
from src.vector_search import FrameVectorSearchEngine


@dataclass
class CandidateSummary:
    rank: int
    candidate_id: int
    expression: str
    matches: int
    mismatches: int
    match_rate: float
    mcc: float
    weighted_matches: float
    weighted_mismatches: float
    initial_alpha: float
    initial_beta: float
    posterior_alpha: float
    posterior_beta: float
    final_belief: float
    final_variance: float
    stv: STV


class EvaluationPipeline:
    """Complete evaluation and reasoning engine."""

    def __init__(self, alpha_0: float = 1.0, beta_0: float = 1.0):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.evaluations: List[CandidateEvaluation] = []
        self.metrics: List[CandidateMetrics] = []
        self.factor_graph: Optional[BetaFactorGraph] = None
        self.pln_reasoner = PLNReasoner()
        self.candidate_summaries: List[CandidateSummary] = []
        self.update_records: List[UpdateRecord] = []

    def run_candidate_evaluation(self) -> List[CandidateEvaluation]:
        """Step 1: Programmatically evaluate all 20 candidate programs on Parity-3."""
        self.evaluations = evaluate_all_candidates()
        return self.evaluations

    def run_correlation_and_weights(self) -> List[CandidateMetrics]:
        """Step 2: Calculate MCC correlation and Hamming stratum weighted counts."""
        if not self.evaluations:
            self.run_candidate_evaluation()
        self.metrics = analyze_all_candidates(self.evaluations, self.alpha_0, self.beta_0)
        return self.metrics

    def build_factor_graph_and_propagate(self) -> BetaFactorGraph:
        """Step 3 & 4: Build bipartite factor graph and perform belief propagation."""
        if not self.metrics:
            self.run_correlation_and_weights()

        fg = BetaFactorGraph("Parity3_Candidate_Graph", self.alpha_0, self.beta_0)
        targets = [row["O"] for row in PARITY_3_DATA]

        for m, ev in zip(self.metrics, self.evaluations):
            var_name = f"Candidate_{m.candidate_id}"
            prior_dist = BetaDistribution(self.alpha_0, self.beta_0)
            fg.add_candidate_variable(var_name, prior_dist)

            # Prior factor
            fg.add_factor(
                ContinuousFactor(
                    name=f"f_prior_C{m.candidate_id}",
                    target_var=var_name,
                    factor_type="prior",
                    alpha_contrib=0.0,
                    beta_contrib=0.0,
                    meaning="Uninformative uniform prior Beta(1,1)",
                )
            )

            # Correlation factor
            fg.add_factor(
                ContinuousFactor(
                    name=f"f_corr_C{m.candidate_id}",
                    target_var=var_name,
                    factor_type="correlation",
                    alpha_contrib=m.derived_success_evidence * 0.5,
                    beta_contrib=m.derived_failure_evidence * 0.5,
                    meaning=f"Global MCC={m.mcc:+.3f} modulated evidence potential",
                )
            )

            # Row observation factors
            for r_idx, (pred, target) in enumerate(zip(ev.predictions, targets)):
                is_match = (pred == target)
                fg.add_factor(
                    ContinuousFactor(
                        name=f"f_obs_C{m.candidate_id}_r{r_idx}",
                        target_var=var_name,
                        factor_type="observation",
                        alpha_contrib=0.5 if is_match else 0.0,
                        beta_contrib=0.0 if is_match else 0.5,
                        meaning=f"Truth table row {r_idx} {'match' if is_match else 'mismatch'}",
                    )
                )

        self.factor_graph = fg
        return fg

    def compute_final_rankings(self) -> List[CandidateSummary]:
        """Compute final rankings of the 20 candidate programs."""
        if not self.factor_graph:
            self.build_factor_graph_and_propagate()

        summaries = []
        for m, ev in zip(self.metrics, self.evaluations):
            var_name = f"Candidate_{m.candidate_id}"
            post_beta = self.factor_graph.compute_variable_posterior(var_name)
            post_stv = post_beta.to_stv()

            summary = CandidateSummary(
                rank=0,  # assigned after sorting
                candidate_id=m.candidate_id,
                expression=m.expression,
                matches=m.matches,
                mismatches=m.mismatches,
                match_rate=m.match_rate,
                mcc=m.mcc,
                weighted_matches=m.weighted_matches,
                weighted_mismatches=m.weighted_mismatches,
                initial_alpha=m.initial_alpha,
                initial_beta=m.initial_beta,
                posterior_alpha=post_beta.alpha,
                posterior_beta=post_beta.beta,
                final_belief=post_beta.mean,
                final_variance=post_beta.variance,
                stv=post_stv,
            )
            summaries.append(summary)

        # Sort descending by final_belief, then MCC, then matches
        summaries.sort(key=lambda s: (s.final_belief, s.mcc, s.matches), reverse=True)
        for i, s in enumerate(summaries, 1):
            s.rank = i

        self.candidate_summaries = summaries
        return summaries

    def run_pln_demonstrations(self) -> Dict[str, Any]:
        """
        Execute concrete, auditable demonstrations of:
        1. Deduction
        2. Induction
        3. Abduction
        4. Revision
        5. Double-counting mitigation
        """
        if not self.candidate_summaries:
            self.compute_final_rankings()

        top_cand = self.candidate_summaries[0]
        runner_up = self.candidate_summaries[1]

        b_top = PLNBelief(
            stv=top_cand.stv,
            provenance=frozenset({f"C{top_cand.candidate_id}_eval"}),
            label=f"Candidate_{top_cand.candidate_id}",
        )

        b_runner = PLNBelief(
            stv=runner_up.stv,
            provenance=frozenset({f"C{runner_up.candidate_id}_eval"}),
            label=f"Candidate_{runner_up.candidate_id}",
        )

        # 1. Deduction: Modus Ponens
        # Premise: Candidate X is accurate.
        # Rule: If Candidate X is accurate, Parity Target is satisfied.
        imp_rule = PLNBelief(
            stv=STV(s=0.92, c=0.85),
            provenance=frozenset({"DomainAxiom_ParityEquivalence"}),
            label="AccurateHypothesis_Implies_TargetSatisfied",
        )
        deduction_res, trace_ded = self.pln_reasoner.deduction_modus_ponens(
            b_top, imp_rule, label="TargetOutputSatisfied"
        )

        # 2. Induction
        # Sample cases B (e.g. Even Parity subsets) show property A (C16 agreement) and C (Target agreement)
        sample_b = PLNBelief(STV(0.50, 0.90), frozenset({"EvenParityRows"}), "EvenParityRows")
        prop_a = PLNBelief(STV(0.60, 0.80), frozenset({"C16_Agreement"}), "C16_Agreement")
        prop_c = PLNBelief(STV(0.50, 0.80), frozenset({"Target_Agreement"}), "Target_Agreement")
        ba_link = PLNBelief(STV(0.75, 0.70), frozenset({"Sample_To_C16"}), "Sample_To_C16")
        bc_link = PLNBelief(STV(0.75, 0.70), frozenset({"Sample_To_Target"}), "Sample_To_Target")
        induction_res, trace_ind = self.pln_reasoner.induction(
            prop_a, sample_b, prop_c, ba_link, bc_link, label="InductiveLink_C16_To_Target"
        )

        # 3. Abduction (Explanatory)
        # Observation: Row 7 (1,1,1) produced output 1.
        # Candidate 16 predicts output 1 for Row 7.
        # Abduce: Candidate 16 is supported as the generating hypothesis.
        obs_row7 = PLNBelief(STV(1.0, 0.95), frozenset({"Obs_Row7"}), "Obs_Row7_Output1")
        hyp_implies_row7 = PLNBelief(STV(1.0, 0.90), frozenset({"C16_Row7_Derivation"}), "C16_Implies_Row7_1")
        abduction_res, trace_abd = self.pln_reasoner.abduction_explanatory(
            b_top, obs_row7, hyp_implies_row7, label="AbductiveSupport_Candidate16"
        )

        # 4. Revision (Independent evidence merging)
        # Combine top candidate belief with independent auxiliary test suite evidence
        b_aux_independent = PLNBelief(
            stv=STV(s=0.75, c=0.60),
            provenance=frozenset({"Auxiliary_Test_Suite_1"}),
            label="Auxiliary_Verification",
        )
        revised_res, trace_rev = self.pln_reasoner.revision(
            b_top, b_aux_independent, label=f"Revised_{b_top.label}"
        )

        # 5. Double-counting avoidance check (Duplicate revision)
        dup_res, trace_dup = self.pln_reasoner.revision(
            b_top, b_top, label=f"DuplicateRevision_{b_top.label}"
        )

        return {
            "deduction": {"result": deduction_res, "trace": trace_ded},
            "induction": {"result": induction_res, "trace": trace_ind},
            "abduction": {"result": abduction_res, "trace": trace_abd},
            "revision": {"result": revised_res, "trace": trace_rev},
            "duplicate_avoidance": {"result": dup_res, "trace": trace_dup},
        }

    def run_bonus_context_frames_pipeline(
        self, query: str = "Add a login page and password validation to the website", top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Bonus: End-to-End Context-Frames Semantic Search & Reasoning Pipeline.
        Flow:
            Context-Frames dataset
                    ↓
            generate embeddings
                    ↓
            vector similarity search (BEFORE Beta BP)
                    ↓
            retrieve relevant frames
                    ↓
            convert to evidence & propositions
                    ↓
            feed into Beta factor graph
                    ↓
            belief propagation
                    ↓
            PLN inference
                    ↓
            final belief
        """
        records = parse_context_frames()
        search_engine = FrameVectorSearchEngine(records)

        # 1. Vector similarity search
        search_results = search_engine.search(query, top_k=top_k)
        evidence_items = search_engine.convert_search_results_to_evidence(search_results)

        # 2. Build Context-Frame Beta Factor Graph
        frame_fg = BetaFactorGraph(name="ContextFrames_FactorGraph", alpha_0=1.0, beta_0=1.0)

        for ev in evidence_items:
            f_id = ev["frame_id"]
            var_name = f"Frame_{f_id}_Completion"
            frame_fg.add_candidate_variable(var_name)

            # Factor representing semantic relevance and status observation
            frame_fg.add_factor(
                ContinuousFactor(
                    name=f"f_sem_status_{f_id}",
                    target_var=var_name,
                    factor_type="observation",
                    alpha_contrib=ev["s_evidence"],
                    beta_contrib=ev["f_evidence"],
                    meaning=f"Status '{ev['status']}' modulated by similarity {ev['similarity']:.3f}",
                    weight=1.0,
                )
            )

        # 3. Perform Beta Belief Propagation
        posteriors = {}
        pln_beliefs = {}
        for ev in evidence_items:
            f_id = ev["frame_id"]
            var_name = f"Frame_{f_id}_Completion"
            post_dist = frame_fg.compute_variable_posterior(var_name)
            posteriors[f_id] = post_dist
            pln_beliefs[f_id] = PLNBelief.from_beta(
                post_dist,
                provenance=frozenset({f"Frame_{f_id}_SearchRetrieval"}),
                label=f_id,
            )

        # 4. PLN Relational Inference over retrieved hierarchy
        # e.g. F19 (login page) and F20 (password validation)
        pln_step = None
        if "F20" in pln_beliefs and "F19" in pln_beliefs:
            # Rule: F20 (password validation completion) -> F19 (secure login completion)
            imp_f20_f19 = PLNBelief(
                stv=STV(s=0.90, c=0.85),
                provenance=frozenset({"Axiom_SubtaskDependency_F20_F19"}),
                label="F20_Implies_SecureLogin",
            )
            derived_f19_security, trace_frame_ded = self.pln_reasoner.deduction_modus_ponens(
                pln_beliefs["F20"], imp_f20_f19, label="Derived_LoginSecurity"
            )
            pln_step = {
                "operation": "Deduction",
                "derived_belief": derived_f19_security,
                "trace": trace_frame_ded,
            }

        return {
            "query": query,
            "embedding_model": search_engine.model_name,
            "similarity_metric": search_engine.similarity_metric,
            "top_k": top_k,
            "search_results": [r.to_dict() for r in search_results],
            "evidence_extracted": evidence_items,
            "beta_posteriors": {k: v.to_dict() for k, v in posteriors.items()},
            "pln_beliefs": {k: str(v) for k, v in pln_beliefs.items()},
            "pln_relational_inference": pln_step,
        }
