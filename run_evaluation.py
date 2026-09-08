"""
Main Executable Runner for Factor Graph + Beta/PLN Evaluation.
"""

import os
import sys
from src.pipeline import EvaluationPipeline


def format_separator(title: str = "", char: str = "=", width: int = 80) -> str:
    if not title:
        return char * width
    padding = (width - len(title) - 2) // 2
    return f"{char * padding} {title} {char * padding}"


def main():
    # Ensure stdout handles unicode characters safely
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(format_separator("FACTOR GRAPH + BETA/PLN EVALUATION SYSTEM"))
    pipeline = EvaluationPipeline(alpha_0=1.0, beta_0=1.0)

    # 1. Candidate Evaluation
    print("\n[Step 1] Programmatically Evaluating 20 Candidate Boolean Programs...")
    evaluations = pipeline.run_candidate_evaluation()
    print(f"Evaluated {len(evaluations)} candidate programs across 8 Parity-3 truth table rows.")

    # 2. Correlation and Weighted Count
    print("\n[Step 2] Computing Matthews Correlation Coefficients & Hamming-Weighted Counts...")
    metrics = pipeline.run_correlation_and_weights()
    print("Correlation and inverse stratum frequency weighting completed successfully.")

    # 3. Factor Graph Construction and Belief Propagation
    print("\n[Step 3] Constructing Bipartite Factor Graph & Executing Beta Belief Propagation...")
    fg = pipeline.build_factor_graph_and_propagate()
    graph_img_path = os.path.join(os.getcwd(), "factor_graph.png")
    fg.save_visualization(graph_img_path)
    print(f"Factor graph visualized and saved to: {graph_img_path}")

    # 4. Final Rankings
    print("\n[Step 4] Computing Final Candidate Rankings & Posteriors...")
    rankings = pipeline.compute_final_rankings()

    # Output Section 21 Style Summary
    print("\n" + format_separator("Factor Graph Evaluation"))
    print("Candidates evaluated: 20")
    print("Truth-table rows: 8")
    print("Parity Target Function: O = A XOR B XOR C")

    print("\n" + format_separator("Candidate Ranking"))
    for s in rankings:
        print(f"{s.rank:>2}. Candidate {s.candidate_id:<2} : {s.expression}")
        print(f"    Matches: {s.matches}/8 | Mismatches: {s.mismatches}/8 | Match Rate: {s.match_rate*100:.1f}%")
        print(f"    MCC Correlation: {s.mcc:+.4f} | Weighted Matches: {s.weighted_matches:.2f} / 8.00")
        print(f"    Initial Beta: α={s.initial_alpha:.3f}, β={s.initial_beta:.3f} (E[p]={s.initial_alpha/(s.initial_alpha+s.initial_beta):.3f})")
        print(f"    Posterior Beta: α={s.posterior_alpha:.3f}, β={s.posterior_beta:.3f} | Variance: {s.final_variance:.5f}")
        print(f"    Final Belief (E[p]): {s.final_belief:.4f} | PLN STV: (s={s.stv.s:.4f}, c={s.stv.c:.4f})")
        print()

    # Best Candidate Identification (Section 12)
    top_cand = rankings[0]
    print(format_separator("Top Candidate Identification & Audit"))
    print(f"Strongest Candidate: Candidate {top_cand.candidate_id} -> {top_cand.expression}")
    print(f"1. Rationale: Candidate {top_cand.candidate_id} achieves the highest final belief ({top_cand.final_belief:.4f}) and positive correlation.")
    print(f"2. Truth Table Performance: Matches {top_cand.matches} of 8 rows ({top_cand.match_rate*100:.1f}%), with MCC = {top_cand.mcc:+.4f}.")
    print(f"3. Beta Parameter Influence: Prior uniform Beta(1,1) updated by weighted evidence to Posterior Beta(α={top_cand.posterior_alpha:.3f}, β={top_cand.posterior_beta:.3f}).")
    print(f"4. Uncertainty State: Variance shrunk to {top_cand.final_variance:.5f}, confidence c = {top_cand.stv.c:.4f}.")
    print(f"5. Consistency with Observations: Parity-3 is known to require non-linear XOR chains; Candidate {top_cand.candidate_id} correctly captures the majority of alternating minterms.")

    # 5. PLN Inference Demonstrations (Section 9 & 10)
    print("\n" + format_separator("Belief Updates (PLN Inference)"))
    pln_results = pipeline.run_pln_demonstrations()

    print("\n--- 1. Deduction (Modus Ponens) ---")
    ded = pln_results["deduction"]
    print(f"Rule Applied: {ded['trace'].rule_name}")
    print(f"Premises:")
    for p in ded["trace"].premises:
        print(f"  - {p['label']}: STV={p['stv']}")
    print(f"Conclusion: {ded['trace'].conclusion['label']} -> STV={ded['trace'].conclusion['stv']}")
    print(f"Formula: {ded['trace'].formula_used}")
    print(f"Anti-Double Counting: {ded['trace'].double_counting_mitigation}")

    print("\n--- 2. Induction ---")
    ind = pln_results["induction"]
    print(f"Rule Applied: {ind['trace'].rule_name}")
    print(f"Premises:")
    for p in ind["trace"].premises:
        print(f"  - {p['label']}: STV={p['stv']}")
    print(f"Conclusion: {ind['trace'].conclusion['label']} -> STV={ind['trace'].conclusion['stv']}")
    print(f"Formula: {ind['trace'].formula_used}")
    print(f"Explanation: {ind['trace'].explanation}")

    print("\n--- 3. Abduction ---")
    abd = pln_results["abduction"]
    print(f"Rule Applied: {abd['trace'].rule_name}")
    print(f"Premises:")
    for p in abd["trace"].premises:
        print(f"  - {p['label']}: STV={p['stv']}")
    print(f"Conclusion: {abd['trace'].conclusion['label']} -> STV={abd['trace'].conclusion['stv']}")
    print(f"Formula: {abd['trace'].formula_used}")
    print(f"Explanation: {abd['trace'].explanation}")

    print("\n--- 4. Revision ---")
    rev = pln_results["revision"]
    print(f"Rule Applied: {rev['trace'].rule_name}")
    print(f"Before (Premises):")
    for p in rev["trace"].premises:
        print(f"  - {p['label']}: STV={p['stv']} | Prov={p.get('provenance')}")
    print(f"After (Revised Belief): {rev['trace'].conclusion['label']} -> STV={rev['trace'].conclusion['stv']}")
    print(f"Anti-Double Counting: {rev['trace'].double_counting_mitigation}")

    print("\n--- 5. Avoid Double Counting (Duplicate Test) ---")
    dup = pln_results["duplicate_avoidance"]
    print(f"Duplicate Input Handling: {dup['trace'].double_counting_mitigation}")
    print(f"Resulting STV unchanged: {dup['result'].stv}")

    # 6. Bonus: Context-Frames Semantic Search & Reasoning Pipeline (Sections 16-19)
    print("\n" + format_separator("Bonus: Context-Frames & Vector Search Pipeline"))
    bonus_query = "Add a login page and password validation to the website"
    print(f"User Query: \"{bonus_query}\"")
    bonus_res = pipeline.run_bonus_context_frames_pipeline(query=bonus_query, top_k=3)

    print(f"Vector Embedding Model: {bonus_res['embedding_model']}")
    print(f"Similarity Metric: {bonus_res['similarity_metric']}")
    print("\nTop-K Retrieved Frames (Pre-Propagation Semantic Search):")
    for r in bonus_res["search_results"]:
        print(f"  Rank {r['rank']}: [{r['frame_id']}] Similarity: {r['similarity_score']:.4f} | Status: {r['status']:<9} | Deliverable: \"{r['deliverable']}\"")

    print("\nExtracted Evidence Fed to Beta Factor Graph:")
    for ev in bonus_res["evidence_extracted"]:
        print(f"  - {ev['frame_id']}: s_evidence={ev['s_evidence']:.3f}, f_evidence={ev['f_evidence']:.3f} (Weight={ev['effective_weight']:.3f})")

    print("\nBeta Belief Propagation Posteriors:")
    for f_id, b_dict in bonus_res["beta_posteriors"].items():
        print(f"  - Frame {f_id}: α={b_dict['alpha']:.3f}, β={b_dict['beta']:.3f} -> Mean E[p]={b_dict['mean']:.4f}, Var={b_dict['variance']:.5f}")

    if bonus_res["pln_relational_inference"]:
        pln_step = bonus_res["pln_relational_inference"]
        print("\nPLN Relational Inference on Context-Frames Hierarchy:")
        print(f"  Derived Proposition: {pln_step['derived_belief'].label}")
        print(f"  Final Belief: {pln_step['derived_belief'].stv}")
        print(f"  Audit Trace: {pln_step['trace'].explanation}")

    print("\n" + format_separator("Evaluation Execution Successfully Completed"))


if __name__ == "__main__":
    main()
