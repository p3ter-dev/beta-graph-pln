# Factor Graph + Beta/PLN Evaluation

A reproducible Python project for evaluating Boolean programs against the 3-bit parity function. It combines:

- Boolean S-expression parsing and evaluation
- MCC correlation and Hamming-weighted evidence
- Beta-distribution belief updates
- Factor-graph belief propagation
- PLN deduction, induction, abduction, revision, and provenance tracking
- Optional Context-Frames vector search with TF-IDF and cosine similarity

## Requirements

Python 3.10 or newer is recommended.

Install the dependencies:

```bash
python -m pip install numpy scikit-learn matplotlib networkx pytest
```

## Run the Evaluation

From the repository root:

```bash
python run_evaluation.py
```

The command evaluates all 20 candidates, prints the rankings and PLN traces, runs the Context-Frames bonus pipeline, and generates `factor_graph.png`.

## Run Tests

```bash
python -m pytest
```

## Project Structure

- `src/truth_table.py` - Parity-3 data and candidate programs
- `src/boolean_evaluator.py` - Boolean expression parser and evaluator
- `src/correlation_weights.py` - MCC and weighted evidence calculations
- `src/beta_model.py` - Beta distributions and PLN truth values
- `src/factor_graph.py` - Factor graphs and belief propagation
- `src/pln_inference.py` - PLN inference rules and evidence provenance
- `src/context_frames.py` - Context-Frames parsing and hierarchy
- `src/vector_search.py` - TF-IDF frame retrieval
- `src/pipeline.py` - End-to-end orchestration
- `run_evaluation.py` - Main executable demonstration
- `tests/` - Automated tests
