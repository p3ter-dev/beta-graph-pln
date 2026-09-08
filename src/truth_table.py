"""
Truth Table and Candidate Program Definitions for Parity-3.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from src.boolean_evaluator import BooleanEvaluator

# The Parity-3 Truth Table: O = A XOR B XOR C
PARITY_3_DATA: List[Dict[str, int]] = [
    {"row_id": 0, "A": 0, "B": 0, "C": 0, "O": 0},
    {"row_id": 1, "A": 0, "B": 0, "C": 1, "O": 1},
    {"row_id": 2, "A": 0, "B": 1, "C": 0, "O": 1},
    {"row_id": 3, "A": 0, "B": 1, "C": 1, "O": 0},
    {"row_id": 4, "A": 1, "B": 0, "C": 0, "O": 1},
    {"row_id": 5, "A": 1, "B": 0, "C": 1, "O": 0},
    {"row_id": 6, "A": 1, "B": 1, "C": 0, "O": 0},
    {"row_id": 7, "A": 1, "B": 1, "C": 1, "O": 1},
]

# The 20 Candidate Boolean Programs
CANDIDATE_EXPRESSIONS: List[Tuple[int, str]] = [
    (1, "(AND A (OR B C))"),
    (2, "(OR (AND A B) C)"),
    (3, "(NOT (AND A (OR B C)))"),
    (4, "(OR (NOT A) (AND B C))"),
    (5, "(AND (OR A B) (NOT C))"),
    (6, "(OR (AND A (NOT B)) (AND B C))"),
    (7, "(AND (NOT A) (OR B (NOT C)))"),
    (8, "(OR (NOT (AND A B)) C)"),
    (9, "(AND (OR A (NOT B)) (OR B C))"),
    (10, "(NOT (OR (AND A C) B))"),
    (11, "(OR (AND A B) (AND (NOT A) C))"),
    (12, "(AND (OR A C) (OR (NOT B) C))"),
    (13, "(OR (AND (NOT A) B) (AND A C))"),
    (14, "(NOT (AND (OR A B) (NOT C)))"),
    (15, "(AND (NOT (OR A C)) (OR B C))"),
    (16, "(OR (AND A (NOT C)) (AND (NOT B) C))"),
    (17, "(AND (OR (NOT A) B) (NOT (AND B C)))"),
    (18, "(OR (NOT (OR A B)) (AND A C))"),
    (19, "(AND (OR A (AND B C)) (OR (NOT B) C))"),
    (20, "(OR (AND A (NOT B)) (AND (OR B C) (NOT A)))"),
]


@dataclass
class CandidateEvaluation:
    candidate_id: int
    expression: str
    predictions: List[int]
    row_matches: List[bool]
    matches: int
    mismatches: int
    match_rate: float

    def to_dict(self) -> Dict:
        return {
            "candidate_id": self.candidate_id,
            "candidate_expression": self.expression,
            "predictions": self.predictions,
            "row_matches": self.row_matches,
            "number_of_matches": self.matches,
            "number_of_mismatches": self.mismatches,
            "match_rate": round(self.match_rate, 4),
        }


def get_ground_truth_outputs() -> List[int]:
    """Return the observed output O vector of length 8."""
    return [row["O"] for row in PARITY_3_DATA]


def evaluate_candidate(candidate_id: int, expression: str) -> CandidateEvaluation:
    """Evaluate a single candidate against all 8 rows of Parity-3."""
    evaluator = BooleanEvaluator(expression)
    predictions = []
    row_matches = []
    matches = 0
    mismatches = 0

    for row in PARITY_3_DATA:
        pred = evaluator.evaluate(row["A"], row["B"], row["C"])
        predictions.append(pred)
        is_match = (pred == row["O"])
        row_matches.append(is_match)
        if is_match:
            matches += 1
        else:
            mismatches += 1

    total = len(PARITY_3_DATA)
    match_rate = matches / total

    return CandidateEvaluation(
        candidate_id=candidate_id,
        expression=expression,
        predictions=predictions,
        row_matches=row_matches,
        matches=matches,
        mismatches=mismatches,
        match_rate=match_rate,
    )


def evaluate_all_candidates() -> List[CandidateEvaluation]:
    """Evaluate all 20 candidate programs programmatically."""
    return [evaluate_candidate(cid, expr) for cid, expr in CANDIDATE_EXPRESSIONS]
