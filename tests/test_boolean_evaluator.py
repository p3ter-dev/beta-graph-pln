"""
Unit tests for the Boolean S-expression parser and evaluator.
"""

import pytest
from src.boolean_evaluator import BooleanEvaluator, evaluate_ast, parse_expression


def test_boolean_primitives():
    # Test NOT
    ev_not = BooleanEvaluator("(NOT A)")
    assert ev_not.evaluate(1, 0, 0) == 0
    assert ev_not.evaluate(0, 0, 0) == 1

    # Test AND
    ev_and = BooleanEvaluator("(AND A B)")
    assert ev_and.evaluate(1, 1, 0) == 1
    assert ev_and.evaluate(1, 0, 0) == 0
    assert ev_and.evaluate(0, 1, 0) == 0

    # Test OR
    ev_or = BooleanEvaluator("(OR A B)")
    assert ev_or.evaluate(0, 0, 0) == 0
    assert ev_or.evaluate(1, 0, 0) == 1
    assert ev_or.evaluate(0, 1, 0) == 1
    assert ev_or.evaluate(1, 1, 0) == 1


def test_compound_expressions():
    # Compound expression from candidate 1: (AND A (OR B C))
    ev = BooleanEvaluator("(AND A (OR B C))")
    assert ev.evaluate(0, 0, 0) == 0
    assert ev.evaluate(0, 1, 1) == 0
    assert ev.evaluate(1, 0, 0) == 0
    assert ev.evaluate(1, 1, 0) == 1
    assert ev.evaluate(1, 0, 1) == 1
    assert ev.evaluate(1, 1, 1) == 1


def test_invalid_syntax():
    with pytest.raises(ValueError):
        BooleanEvaluator("(AND A")

    with pytest.raises(ValueError):
        ev = BooleanEvaluator("(UNKNOWN A B)")
        ev.evaluate(0, 0, 0)
