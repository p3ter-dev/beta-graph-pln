"""
Boolean S-Expression Evaluator.

Parses and evaluates candidate Boolean expressions containing:
- Logical operators: AND, OR, NOT
- Variables: A, B, C (0 or 1)
- Nested compound expressions
"""

from typing import Any, Dict, List, Union

ASTNode = Union[str, int, List[Any]]


def tokenize(expr_str: str) -> List[str]:
    """Tokenize an S-expression string into parentheses, operators, and symbols."""
    # Replace parentheses with spaced equivalents
    spaced = expr_str.replace("(", " ( ").replace(")", " ) ")
    tokens = [tok.strip() for tok in spaced.split() if tok.strip()]
    return tokens


def parse_sexpr(tokens: List[str]) -> ASTNode:
    """Parse tokens into a nested list AST."""
    if not tokens:
        raise ValueError("Unexpected end of expression while parsing.")

    token = tokens.pop(0)

    if token == "(":
        sub_list = []
        while tokens and tokens[0] != ")":
            sub_list.append(parse_sexpr(tokens))
        if not tokens:
            raise ValueError("Mismatched opening parenthesis '('")
        tokens.pop(0)  # Remove ')'
        return sub_list
    elif token == ")":
        raise ValueError("Unexpected closing parenthesis ')'")
    else:
        # Symbol, number, or variable
        upper_tok = token.upper()
        if upper_tok in ("0", "1"):
            return int(upper_tok)
        if upper_tok in ("TRUE", "FALSE"):
            return 1 if upper_tok == "TRUE" else 0
        return upper_tok


def parse_expression(expr_str: str) -> ASTNode:
    """Parse a Boolean S-expression string into an AST."""
    tokens = tokenize(expr_str)
    ast = parse_sexpr(tokens)
    if tokens:
        raise ValueError(f"Extraneous tokens after expression: {tokens}")
    return ast


def evaluate_ast(ast: ASTNode, env: Dict[str, int]) -> int:
    """
    Recursively evaluate an AST given an environment mapping variable names to 0 or 1.
    """
    if isinstance(ast, int):
        return 1 if ast != 0 else 0

    if isinstance(ast, str):
        var = ast.upper()
        if var in env:
            return 1 if env[var] != 0 else 0
        raise ValueError(f"Undefined variable in environment: {var}")

    if isinstance(ast, list):
        if not ast:
            raise ValueError("Empty sub-expression encountered.")

        op = str(ast[0]).upper()
        args = ast[1:]

        if op == "NOT":
            if len(args) != 1:
                raise ValueError(f"NOT operator expects exactly 1 argument, got {len(args)} in {ast}")
            val = evaluate_ast(args[0], env)
            return 0 if val else 1

        elif op == "AND":
            if len(args) < 2:
                raise ValueError(f"AND operator expects at least 2 arguments, got {len(args)} in {ast}")
            for arg in args:
                if evaluate_ast(arg, env) == 0:
                    return 0
            return 1

        elif op == "OR":
            if len(args) < 2:
                raise ValueError(f"OR operator expects at least 2 arguments, got {len(args)} in {ast}")
            for arg in args:
                if evaluate_ast(arg, env) == 1:
                    return 1
            return 0

        elif op == "XOR":
            if len(args) != 2:
                raise ValueError(f"XOR operator expects 2 arguments, got {len(args)} in {ast}")
            v1 = evaluate_ast(args[0], env)
            v2 = evaluate_ast(args[1], env)
            return v1 ^ v2

        else:
            raise ValueError(f"Unknown operator: {op}")

    raise TypeError(f"Invalid AST node type: {type(ast)}")


class BooleanEvaluator:
    """High-level evaluator for candidate Boolean programs."""

    def __init__(self, expr_str: str):
        self.raw_expr = expr_str.strip()
        self.ast = parse_expression(self.raw_expr)

    def evaluate(self, a: int, b: int, c: int) -> int:
        env = {"A": int(a), "B": int(b), "C": int(c)}
        return evaluate_ast(self.ast, env)

    def evaluate_truth_table(self, rows: List[Dict[str, int]]) -> List[int]:
        """Evaluate expression across a list of row dicts containing {'A', 'B', 'C'}."""
        return [self.evaluate(r["A"], r["B"], r["C"]) for r in rows]

    def __repr__(self) -> str:
        return f"BooleanEvaluator({self.raw_expr})"
