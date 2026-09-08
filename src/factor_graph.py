"""
Factor Graph Representation and Belief Propagation.

Supports:
1. Discrete Factor Graph with Sum-Product Message Passing:
   - Variable nodes (circles)
   - Factor nodes (squares)
   - Directed message passing between variables and factors
   - Marginalization and belief extraction
2. Beta-Bernoulli Parameter Factor Graph:
   - Conjugate exponential family message passing (natural parameters)
   - Local observation factors, prior factors, and correlation factors
3. Visualization:
   - Generates high-resolution PNG factor graph diagrams using networkx & matplotlib
   - Produces textual factorization tables
"""

from collections import defaultdict
from dataclasses import dataclass
import itertools
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from src.beta_model import BetaDistribution


@dataclass
class DiscreteVariable:
    """Discrete random variable in a factor graph."""
    name: str
    domain: List[Any]  # e.g. [0, 1] or ["F", "T"]

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, DiscreteVariable) and self.name == other.name


@dataclass
class DiscreteFactor:
    """
    Local factor f(x_1, ..., x_k) mapping variable assignments to non-negative potentials.
    """
    name: str
    variables: List[DiscreteVariable]
    potential_fn: Callable[..., float]
    meaning: str = ""

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, DiscreteFactor) and self.name == other.name

    def evaluate(self, assignment: Dict[str, Any]) -> float:
        args = [assignment[v.name] for v in self.variables]
        return float(self.potential_fn(*args))


class DiscreteFactorGraph:
    """
    Bipartite Factor Graph supporting classical Sum-Product Belief Propagation.
    """

    def __init__(self, name: str = "DiscreteFactorGraph"):
        self.name = name
        self.variables: Dict[str, DiscreteVariable] = {}
        self.factors: Dict[str, DiscreteFactor] = {}
        self.var_neighbors: Dict[str, Set[str]] = defaultdict(set)  # var -> set of factor names
        self.factor_neighbors: Dict[str, Set[str]] = defaultdict(set)  # factor -> set of var names

        # Messages: (sender, receiver) -> dict mapping state to float
        self.messages: Dict[Tuple[str, str], Dict[Any, float]] = {}

    def add_variable(self, name: str, domain: List[Any]) -> DiscreteVariable:
        if name in self.variables:
            return self.variables[name]
        var = DiscreteVariable(name, domain)
        self.variables[name] = var
        return var

    def add_factor(
        self,
        name: str,
        variables: List[DiscreteVariable],
        potential_fn: Callable[..., float],
        meaning: str = "",
    ) -> DiscreteFactor:
        factor = DiscreteFactor(name, variables, potential_fn, meaning)
        self.factors[name] = factor
        for var in variables:
            self.var_neighbors[var.name].add(name)
            self.factor_neighbors[name].add(var.name)
        return factor

    def initialize_messages(self):
        """Initialize all directed messages to uniform (1.0)."""
        self.messages.clear()
        for f_name, factor in self.factors.items():
            for var in factor.variables:
                v_name = var.name
                # m(v -> f)
                self.messages[(v_name, f_name)] = {val: 1.0 for val in var.domain}
                # m(f -> v)
                self.messages[(f_name, v_name)] = {val: 1.0 for val in var.domain}

    def run_sum_product_bp(self, max_iterations: int = 20, tol: float = 1e-6) -> int:
        """
        Execute parallel Sum-Product Belief Propagation.
        Equations implemented:
        Variable -> Factor:
            m_{x -> f}(x) = prod_{h in N(x) \\ {f}} m_{h -> x}(x)
        Factor -> Variable:
            m_{f -> x}(x) = sum_{N(f) \\ {x}} [ f(x, y) * prod_{y in N(f) \\ {x}} m_{y -> f}(y) ]
        """
        self.initialize_messages()

        for iteration in range(max_iterations):
            max_delta = 0.0
            new_messages: Dict[Tuple[str, str], Dict[Any, float]] = {}

            # 1. Update Variable -> Factor messages
            for v_name, var in self.variables.items():
                nbr_factors = list(self.var_neighbors[v_name])
                for target_f in nbr_factors:
                    msg = {}
                    other_factors = [f for f in nbr_factors if f != target_f]
                    for val in var.domain:
                        prod = 1.0
                        for other_f in other_factors:
                            prod *= self.messages[(other_f, v_name)][val]
                        msg[val] = prod
                    # Normalize
                    tot = sum(msg.values())
                    if tot > 0:
                        msg = {k: v / tot for k, v in msg.items()}
                    new_messages[(v_name, target_f)] = msg

            # 2. Update Factor -> Variable messages
            for f_name, factor in self.factors.items():
                nbr_vars = factor.variables
                for target_var in nbr_vars:
                    target_name = target_var.name
                    other_vars = [v for v in nbr_vars if v.name != target_name]
                    msg = {val: 0.0 for val in target_var.domain}

                    # Sum over all configurations of other variables
                    if not other_vars:
                        # Unary factor
                        for val in target_var.domain:
                            msg[val] = factor.evaluate({target_name: val})
                    else:
                        other_domains = [v.domain for v in other_vars]
                        for other_vals in itertools.product(*other_domains):
                            assign = dict(zip([v.name for v in other_vars], other_vals))
                            prod_incoming = 1.0
                            for o_var, o_val in zip(other_vars, other_vals):
                                prod_incoming *= self.messages[(o_var.name, f_name)][o_val]

                            for target_val in target_var.domain:
                                full_assign = dict(assign)
                                full_assign[target_name] = target_val
                                pot = factor.evaluate(full_assign)
                                msg[target_val] += pot * prod_incoming

                    # Normalize
                    tot = sum(msg.values())
                    if tot > 0:
                        msg = {k: v / tot for k, v in msg.items()}
                    new_messages[(f_name, target_name)] = msg

            # Check convergence
            for key, msg in new_messages.items():
                old_msg = self.messages.get(key, {})
                for val, p in msg.items():
                    delta = abs(p - old_msg.get(val, 0.0))
                    if delta > max_delta:
                        max_delta = delta

            self.messages.update(new_messages)
            if max_delta < tol:
                return iteration + 1

        return max_iterations

    def get_belief(self, var_name: str) -> Dict[Any, float]:
        """
        Compute marginal belief: b(x) = const * prod_{f in N(x)} m_{f -> x}(x)
        """
        var = self.variables[var_name]
        nbr_factors = self.var_neighbors[var_name]
        unnormalized = {}
        for val in var.domain:
            prod = 1.0
            for f_name in nbr_factors:
                prod *= self.messages.get((f_name, var_name), {}).get(val, 1.0)
            unnormalized[val] = prod

        tot = sum(unnormalized.values())
        if tot <= 0:
            return {val: 1.0 / len(var.domain) for val in var.domain}
        return {val: p / tot for val, p in unnormalized.items()}


@dataclass
class ContinuousFactor:
    """Factor in a Beta-Bernoulli parameter factor graph."""
    name: str
    target_var: str
    factor_type: str  # 'prior', 'observation', 'correlation', 'cross_candidate'
    alpha_contrib: float
    beta_contrib: float
    meaning: str
    weight: float = 1.0

    def message_to_variable(self) -> Tuple[float, float]:
        """Return (delta_alpha, delta_beta) passed to the variable."""
        return (self.alpha_contrib * self.weight, self.beta_contrib * self.weight)


class BetaFactorGraph:
    """
    Factor graph representing the uncertainty model over candidate hypotheses.
    Variables: Candidate hypothesis parameters theta_i (with prior Beta(alpha_0, beta_0)).
    Factors:
    - Prior factors: psi_prior(theta_i)
    - Observation compatibility factors: psi_obs_r(theta_i, y_{i,r}, o_r)
    - Correlation factors: psi_corr(theta_i)
    - Consistency factors: psi_cons(theta_i, theta_j)
    """

    def __init__(self, name: str = "BetaFactorGraph", alpha_0: float = 1.0, beta_0: float = 1.0):
        self.name = name
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.variable_priors: Dict[str, BetaDistribution] = {}
        self.factors: Dict[str, List[ContinuousFactor]] = defaultdict(list)  # var_name -> list of factors
        self.edges: List[Tuple[str, str, str]] = []  # (factor_name, var_name, meaning)

    def add_candidate_variable(self, var_name: str, prior: Optional[BetaDistribution] = None):
        dist = prior or BetaDistribution(self.alpha_0, self.beta_0)
        self.variable_priors[var_name] = dist

    def add_factor(self, factor: ContinuousFactor):
        self.factors[factor.target_var].append(factor)
        self.edges.append((factor.name, factor.target_var, factor.meaning))

    def compute_variable_posterior(self, var_name: str) -> BetaDistribution:
        """
        Belief propagation / conjugate update for variable:
        alpha_post = alpha_prior + sum_{f in N(var)} delta_alpha(f)
        beta_post  = beta_prior  + sum_{f in N(var)} delta_beta(f)
        """
        prior = self.variable_priors.get(var_name, BetaDistribution(self.alpha_0, self.beta_0))
        tot_alpha = prior.alpha
        tot_beta = prior.beta

        for factor in self.factors.get(var_name, []):
            d_alpha, d_beta = factor.message_to_variable()
            tot_alpha += d_alpha
            tot_beta += d_beta

        return BetaDistribution(tot_alpha, tot_beta)

    def get_factorization_text(self) -> str:
        """Produce a structured text audit of all variables, factors, domains, and edges."""
        lines = [
            "=" * 70,
            f"FACTOR GRAPH FACTORIZATION AUDIT: {self.name}",
            "=" * 70,
            f"Global Prior: Beta(alpha_0={self.alpha_0}, beta_0={self.beta_0})",
            f"Total Variables: {len(self.variable_priors)}",
            f"Total Factors: {sum(len(f_list) for f_list in self.factors.values())}",
            f"Total Edges: {len(self.edges)}",
            "-" * 70,
            "VARIABLES & DOMAINS:",
        ]
        for var_name, prior in sorted(self.variable_priors.items()):
            lines.append(f"  * Variable: {var_name:<16} Domain: theta in [0, 1]  Prior: {prior}")

        lines.append("-" * 70)
        lines.append("FACTORS, ARGUMENTS & LOCAL POTENTIALS:")
        for var_name, f_list in sorted(self.factors.items()):
            lines.append(f"  Target Variable: {var_name}")
            for f in f_list:
                d_a, d_b = f.message_to_variable()
                lines.append(
                    f"    - Factor: {f.name:<24} Type: {f.factor_type:<12} "
                    f"Messages: [Δα={d_a:+.3f}, Δβ={d_b:+.3f}]  Meaning: {f.meaning}"
                )

        lines.append("-" * 70)
        lines.append("EDGES (Factor -> Variable Dependency):")
        for f_name, v_name, meaning in self.edges[:25]:
            lines.append(f"  Edge: {f_name} ---> {v_name} ({meaning})")
        if len(self.edges) > 25:
            lines.append(f"  ... [{len(self.edges) - 25} additional observation edges]")

        lines.append("=" * 70)
        return "\n".join(lines)

    def save_visualization(self, filepath: str = "factor_graph.png"):
        """
        Render and save a high-resolution bipartite factor graph diagram.
        Variables: Light blue circles
        Factors: Coral/Orange squares
        """
        G = nx.Graph()

        # Add nodes
        for var_name in self.variable_priors.keys():
            G.add_node(var_name, bipartite=0, node_type="variable")

        # Select a clean representative subset of factors for clear visualization
        displayed_factors = []
        for var_name, f_list in self.factors.items():
            for f in f_list:
                # include prior, correlation, and representative observation factors
                if f.factor_type in ("prior", "correlation", "cross_candidate") or f.name.endswith("_r0") or f.name.endswith("_r7"):
                    displayed_factors.append(f)

        for f in displayed_factors:
            G.add_node(f.name, bipartite=1, node_type="factor")
            G.add_edge(f.name, f.target_var)

        plt.figure(figsize=(16, 10), dpi=200)

        # Bipartite layout
        var_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "variable"]
        factor_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "factor"]

        pos = {}
        # Place variables on top row
        v_step = 1.0 / (len(var_nodes) + 1)
        for i, node in enumerate(sorted(var_nodes)):
            pos[node] = ((i + 1) * v_step, 0.8)

        # Place factors on bottom row
        f_step = 1.0 / (len(factor_nodes) + 1)
        for j, node in enumerate(sorted(factor_nodes)):
            pos[node] = ((j + 1) * f_step, 0.2)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color="#7f8c8d", alpha=0.6, width=1.2)

        # Draw variable nodes (circles)
        nx.draw_networkx_nodes(
            G, pos, nodelist=var_nodes, node_shape="o", node_color="#3498db",
            node_size=800, edgecolors="#2980b9", linewidths=2.0, label="Variables (Candidates)"
        )

        # Draw factor nodes (squares)
        nx.draw_networkx_nodes(
            G, pos, nodelist=factor_nodes, node_shape="s", node_color="#e67e22",
            node_size=600, edgecolors="#d35400", linewidths=2.0, label="Factors (Evidence / Prior)"
        )

        # Node labels
        var_labels = {n: n.replace("Candidate_", "C") for n in var_nodes}
        nx.draw_networkx_labels(G, pos, labels=var_labels, font_size=8, font_family="sans-serif", font_weight="bold", font_color="white")

        factor_labels = {n: n.replace("Factor_", "f_").replace("Candidate_", "C") for n in factor_nodes}
        # Draw factor labels slightly below
        pos_f_labels = {k: (v[0], v[1] - 0.06) for k, v in pos.items() if k in factor_nodes}
        nx.draw_networkx_labels(G, pos_f_labels, labels=factor_labels, font_size=6, font_family="sans-serif")

        plt.title("Factor Graph: Candidate Boolean Hypotheses & Evidence Factors (Bipartite)", fontsize=14, fontweight="bold", pad=20)
        plt.axis("off")
        plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#bdc3c7")
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight", dpi=200)
        plt.close()
