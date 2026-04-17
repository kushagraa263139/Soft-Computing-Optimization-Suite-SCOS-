"""Structured tutorial script for foundational soft computing concepts.

Run this module to print a compact, formatted tutorial that introduces soft
computing, its taxonomy, and key optimization concepts.
"""

from __future__ import annotations

from tabulate import tabulate

SECTION_LINE: str = "=" * 78
SUBSECTION_LINE: str = "-" * 78


def print_soft_computing_overview() -> None:
    """Print the conceptual overview of soft computing."""

    print(SECTION_LINE)
    print("1) WHAT IS SOFT COMPUTING?")
    print(SECTION_LINE)
    print(
        "Soft computing is a collection of methods that tolerate uncertainty, "
        "imprecision, and partial truth to produce robust approximate solutions."
    )
    print()
    print("Hard Computing vs Soft Computing")
    print(SUBSECTION_LINE)
    print(
        "Hard computing relies on exact models and deterministic logic, while "
        "soft computing accepts noisy data and nonlinearity to find practical "
        "near-optimal answers."
    )
    print()
    print("Key Paradigms")
    print(SUBSECTION_LINE)
    print("- Fuzzy Logic: reasoning with degrees of truth.")
    print("- Neural Networks: learning nonlinear mappings from data.")
    print("- Evolutionary Computing: optimization through selection and variation.")
    print("- Swarm Intelligence: collective adaptation via simple agents.")
    print()
    print("Why We Need It")
    print(SUBSECTION_LINE)
    print("- Many real problems are NP-hard or non-convex.")
    print("- Real systems contain uncertainty, noise, and incomplete knowledge.")
    print("- Approximate but robust solutions are often more useful than exact ones.")


def print_taxonomy_table() -> None:
    """Print a taxonomy table for FCM, GA, and PSO."""

    print()
    print(SECTION_LINE)
    print("2) TAXONOMY")
    print(SECTION_LINE)

    rows: list[list[str]] = [
        ["FCM", "Fuzzy sets", "Clustering", "Overlapping clusters"],
        ["GA", "Evolution", "Optimization", "Discrete/combinatorial"],
        ["PSO", "Bird flocking", "Optimization", "Continuous spaces"],
    ]

    headers: list[str] = ["Technique", "Inspired By", "Type", "Best For"]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def print_key_concepts() -> None:
    """Print core concepts used across population-based algorithms."""

    print()
    print(SECTION_LINE)
    print("3) KEY CONCEPTS")
    print(SECTION_LINE)
    print("Fitness Landscape")
    print(SUBSECTION_LINE)
    print(
        "An objective surface where each point is a candidate solution and its "
        "height is fitness/cost. Rugged landscapes contain many local optima."
    )
    print()
    print("Exploration vs Exploitation")
    print(SUBSECTION_LINE)
    print(
        "Exploration searches new regions to avoid local traps. Exploitation "
        "refines promising regions for higher precision. Good algorithms balance "
        "both across iterations."
    )
    print()
    print("Population-Based Search")
    print(SUBSECTION_LINE)
    print(
        "Instead of one candidate, a population of solutions evolves together, "
        "which improves robustness and diversity in complex spaces."
    )
    print()
    print("Convergence")
    print(SUBSECTION_LINE)
    print(
        "Convergence describes stabilization toward high-quality solutions. "
        "Fast convergence is useful, but premature convergence can reduce quality."
    )


def main() -> None:
    """Entry point for printing the tutorial."""

    print_soft_computing_overview()
    print_taxonomy_table()
    print_key_concepts()


if __name__ == "__main__":
    main()
