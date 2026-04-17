"""Genetic Algorithm demo on Sphere and Rastrigin benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ga import GeneticAlgorithm
from utils.benchmark_functions import rastrigin, sphere
from utils.visualizer import plot_convergence

SEED: int = 42
DPI: int = 150
RASTRIGIN_DIM: int = 10
SPHERE_DIM: int = 20
POP_SIZE: int = 50
MAX_GEN: int = 200
GA_CROSSOVER_RATE: float = 0.9
GA_MUTATION_RATE: float = 0.03
GA_ELITISM: int = 2
GA_SELECTION: str = "tournament"
GA_CROSSOVER_METHOD: str = "sbx"
SENSITIVITY_POP_SIZES: list[int] = [20, 50, 100, 200]
TRUE_OPTIMUM: float = 0.0
BAR_FIGSIZE: tuple[float, float] = (8.0, 5.0)
HEADING_WIDTH: int = 70


def save_and_show(fig: plt.Figure, output_path: Path) -> None:
    """Save a matplotlib figure and display it interactively."""

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.show()


def run_single_case(
    name: str,
    func,
    n_dim: int,
    output_dir: Path,
    random_state: int,
) -> None:
    """Run one GA optimization case and print summary metrics."""

    ga = GeneticAlgorithm(
        func=func,
        bounds=func.bounds,
        n_dim=n_dim,
        pop_size=POP_SIZE,
        max_gen=MAX_GEN,
        crossover_rate=GA_CROSSOVER_RATE,
        mutation_rate=GA_MUTATION_RATE,
        elitism=GA_ELITISM,
        selection=GA_SELECTION,
        random_state=random_state,
        crossover_method=GA_CROSSOVER_METHOD,
    )

    best_x, best_val, history = ga.run()
    error: float = abs(best_val - TRUE_OPTIMUM)

    print("\n" + "=" * HEADING_WIDTH)
    print(f"{name} optimization ({n_dim}D)")
    print("=" * HEADING_WIDTH)
    print(f"Best solution found (first 5 genes): {np.round(best_x[:5], 6)}")
    print(f"Best objective value: {best_val:.8f}")
    print(f"True optimum: {TRUE_OPTIMUM:.8f}")
    print(f"Absolute error: {error:.8f}")

    fig_conv = plot_convergence({f"GA-{name}": history["best"]}, f"GA on {name}")
    save_and_show(fig_conv, output_dir / f"ga_{name.lower()}_convergence.png")


def sensitivity_study(output_dir: Path) -> None:
    """Analyze final fitness as a function of population size."""

    final_values: list[float] = []

    print("\n" + "=" * HEADING_WIDTH)
    print("GA Parameter Sensitivity: population size")
    print("=" * HEADING_WIDTH)

    for pop_size in SENSITIVITY_POP_SIZES:
        ga = GeneticAlgorithm(
            func=rastrigin,
            bounds=rastrigin.bounds,
            n_dim=RASTRIGIN_DIM,
            pop_size=pop_size,
            max_gen=MAX_GEN,
            crossover_rate=GA_CROSSOVER_RATE,
            mutation_rate=GA_MUTATION_RATE,
            elitism=GA_ELITISM,
            selection=GA_SELECTION,
            random_state=SEED + pop_size,
            crossover_method=GA_CROSSOVER_METHOD,
        )
        _, best_val, _ = ga.run()
        final_values.append(best_val)
        print(f"pop_size={pop_size:3d} -> final best fitness={best_val:.6f}")

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    bars = ax.bar(
        [str(p) for p in SENSITIVITY_POP_SIZES],
        final_values,
        color=["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16"],
    )
    for bar, val in zip(bars, final_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Population Size Sensitivity on Rastrigin (10D)")
    ax.set_xlabel("Population Size")
    ax.set_ylabel("Final Best Fitness (lower is better)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    print("\nComment on exploration-exploitation tradeoff:")
    print(
        "Smaller populations converge faster but can lose diversity and get "
        "trapped in local minima. Larger populations improve exploration and "
        "robustness on multimodal landscapes, but increase computational cost "
        "per generation."
    )

    save_and_show(fig, output_dir / "ga_population_sensitivity.png")


def main() -> None:
    """Execute all GA demo experiments."""

    output_dir: Path = Path(__file__).resolve().parent

    run_single_case("Rastrigin", rastrigin, RASTRIGIN_DIM, output_dir, SEED)
    run_single_case("Sphere", sphere, SPHERE_DIM, output_dir, SEED + 1)
    sensitivity_study(output_dir)


if __name__ == "__main__":
    main()
