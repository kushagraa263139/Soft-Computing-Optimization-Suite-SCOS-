"""Capstone benchmark comparison across FCM-inspired search, GA, and PSO.

This script runs a fair evaluation budget across benchmark functions and
summarizes quality, consistency, and convergence behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.benchmark_functions import ackley, rastrigin, rosenbrock, sphere

GA_DIR: Path = PROJECT_ROOT / "03_genetic_algorithm"
PSO_DIR: Path = PROJECT_ROOT / "04_pso"
FCM_DIR: Path = PROJECT_ROOT / "02_fcm"
for module_dir in [GA_DIR, PSO_DIR, FCM_DIR]:
    if str(module_dir) not in sys.path:
        sys.path.append(str(module_dir))

from ga import GeneticAlgorithm
from pso import ParticleSwarmOptimizer
from fcm import FuzzyCMeans

SEEDS: list[int] = list(range(10))
N_RUNS: int = 10
N_DIM: int = 10
POP_SIZE: int = 50
MAX_ITER: int = 200
TOTAL_EVALS: int = POP_SIZE * MAX_ITER
N_CLUSTERS_FCM_OPT: int = 3
FCM_M: float = 2.0
NOISE_SCALE: float = 0.08
DPI: int = 150
EPSILON: float = 1e-12
ALPHA_LOW: float = 0.3
ALPHA_HIGH: float = 0.9
SUMMARY_LINE_WIDTH: int = 90
SUMMARY_TABLEFMT: str = "github"
FCM_INNER_MAX_ITER: int = 50
FCM_INNER_TOL: float = 1e-5
GA_CROSSOVER_RATE: float = 0.9
GA_MUTATION_RATE: float = 0.03
GA_ELITISM: int = 2
GA_SELECTION: str = "tournament"
GA_CROSSOVER_METHOD: str = "sbx"
PSO_W: float = 0.7
PSO_C1: float = 1.5
PSO_C2: float = 1.5
THRESHOLD_MULTIPLIER: float = 1.1


def save_and_show(fig: plt.Figure, output_path: Path) -> None:
    """Save figure to PNG and show interactively."""

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.show()


def run_fcm_optimizer(
    func: Callable[[np.ndarray], float],
    bounds: dict[str, float],
    n_dim: int,
    pop_size: int,
    max_iter: int,
    seed: int,
) -> tuple[float, list[float]]:
    """FCM-inspired optimizer with equal function evaluation budget.

    Procedure:
        1. Maintain a population of candidate solutions.
        2. Cluster candidates using FCM to identify soft structure.
        3. Select strongest cluster center by objective value.
        4. Resample candidates around fuzzy-weighted center mixture.

    Returns:
        best fitness value, convergence history of global best values.
    """

    rng = np.random.default_rng(seed)
    low: float = float(bounds["low"])
    high: float = float(bounds["high"])
    span: float = high - low

    population: np.ndarray = rng.uniform(low, high, size=(pop_size, n_dim))
    best_global: float = np.inf
    history: list[float] = []

    for _ in range(max_iter):
        fitness: np.ndarray = np.array([func(ind) for ind in population], dtype=float)
        current_best: float = float(np.min(fitness))
        if current_best < best_global:
            best_global = current_best

        history.append(best_global)

        fcm_model = FuzzyCMeans(
            n_clusters=N_CLUSTERS_FCM_OPT,
            m=FCM_M,
            max_iter=FCM_INNER_MAX_ITER,
            tol=FCM_INNER_TOL,
            random_state=seed,
        )
        fcm_model.fit(population)

        assert fcm_model.centers_ is not None
        assert fcm_model.U_ is not None

        centers: np.ndarray = fcm_model.centers_
        center_scores: np.ndarray = np.array([func(center) for center in centers])
        best_center_idx: int = int(np.argmin(center_scores))

        # Convert memberships toward best center into sampling probabilities.
        weights: np.ndarray = fcm_model.U_[:, best_center_idx]
        probs: np.ndarray = weights / np.maximum(np.sum(weights), EPSILON)

        selected_idx: np.ndarray = rng.choice(pop_size, size=pop_size, p=probs)
        anchors: np.ndarray = population[selected_idx]

        # Mix around anchors and best fuzzy center to balance exploration and
        # exploitation while keeping the same evaluation budget as GA/PSO.
        alpha: np.ndarray = rng.uniform(ALPHA_LOW, ALPHA_HIGH, size=(pop_size, 1))
        gaussian_noise: np.ndarray = rng.normal(
            0.0,
            NOISE_SCALE * span,
            size=(pop_size, n_dim),
        )
        population = (
            alpha * anchors
            + (1.0 - alpha) * centers[best_center_idx][np.newaxis, :]
            + gaussian_noise
        )
        population = np.clip(population, low, high)

    return best_global, history


def run_ga(
    func: Callable[[np.ndarray], float],
    bounds: dict[str, float],
    seed: int,
) -> tuple[float, list[float]]:
    """Run GA under fair evaluation settings."""

    ga = GeneticAlgorithm(
        func=func,
        bounds=bounds,
        n_dim=N_DIM,
        pop_size=POP_SIZE,
        max_gen=MAX_ITER,
        crossover_rate=GA_CROSSOVER_RATE,
        mutation_rate=GA_MUTATION_RATE,
        elitism=GA_ELITISM,
        selection=GA_SELECTION,
        random_state=seed,
        crossover_method=GA_CROSSOVER_METHOD,
    )
    _, best_f, history = ga.run()
    return best_f, history["best"]


def run_pso(
    func: Callable[[np.ndarray], float],
    bounds: dict[str, float],
    seed: int,
) -> tuple[float, list[float]]:
    """Run PSO under fair evaluation settings."""

    pso = ParticleSwarmOptimizer(
        func=func,
        bounds=bounds,
        n_dim=N_DIM,
        n_particles=POP_SIZE,
        max_iter=MAX_ITER,
        w=PSO_W,
        c1=PSO_C1,
        c2=PSO_C2,
        w_decay=True,
        random_state=seed,
    )
    _, best_f, history = pso.run()
    return best_f, [float(v) for v in history["gbest_fitness"]]


def summarize_results(
    all_results: dict[str, dict[str, dict[str, list[float] | list[list[float]]]]],
) -> None:
    """Print tabulated summary statistics across functions and algorithms."""

    rows: list[list[str]] = []
    algorithms: list[str] = ["FCM", "GA", "PSO"]

    for func_name, by_algo in all_results.items():
        for algo in algorithms:
            values: np.ndarray = np.array(by_algo[algo]["best_fitness"], dtype=float)
            rows.append(
                [
                    func_name,
                    algo,
                    f"{np.min(values):.6e}",
                    f"{np.mean(values):.6e} +- {np.std(values):.6e}",
                    f"{np.median(values):.6e}",
                    str(TOTAL_EVALS),
                ]
            )

    headers: list[str] = ["Function", "Algorithm", "Best", "Mean +- Std", "Median", "Evals"]

    print("\n" + "=" * SUMMARY_LINE_WIDTH)
    print("SUMMARY TABLE")
    print("=" * SUMMARY_LINE_WIDTH)
    print(tabulate(rows, headers=headers, tablefmt=SUMMARY_TABLEFMT))


def boxplot_results(
    all_results: dict[str, dict[str, dict[str, list[float] | list[list[float]]]]],
    output_dir: Path,
) -> None:
    """Create 2x2 box plots of final fitness distributions."""

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))
    algorithms: list[str] = ["FCM", "GA", "PSO"]

    for ax, (func_name, by_algo) in zip(axes.flatten(), all_results.items()):
        data: list[np.ndarray] = [
            np.array(by_algo[algo]["best_fitness"], dtype=float) for algo in algorithms
        ]
        ax.boxplot(data, labels=algorithms)
        ax.set_title(func_name)
        ax.set_ylabel("Final best fitness")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Fitness Distributions Across 10 Independent Runs", y=1.02)
    save_and_show(fig, output_dir / "benchmark_boxplots.png")


def median_convergence_plot(
    all_results: dict[str, dict[str, dict[str, list[float] | list[list[float]]]]],
    output_dir: Path,
) -> None:
    """Plot median convergence curves for each function and algorithm."""

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
    algorithms: list[str] = ["FCM", "GA", "PSO"]

    for ax, (func_name, by_algo) in zip(axes.flatten(), all_results.items()):
        for algo in algorithms:
            curves = np.array(by_algo[algo]["convergence"], dtype=float)
            median_curve = np.median(curves, axis=0)
            ax.plot(median_curve, linewidth=2.0, label=algo)
        ax.set_title(func_name)
        ax.set_xlabel("Iteration / Generation")
        ax.set_ylabel("Median best fitness")
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend()

    fig.suptitle("Median Convergence Curves (log-scale y)", y=1.02)
    save_and_show(fig, output_dir / "benchmark_median_convergence.png")


def radar_chart(
    all_results: dict[str, dict[str, dict[str, list[float] | list[list[float]]]]],
    output_dir: Path,
) -> None:
    """Create radar chart over normalized aggregate performance criteria."""

    algorithms: list[str] = ["FCM", "GA", "PSO"]
    criteria: list[str] = [
        "Speed",
        "Quality",
        "Consistency",
        "Multimodal Robustness",
        "Computational Cost",
    ]

    quality: dict[str, float] = {}
    consistency: dict[str, float] = {}
    speed: dict[str, float] = {}
    robustness: dict[str, float] = {}
    cost: dict[str, float] = {}

    multimodal_funcs: list[str] = ["Rastrigin 10D", "Ackley 10D"]

    for algo in algorithms:
        all_final: list[float] = []
        median_iters_to_threshold: list[float] = []
        multi_scores: list[float] = []

        for func_name, by_algo in all_results.items():
            finals = np.array(by_algo[algo]["best_fitness"], dtype=float)
            all_final.extend(finals.tolist())

            curves = np.array(by_algo[algo]["convergence"], dtype=float)
            threshold: float = float(np.median(finals) * THRESHOLD_MULTIPLIER + EPSILON)

            per_run_iters: list[int] = []
            for curve in curves:
                reached = np.where(curve <= threshold)[0]
                if reached.size == 0:
                    per_run_iters.append(MAX_ITER)
                else:
                    per_run_iters.append(int(reached[0] + 1))
            median_iters_to_threshold.append(float(np.median(per_run_iters)))

            if func_name in multimodal_funcs:
                multi_scores.append(float(np.median(finals)))

        quality[algo] = -float(np.log10(np.median(all_final) + EPSILON))
        consistency[algo] = 1.0 / (float(np.std(all_final)) + EPSILON)
        speed[algo] = 1.0 / (float(np.mean(median_iters_to_threshold)) + EPSILON)
        robustness[algo] = -float(np.log10(np.mean(multi_scores) + EPSILON))
        cost[algo] = 1.0 / float(TOTAL_EVALS)

    raw_metrics: dict[str, dict[str, float]] = {
        "Speed": speed,
        "Quality": quality,
        "Consistency": consistency,
        "Multimodal Robustness": robustness,
        "Computational Cost": cost,
    }

    # Normalize each criterion to [0, 1] so polygons are comparable.
    normalized: dict[str, list[float]] = {algo: [] for algo in algorithms}
    for criterion in criteria:
        vals = np.array([raw_metrics[criterion][algo] for algo in algorithms], dtype=float)
        min_val: float = float(np.min(vals))
        max_val: float = float(np.max(vals))
        if abs(max_val - min_val) < EPSILON:
            scaled = np.ones_like(vals)
        else:
            scaled = (vals - min_val) / (max_val - min_val)
        for idx, algo in enumerate(algorithms):
            normalized[algo].append(float(scaled[idx]))

    angles = np.linspace(0.0, 2.0 * np.pi, len(criteria), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(8.0, 8.0))
    ax = fig.add_subplot(111, polar=True)

    for algo in algorithms:
        values = normalized[algo] + [normalized[algo][0]]
        ax.plot(angles, values, linewidth=2.0, label=algo)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria)
    ax.set_yticklabels([])
    ax.set_title("Algorithm Profile Radar Chart")
    ax.legend(loc="upper right")

    save_and_show(fig, output_dir / "benchmark_radar_chart.png")


def print_analysis() -> None:
    """Print a written analytical discussion (>= 200 words)."""

    analysis_text: str = (
        "Across continuous optimization benchmarks, PSO often outperforms GA "
        "when the search space is smooth enough that directional information can "
        "be exploited through velocity memory and social learning. The inertia, "
        "cognitive, and social terms let particles keep useful momentum while "
        "still reacting to local and global discoveries, which can accelerate "
        "convergence on Sphere and often Ackley. GA can be stronger when the "
        "landscape is highly rugged or when maintaining long-term diversity is "
        "critical, because recombination plus mutation can jump between distant "
        "basins in ways a tightly clustered swarm might not. In multimodal "
        "problems such as Rastrigin, both methods must prevent premature "
        "convergence: PSO does so with inertia scheduling and stochastic pull, "
        "while GA relies on mutation pressure, parent selection strategy, and "
        "population size.\n\n"
        "FCM is not a direct optimizer in the classical sense, but its soft "
        "assignment principle is very useful when boundaries are ambiguous. In "
        "clustering, soft membership can outperform hard assignment because real "
        "data often includes overlap, noise, and transitional samples. Instead of "
        "forcing every point into one class, FCM quantifies uncertainty via "
        "membership values, which is especially useful for decision support, "
        "anomaly screening, medical stratification, and customer segmentation with "
        "hybrid behavior patterns.\n\n"
        "Exploration-exploitation balance differs by method. GA explores via "
        "recombination and mutation, then exploits through selection and elitism. "
        "PSO explores early with larger inertia and exploits later as particles "
        "cohere around elite regions. FCM-based search, when adapted for "
        "optimization, explores by resampling around fuzzy structures and exploits "
        "by concentrating around high-quality centers. In practice, PSO is popular "
        "for continuous control and tuning, GA for combinatorial design and "
        "mixed-variable optimization, and FCM for interpretable soft partitioning "
        "in uncertain data regimes."
    )

    print("\n" + "=" * SUMMARY_LINE_WIDTH)
    print("WRITTEN ANALYSIS")
    print("=" * SUMMARY_LINE_WIDTH)
    print(analysis_text)


def main() -> None:
    """Run full benchmark comparison pipeline."""

    output_dir: Path = Path(__file__).resolve().parent

    function_cases: list[tuple[str, Callable[[np.ndarray], float]]] = [
        ("Sphere 10D", sphere),
        ("Rastrigin 10D", rastrigin),
        ("Rosenbrock 10D", rosenbrock),
        ("Ackley 10D", ackley),
    ]

    all_results: dict[str, dict[str, dict[str, list[float] | list[list[float]]]]] = {}

    for func_name, func in function_cases:
        print("\n" + "#" * SUMMARY_LINE_WIDTH)
        print(f"Processing {func_name}")
        print("#" * SUMMARY_LINE_WIDTH)

        all_results[func_name] = {
            "FCM": {"best_fitness": [], "convergence": []},
            "GA": {"best_fitness": [], "convergence": []},
            "PSO": {"best_fitness": [], "convergence": []},
        }

        for seed in tqdm(SEEDS, desc=f"{func_name} runs", total=N_RUNS):
            fcm_best, fcm_curve = run_fcm_optimizer(
                func=func,
                bounds=func.bounds,
                n_dim=N_DIM,
                pop_size=POP_SIZE,
                max_iter=MAX_ITER,
                seed=seed,
            )
            ga_best, ga_curve = run_ga(func=func, bounds=func.bounds, seed=seed)
            pso_best, pso_curve = run_pso(func=func, bounds=func.bounds, seed=seed)

            all_results[func_name]["FCM"]["best_fitness"].append(fcm_best)
            all_results[func_name]["FCM"]["convergence"].append(fcm_curve)

            all_results[func_name]["GA"]["best_fitness"].append(ga_best)
            all_results[func_name]["GA"]["convergence"].append(ga_curve)

            all_results[func_name]["PSO"]["best_fitness"].append(pso_best)
            all_results[func_name]["PSO"]["convergence"].append(pso_curve)

    summarize_results(all_results)
    boxplot_results(all_results, output_dir)
    median_convergence_plot(all_results, output_dir)
    radar_chart(all_results, output_dir)
    print_analysis()


if __name__ == "__main__":
    main()
