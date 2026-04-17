"""Particle Swarm Optimization demo with visual and sensitivity analyses."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pso import ParticleSwarmOptimizer
from utils.benchmark_functions import ackley, rastrigin, rosenbrock
from utils.visualizer import plot_convergence

SEED: int = 42
DPI: int = 150
DIM_MAIN: int = 10
DIM_VIS: int = 2
N_PARTICLES: int = 50
MAX_ITER: int = 200
PSO_W_FIXED: float = 0.7
PSO_C1: float = 1.5
PSO_C2: float = 1.5
SNAPSHOT_ITERS: list[int] = [1, 25, 50, 100]
W_VALUES: list[float] = [0.4, 0.7, 0.9]
C_VALUES: list[float] = [1.0, 1.5, 2.0]
CONTOUR_POINTS: int = 200
CONTOUR_LEVELS: int = 40
SNAPSHOT_FIGSIZE: tuple[float, float] = (12.0, 10.0)
HEATMAP_FIGSIZE: tuple[float, float] = (7.0, 5.5)
PARTICLE_MARKER_SIZE: int = 28
BEST_MARKER_SIZE: int = 250
PARTICLE_EDGE_WIDTH: float = 0.5
BEST_EDGE_WIDTH: float = 0.9
VELOCITY_CLAMP_SCALE: float = 0.2
RANDOM_W_LOW: float = 0.5
RANDOM_W_HIGH: float = 1.0


def save_and_show(fig: plt.Figure, output_path: Path) -> None:
    """Save a figure as PNG and show interactively."""

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.show()


def run_high_dim_cases() -> None:
    """Run PSO on three 10D benchmark functions and print errors."""

    cases: list[tuple[str, object]] = [
        ("Rastrigin", rastrigin),
        ("Rosenbrock", rosenbrock),
        ("Ackley", ackley),
    ]

    print("=" * 72)
    print("PSO High-Dimensional Optimization")
    print("=" * 72)

    for idx, (name, func) in enumerate(cases):
        pso = ParticleSwarmOptimizer(
            func=func,
            bounds=func.bounds,
            n_dim=DIM_MAIN,
            n_particles=N_PARTICLES,
            max_iter=MAX_ITER,
            w=PSO_W_FIXED,
            c1=PSO_C1,
            c2=PSO_C2,
            w_decay=True,
            random_state=SEED + idx,
        )
        _, best_f, _ = pso.run()
        error: float = abs(best_f - 0.0)

        print(f"{name:10s} | best found: {best_f:.8f} | true optimum: 0.0 | error: {error:.8f}")


def contour_snapshots(output_dir: Path) -> None:
    """Create 2x2 contour snapshots of swarm positions over iterations."""

    pso = ParticleSwarmOptimizer(
        func=rastrigin,
        bounds=rastrigin.bounds,
        n_dim=DIM_VIS,
        n_particles=N_PARTICLES,
        max_iter=MAX_ITER,
        w=PSO_W_FIXED,
        c1=PSO_C1,
        c2=PSO_C2,
        w_decay=True,
        random_state=SEED,
    )
    _, _, history = pso.run()

    low: float = float(rastrigin.bounds["low"])
    high: float = float(rastrigin.bounds["high"])
    axis = np.linspace(low, high, CONTOUR_POINTS)
    xx, yy = np.meshgrid(axis, axis)
    zz = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = rastrigin(np.array([xx[i, j], yy[i, j]]))

    fig, axes = plt.subplots(2, 2, figsize=SNAPSHOT_FIGSIZE)
    axes_flat = axes.flatten()

    all_positions = history["all_positions"]
    gbest_positions = history["gbest_position"]

    for ax, iter_id in zip(axes_flat, SNAPSHOT_ITERS):
        idx: int = iter_id - 1
        contour = ax.contourf(xx, yy, zz, levels=CONTOUR_LEVELS, cmap="viridis")
        positions = all_positions[idx]
        best = gbest_positions[idx]
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c="white",
            s=PARTICLE_MARKER_SIZE,
            edgecolors="black",
            linewidths=PARTICLE_EDGE_WIDTH,
            label="Particles",
        )
        ax.scatter(
            best[0],
            best[1],
            c="yellow",
            marker="*",
            s=BEST_MARKER_SIZE,
            edgecolors="black",
            linewidths=BEST_EDGE_WIDTH,
            label="Global best",
        )
        ax.set_title(f"Iteration {iter_id}")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=0.8, label="f(x)")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("PSO Swarm Convergence on 2D Rastrigin", y=1.02)
    save_and_show(fig, output_dir / "pso_rastrigin_2d_snapshots.png")

    fig_conv = plot_convergence(
        {"PSO 2D Rastrigin": [float(v) for v in history["gbest_fitness"]]},
        "PSO Convergence on 2D Rastrigin",
    )
    save_and_show(fig_conv, output_dir / "pso_rastrigin_2d_convergence.png")


def sensitivity_heatmap(output_dir: Path) -> None:
    """Plot final fitness heatmap for inertia/social-cognitive settings."""

    heatmap = np.zeros((len(W_VALUES), len(C_VALUES)), dtype=float)

    for i, w_val in enumerate(W_VALUES):
        for j, c_val in enumerate(C_VALUES):
            pso = ParticleSwarmOptimizer(
                func=rastrigin,
                bounds=rastrigin.bounds,
                n_dim=DIM_MAIN,
                n_particles=N_PARTICLES,
                max_iter=MAX_ITER,
                w=w_val,
                c1=c_val,
                c2=c_val,
                w_decay=False,
                random_state=SEED + i * 10 + j,
            )
            _, best_f, _ = pso.run()
            heatmap[i, j] = best_f

    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    im = ax.imshow(heatmap, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, label="Final best fitness")
    ax.set_xticks(range(len(C_VALUES)))
    ax.set_xticklabels([str(v) for v in C_VALUES])
    ax.set_yticks(range(len(W_VALUES)))
    ax.set_yticklabels([str(v) for v in W_VALUES])
    ax.set_xlabel("c1 = c2")
    ax.set_ylabel("w")
    ax.set_title("PSO Sensitivity Heatmap on Rastrigin (10D)")

    for i in range(len(W_VALUES)):
        for j in range(len(C_VALUES)):
            ax.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center", color="white")

    save_and_show(fig, output_dir / "pso_sensitivity_heatmap.png")


def run_random_inertia_strategy(
    func,
    bounds: dict[str, float],
    n_dim: int,
    n_particles: int,
    max_iter: int,
    c1: float,
    c2: float,
    random_state: int,
) -> tuple[np.ndarray, float, list[float]]:
    """Run a PSO variant where inertia is sampled randomly each iteration."""

    rng = np.random.default_rng(random_state)
    low: float = float(bounds["low"])
    high: float = float(bounds["high"])
    velocity_scale: float = VELOCITY_CLAMP_SCALE * (high - low)

    positions = rng.uniform(low, high, size=(n_particles, n_dim))
    velocities = rng.uniform(-velocity_scale, velocity_scale, size=(n_particles, n_dim))

    pbest_positions = positions.copy()
    pbest_fitness = np.array([func(p) for p in positions], dtype=float)
    gbest_idx: int = int(np.argmin(pbest_fitness))
    gbest = pbest_positions[gbest_idx].copy()
    gbest_f = float(pbest_fitness[gbest_idx])

    history: list[float] = []

    for _ in range(max_iter):
        w_rand: float = float(rng.uniform(RANDOM_W_LOW, RANDOM_W_HIGH))

        for i in range(n_particles):
            r1 = rng.random(n_dim)
            r2 = rng.random(n_dim)
            velocities[i] = (
                w_rand * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest - positions[i])
            )
            velocities[i] = np.clip(velocities[i], -velocity_scale, velocity_scale)
            positions[i] = positions[i] + velocities[i]

            low_mask = positions[i] < low
            high_mask = positions[i] > high
            out_mask = low_mask | high_mask
            if np.any(out_mask):
                velocities[i, out_mask] *= -1.0
                positions[i] = np.clip(positions[i], low, high)

        current_f = np.array([func(p) for p in positions], dtype=float)
        improved = current_f < pbest_fitness
        pbest_positions[improved] = positions[improved]
        pbest_fitness[improved] = current_f[improved]

        idx_best = int(np.argmin(pbest_fitness))
        if pbest_fitness[idx_best] < gbest_f:
            gbest_f = float(pbest_fitness[idx_best])
            gbest = pbest_positions[idx_best].copy()

        history.append(gbest_f)

    return gbest, gbest_f, history


def compare_inertia_strategies(output_dir: Path) -> None:
    """Compare fixed, linear-decay, and random inertia weight schedules."""

    fixed = ParticleSwarmOptimizer(
        func=rastrigin,
        bounds=rastrigin.bounds,
        n_dim=DIM_MAIN,
        n_particles=N_PARTICLES,
        max_iter=MAX_ITER,
        w=PSO_W_FIXED,
        c1=PSO_C1,
        c2=PSO_C2,
        w_decay=False,
        random_state=SEED + 100,
    )
    _, _, hist_fixed = fixed.run()

    decayed = ParticleSwarmOptimizer(
        func=rastrigin,
        bounds=rastrigin.bounds,
        n_dim=DIM_MAIN,
        n_particles=N_PARTICLES,
        max_iter=MAX_ITER,
        w=PSO_W_FIXED,
        c1=PSO_C1,
        c2=PSO_C2,
        w_decay=True,
        random_state=SEED + 101,
    )
    _, _, hist_decay = decayed.run()

    _, _, hist_random = run_random_inertia_strategy(
        func=rastrigin,
        bounds=rastrigin.bounds,
        n_dim=DIM_MAIN,
        n_particles=N_PARTICLES,
        max_iter=MAX_ITER,
        c1=PSO_C1,
        c2=PSO_C2,
        random_state=SEED + 102,
    )

    fig = plot_convergence(
        {
            "Fixed w=0.7": [float(v) for v in hist_fixed["gbest_fitness"]],
            "Linear decay 0.9->0.4": [float(v) for v in hist_decay["gbest_fitness"]],
            "Random w~U(0.5,1.0)": hist_random,
        },
        "PSO Inertia Weight Strategy Comparison",
    )
    save_and_show(fig, output_dir / "pso_inertia_strategy_comparison.png")


def main() -> None:
    """Run PSO demo suite."""

    output_dir: Path = Path(__file__).resolve().parent

    run_high_dim_cases()
    contour_snapshots(output_dir)
    sensitivity_heatmap(output_dir)
    compare_inertia_strategies(output_dir)


if __name__ == "__main__":
    main()
