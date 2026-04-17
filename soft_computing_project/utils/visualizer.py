"""Reusable visualization utilities for soft computing experiments.

This module contains plotting helpers for convergence analysis, clustering
results, and population/swarm position overlays on objective contours.
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

CONVERGENCE_FIGSIZE: tuple[float, float] = (10.0, 5.0)
CLUSTER_FIGSIZE: tuple[float, float] = (7.0, 6.0)
POPULATION_FIGSIZE: tuple[float, float] = (7.0, 6.0)
CONTOUR_GRID_POINTS: int = 220
CONTOUR_LEVELS: int = 35
SCATTER_SIZE: int = 45
BEST_MARKER_SIZE: int = 250


def plot_convergence(histories: dict[str, list[float]], title: str) -> Figure:
    """Plot convergence curves for multiple algorithms on one chart."""

    fig, ax = plt.subplots(figsize=CONVERGENCE_FIGSIZE)
    for name, values in histories.items():
        if len(values) == 0:
            continue
        ax.plot(values, linewidth=2.0, label=name)
    ax.set_title(title)
    ax.set_xlabel("Iteration / Generation")
    ax.set_ylabel("Best Fitness")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    title: str,
) -> Figure:
    """Scatter plot of 2D clustered samples with highlighted centers."""

    fig, ax = plt.subplots(figsize=CLUSTER_FIGSIZE)
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.75,
        s=SCATTER_SIZE,
        edgecolors="k",
        linewidths=0.2,
    )
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="red",
        marker="X",
        s=BEST_MARKER_SIZE,
        edgecolors="white",
        linewidths=1.2,
        label="Centers",
    )
    fig.colorbar(scatter, ax=ax, label="Cluster label")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    return fig


def plot_population_2d(
    particles: np.ndarray,
    best: np.ndarray,
    iteration: int,
    func: Callable[[np.ndarray], float],
    bounds: dict[str, float],
) -> Figure:
    """Plot population/swarm positions on top of a 2D objective contour."""

    low: float = float(bounds["low"])
    high: float = float(bounds["high"])
    line_space: np.ndarray = np.linspace(low, high, CONTOUR_GRID_POINTS)
    xx, yy = np.meshgrid(line_space, line_space)
    zz = np.zeros_like(xx)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = func(np.array([xx[i, j], yy[i, j]], dtype=float))

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=POPULATION_FIGSIZE)
    contour = ax.contourf(xx, yy, zz, levels=CONTOUR_LEVELS, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="f(x)")

    ax.scatter(
        particles[:, 0],
        particles[:, 1],
        c="white",
        s=SCATTER_SIZE,
        edgecolors="black",
        linewidths=0.6,
        label="Population",
    )
    ax.scatter(
        best[0],
        best[1],
        c="yellow",
        marker="*",
        s=BEST_MARKER_SIZE,
        edgecolors="black",
        linewidths=1.0,
        label="Best",
    )
    ax.set_title(f"Population Overlay - Iteration {iteration}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
