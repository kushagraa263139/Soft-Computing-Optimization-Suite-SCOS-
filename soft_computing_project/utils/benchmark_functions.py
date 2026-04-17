"""Benchmark objective functions and 2D visualization helpers.

This module provides classic optimization benchmark functions used to test
metaheuristics such as Genetic Algorithms and Particle Swarm Optimization.
It also includes a reusable 2D surface + contour plotting utility.
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

PI2: float = 2.0 * np.pi
ACKLEY_A: float = 20.0
ACKLEY_B: float = 0.2
ACKLEY_C: float = PI2
SURFACE_GRID_POINTS: int = 180
FIG_WIDTH: float = 12.0
FIG_HEIGHT: float = 5.5


class FunctionBounds:
    """Default search bounds used by all benchmark functions."""

    bounds: dict[str, float] = {"low": -5.12, "high": 5.12}


def sphere(x: np.ndarray) -> float:
    """Sphere function.

    Formula:
        f(x) = sum_{i=1}^{d} x_i^2

    Typical domain bounds:
        x_i in [-5.12, 5.12]

    Global minimum:
        f(x*) = 0 at x* = (0, 0, ..., 0)
    """

    x_arr: np.ndarray = np.asarray(x, dtype=float)
    return float(np.sum(x_arr**2))


sphere.bounds = FunctionBounds.bounds


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function.

    Formula:
        f(x) = 10d + sum_{i=1}^{d} [x_i^2 - 10 cos(2 pi x_i)]

    Typical domain bounds:
        x_i in [-5.12, 5.12]

    Global minimum:
        f(x*) = 0 at x* = (0, 0, ..., 0)
    """

    x_arr: np.ndarray = np.asarray(x, dtype=float)
    dim: int = x_arr.size
    return float(10.0 * dim + np.sum(x_arr**2 - 10.0 * np.cos(PI2 * x_arr)))


rastrigin.bounds = FunctionBounds.bounds


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function.

    Formula:
        f(x) = sum_{i=1}^{d-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Typical domain bounds:
        x_i in [-5.12, 5.12]

    Global minimum:
        f(x*) = 0 at x* = (1, 1, ..., 1)
    """

    x_arr: np.ndarray = np.asarray(x, dtype=float)
    if x_arr.size < 2:
        return float((1.0 - x_arr[0]) ** 2)
    return float(
        np.sum(
            100.0 * (x_arr[1:] - x_arr[:-1] ** 2) ** 2 + (1.0 - x_arr[:-1]) ** 2
        )
    )


rosenbrock.bounds = FunctionBounds.bounds


def ackley(x: np.ndarray) -> float:
    """Ackley function.

    Formula:
        f(x) = -a exp(-b sqrt((1/d) sum x_i^2))
               -exp((1/d) sum cos(c x_i)) + a + e

    Typical domain bounds:
        x_i in [-5.12, 5.12]

    Global minimum:
        f(x*) = 0 at x* = (0, 0, ..., 0)
    """

    x_arr: np.ndarray = np.asarray(x, dtype=float)
    dim: int = x_arr.size
    if dim == 0:
        return 0.0
    sum_sq: float = float(np.sum(x_arr**2))
    sum_cos: float = float(np.sum(np.cos(ACKLEY_C * x_arr)))
    term1: float = -ACKLEY_A * np.exp(-ACKLEY_B * np.sqrt(sum_sq / dim))
    term2: float = -np.exp(sum_cos / dim)
    return float(term1 + term2 + ACKLEY_A + np.e)


ackley.bounds = FunctionBounds.bounds


def plot_2d_surface(
    func: Callable[[np.ndarray], float],
    bounds: dict[str, float],
    title: str,
) -> Figure:
    """Plot a 2D benchmark function as a 3D surface and contour map.

    Args:
        func: Objective function that accepts a 1D numpy array with 2 elements.
        bounds: Dictionary with keys `low` and `high` defining plotting bounds.
        title: Figure title prefix.

    Returns:
        Matplotlib Figure containing a surface subplot and contour subplot.
    """

    low: float = float(bounds["low"])
    high: float = float(bounds["high"])
    axis: np.ndarray = np.linspace(low, high, SURFACE_GRID_POINTS)
    xx, yy = np.meshgrid(axis, axis)
    zz = np.zeros_like(xx)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = func(np.array([xx[i, j], yy[i, j]], dtype=float))

    fig: Figure = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax_surface = fig.add_subplot(1, 2, 1, projection="3d")
    surface = ax_surface.plot_surface(
        xx,
        yy,
        zz,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    fig.colorbar(surface, ax=ax_surface, shrink=0.65, pad=0.10)
    ax_surface.set_title(f"{title} - Surface")
    ax_surface.set_xlabel("x1")
    ax_surface.set_ylabel("x2")
    ax_surface.set_zlabel("f(x)")

    ax_contour = fig.add_subplot(1, 2, 2)
    contour = ax_contour.contourf(xx, yy, zz, levels=50, cmap="viridis")
    fig.colorbar(contour, ax=ax_contour)
    ax_contour.set_title(f"{title} - Contour")
    ax_contour.set_xlabel("x1")
    ax_contour.set_ylabel("x2")

    fig.tight_layout()
    return fig
