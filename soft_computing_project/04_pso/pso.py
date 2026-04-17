"""Particle Swarm Optimization implementation from first principles.

This module provides a continuous PSO optimizer with inertia handling,
cognitive/social learning terms, velocity clamping, and boundary reflection.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

W_START: float = 0.9
W_END: float = 0.4
VELOCITY_CLAMP_SCALE: float = 0.2
PROGRESS_INTERVAL: int = 25


class ParticleSwarmOptimizer:
    """Particle Swarm Optimizer for minimization tasks."""

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: dict[str, float],
        n_dim: int,
        n_particles: int = 30,
        max_iter: int = 200,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        w_decay: bool = True,
        random_state: int = 42,
    ) -> None:
        self.func: Callable[[np.ndarray], float] = func
        self.bounds: dict[str, float] = bounds
        self.low: float = float(bounds["low"])
        self.high: float = float(bounds["high"])
        self.n_dim: int = n_dim
        self.n_particles: int = n_particles
        self.max_iter: int = max_iter
        self.w: float = w
        self.c1: float = c1
        self.c2: float = c2
        self.w_decay: bool = w_decay
        self.random_state: int = random_state

        self.rng: np.random.Generator = np.random.default_rng(random_state)

    def _get_inertia_weight(self, iteration: int) -> float:
        """Compute inertia weight for current iteration."""

        if not self.w_decay:
            return self.w
        if self.max_iter <= 1:
            return W_END
        ratio: float = (iteration - 1) / float(self.max_iter - 1)
        return W_START - (W_START - W_END) * ratio

    def run(self) -> tuple[np.ndarray, float, dict[str, list[np.ndarray] | list[float]]]:
        """Run PSO optimization and return best solution and evolution history."""

        positions: np.ndarray = self.rng.uniform(
            self.low,
            self.high,
            size=(self.n_particles, self.n_dim),
        )

        velocity_scale: float = VELOCITY_CLAMP_SCALE * (self.high - self.low)
        velocities: np.ndarray = self.rng.uniform(
            -velocity_scale,
            velocity_scale,
            size=(self.n_particles, self.n_dim),
        )

        pbest_positions: np.ndarray = positions.copy()
        pbest_fitness: np.ndarray = np.array(
            [self.func(pos) for pos in positions],
            dtype=float,
        )

        gbest_idx: int = int(np.argmin(pbest_fitness))
        gbest_position: np.ndarray = pbest_positions[gbest_idx].copy()
        gbest_fitness: float = float(pbest_fitness[gbest_idx])

        history: dict[str, list[np.ndarray] | list[float]] = {
            "gbest_fitness": [],
            "mean_fitness": [],
            "gbest_position": [],
            "all_positions": [],
        }

        v_max: float = velocity_scale

        for iteration in range(1, self.max_iter + 1):
            inertia_w: float = self._get_inertia_weight(iteration)

            for i in range(self.n_particles):
                r1: np.ndarray = self.rng.random(self.n_dim)
                r2: np.ndarray = self.rng.random(self.n_dim)

                # Inertia term keeps the particle moving in its previous direction,
                # preserving momentum and supporting broad exploration.
                inertia_term: np.ndarray = inertia_w * velocities[i]

                # Cognitive term pulls particle i toward its own historical best
                # position, encouraging individualized learning/refinement.
                cognitive_term: np.ndarray = (
                    self.c1 * r1 * (pbest_positions[i] - positions[i])
                )

                # Social term pulls particle i toward globally best-known position,
                # enabling group-level information sharing and coordination.
                social_term: np.ndarray = self.c2 * r2 * (gbest_position - positions[i])

                # Full PSO velocity equation combining momentum and both attractors.
                velocities[i] = inertia_term + cognitive_term + social_term

                # Velocity clamping limits step size and prevents instability.
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Position update advances the particle through the search space.
                positions[i] = positions[i] + velocities[i]

                # Reflective boundary handling:
                # if a coordinate exceeds domain, reverse that velocity component
                # and clip position back inside feasible bounds.
                low_mask: np.ndarray = positions[i] < self.low
                high_mask: np.ndarray = positions[i] > self.high
                out_mask: np.ndarray = low_mask | high_mask
                if np.any(out_mask):
                    velocities[i, out_mask] *= -1.0
                    positions[i] = np.clip(positions[i], self.low, self.high)

            current_fitness: np.ndarray = np.array(
                [self.func(pos) for pos in positions],
                dtype=float,
            )

            improved_mask: np.ndarray = current_fitness < pbest_fitness
            pbest_positions[improved_mask] = positions[improved_mask]
            pbest_fitness[improved_mask] = current_fitness[improved_mask]

            best_idx_iter: int = int(np.argmin(pbest_fitness))
            if pbest_fitness[best_idx_iter] < gbest_fitness:
                gbest_fitness = float(pbest_fitness[best_idx_iter])
                gbest_position = pbest_positions[best_idx_iter].copy()

            history["gbest_fitness"].append(gbest_fitness)
            history["mean_fitness"].append(float(np.mean(current_fitness)))
            history["gbest_position"].append(gbest_position.copy())
            history["all_positions"].append(positions.copy())

            if (
                iteration % PROGRESS_INTERVAL == 0
                or iteration == 1
                or iteration == self.max_iter
            ):
                print(
                    f"Iteration {iteration:3d} | "
                    f"gbest fitness: {gbest_fitness:.6f} | "
                    f"w: {inertia_w:.3f}"
                )

        return gbest_position, gbest_fitness, history
