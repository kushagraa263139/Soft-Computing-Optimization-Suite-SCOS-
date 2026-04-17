"""Genetic Algorithm implementation for continuous optimization.

This module provides a real-valued GA with tournament/roulette selection,
SBX/BLX-alpha crossover, Gaussian mutation, and elitism.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

TOURNAMENT_SIZE: int = 3
SBX_ETA: float = 20.0
BLX_ALPHA: float = 0.5
MUTATION_SIGMA_SCALE: float = 0.1
PROGRESS_INTERVAL: int = 25
EPSILON: float = 1e-12


class GeneticAlgorithm:
    """Real-valued Genetic Algorithm.

    Biological analogies:
        - Selection: survival of fitter individuals.
        - Crossover: recombination of parental genetic material.
        - Mutation: random variation introducing new traits.
        - Elitism: preserving top individuals across generations.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: dict[str, float],
        n_dim: int,
        pop_size: int = 50,
        max_gen: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.01,
        elitism: int = 2,
        selection: str = "tournament",
        random_state: int = 42,
        crossover_method: str = "sbx",
    ) -> None:
        self.func: Callable[[np.ndarray], float] = func
        self.bounds: dict[str, float] = bounds
        self.low: float = float(bounds["low"])
        self.high: float = float(bounds["high"])
        self.n_dim: int = n_dim
        self.pop_size: int = pop_size
        self.max_gen: int = max_gen
        self.crossover_rate: float = crossover_rate
        self.mutation_rate: float = mutation_rate
        self.elitism: int = elitism
        self.selection: str = selection.lower()
        self.crossover_method: str = crossover_method.lower()
        self.random_state: int = random_state

        self.rng: np.random.Generator = np.random.default_rng(random_state)
        self.best_individual_: np.ndarray | None = None
        self.best_fitness_: float = -np.inf

    def _initialize_population(self) -> np.ndarray:
        """Create random initial population within variable bounds."""

        return self.rng.uniform(self.low, self.high, size=(self.pop_size, self.n_dim))

    def _evaluate_objective(self, population: np.ndarray) -> np.ndarray:
        """Evaluate objective values for all individuals (minimization target)."""

        return np.array([self.func(individual) for individual in population], dtype=float)

    def _evaluate_fitness(self, objective_values: np.ndarray) -> np.ndarray:
        """Convert objective to fitness via negation because we minimize."""

        return -objective_values

    def _select_parent(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Select one parent using configured selection operator."""

        if self.selection == "tournament":
            return self._tournament_selection(population, fitness)
        if self.selection == "roulette":
            return self._roulette_selection(population, fitness)
        raise ValueError("selection must be 'tournament' or 'roulette'.")

    def _tournament_selection(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> np.ndarray:
        """Tournament selection.

        Biological analogy:
            A small local competition where only the fittest candidate survives
            to reproduce.
        """

        idx: np.ndarray = self.rng.choice(
            self.pop_size,
            size=TOURNAMENT_SIZE,
            replace=False,
        )
        winner_idx: int = int(idx[np.argmax(fitness[idx])])
        return population[winner_idx].copy()

    def _roulette_selection(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> np.ndarray:
        """Roulette-wheel selection.

        Biological analogy:
            Individuals occupy wheel slices proportional to reproductive success,
            so fitter chromosomes have higher probability of offspring.
        """

        # Shift fitness to non-negative so probabilities are valid.
        shifted: np.ndarray = fitness - np.min(fitness) + EPSILON
        probs: np.ndarray = shifted / np.sum(shifted)
        idx: int = int(self.rng.choice(self.pop_size, p=probs))
        return population[idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply configured crossover operator to produce two offspring."""

        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        if self.crossover_method == "sbx":
            return self._sbx_crossover(parent1, parent2)
        if self.crossover_method == "blx":
            return self._blx_alpha_crossover(parent1, parent2)
        raise ValueError("crossover_method must be 'sbx' or 'blx'.")

    def _sbx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX).

        Biological analogy:
            Recombination creates children distributed around both parents,
            mimicking inheritance with controllable spread.
        """

        u: np.ndarray = self.rng.random(self.n_dim)
        beta: np.ndarray = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (SBX_ETA + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (SBX_ETA + 1.0)),
        )

        child1: np.ndarray = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
        child2: np.ndarray = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)
        return np.clip(child1, self.low, self.high), np.clip(child2, self.low, self.high)

    def _blx_alpha_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend crossover (BLX-alpha).

        Biological analogy:
            Offspring traits are sampled from an expanded interval around both
            parents, increasing diversity and exploratory behavior.
        """

        min_gene: np.ndarray = np.minimum(parent1, parent2)
        max_gene: np.ndarray = np.maximum(parent1, parent2)
        diff: np.ndarray = max_gene - min_gene

        low_ext: np.ndarray = min_gene - BLX_ALPHA * diff
        high_ext: np.ndarray = max_gene + BLX_ALPHA * diff

        child1: np.ndarray = self.rng.uniform(low_ext, high_ext)
        child2: np.ndarray = self.rng.uniform(low_ext, high_ext)
        return np.clip(child1, self.low, self.high), np.clip(child2, self.low, self.high)

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation on each gene with independent probability.

        Biological analogy:
            Random DNA perturbations introduce novelty that helps escape local
            optima and preserves population diversity.
        """

        sigma: float = MUTATION_SIGMA_SCALE * (self.high - self.low)
        mutation_mask: np.ndarray = self.rng.random(self.n_dim) < self.mutation_rate
        gaussian_noise: np.ndarray = self.rng.normal(0.0, sigma, size=self.n_dim)
        mutated: np.ndarray = individual.copy()
        mutated[mutation_mask] += gaussian_noise[mutation_mask]
        return np.clip(mutated, self.low, self.high)

    def run(self) -> tuple[np.ndarray, float, dict[str, list[float]]]:
        """Run GA optimization and return global best individual and history."""

        population: np.ndarray = self._initialize_population()
        history: dict[str, list[float]] = {"best": [], "mean": [], "std": []}

        for gen in range(1, self.max_gen + 1):
            objective_values: np.ndarray = self._evaluate_objective(population)
            fitness: np.ndarray = self._evaluate_fitness(objective_values)

            best_idx: int = int(np.argmax(fitness))
            if fitness[best_idx] > self.best_fitness_:
                self.best_fitness_ = float(fitness[best_idx])
                self.best_individual_ = population[best_idx].copy()

            history["best"].append(float(np.min(objective_values)))
            history["mean"].append(float(np.mean(objective_values)))
            history["std"].append(float(np.std(objective_values)))

            if gen % PROGRESS_INTERVAL == 0 or gen == 1 or gen == self.max_gen:
                print(
                    f"Generation {gen:3d} | "
                    f"Best objective: {history['best'][-1]:.6f} | "
                    f"Mean objective: {history['mean'][-1]:.6f}"
                )

            elite_idx: np.ndarray = np.argsort(objective_values)[: self.elitism]
            next_population: list[np.ndarray] = [population[i].copy() for i in elite_idx]

            while len(next_population) < self.pop_size:
                parent1: np.ndarray = self._select_parent(population, fitness)
                parent2: np.ndarray = self._select_parent(population, fitness)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                next_population.append(child1)
                if len(next_population) < self.pop_size:
                    next_population.append(child2)

            population = np.vstack(next_population)

        assert self.best_individual_ is not None
        best_objective: float = -self.best_fitness_
        return self.best_individual_, best_objective, history
