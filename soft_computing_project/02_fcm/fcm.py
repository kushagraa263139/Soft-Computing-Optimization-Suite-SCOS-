"""Fuzzy C-Means (FCM) clustering implementation from first principles.

This module implements the FCM optimization loop using only NumPy, including
soft memberships, centroid updates, objective history, and cluster validity
indices (partition coefficient and partition entropy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON: float = 1e-12
LOG_EPSILON: float = 1e-12


@dataclass
class FCMState:
    """Small container for intermediate FCM optimization state."""

    U: np.ndarray
    centers: np.ndarray
    distances: np.ndarray
    objective: float


class FuzzyCMeans:
    """Fuzzy C-Means clustering.

    Args:
        n_clusters: Number of clusters c.
        m: Fuzziness coefficient (> 1). As m -> 1, clustering becomes harder.
        max_iter: Maximum number of iterations.
        tol: Stop when max absolute change in U falls below tol.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        max_iter: int = 150,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2.")
        if m <= 1.0:
            raise ValueError("m must be > 1.0 for valid fuzzy memberships.")

        self.n_clusters: int = n_clusters
        self.m: float = m
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.random_state: int = random_state

        self.centers_: np.ndarray | None = None
        self.U_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.history_J_: list[float] = []
        self.n_iter_: int = 0

    def _initialize_membership(self, n_samples: int) -> np.ndarray:
        """Initialize random memberships and row-normalize to sum to 1."""

        rng: np.random.Generator = np.random.default_rng(self.random_state)
        U: np.ndarray = rng.random((n_samples, self.n_clusters))
        row_sums: np.ndarray = U.sum(axis=1, keepdims=True)
        return U / np.maximum(row_sums, EPSILON)

    def _compute_centers(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Compute fuzzy cluster centers using membership powers.

        Formula:
            v_j = sum_i(u_ij^m * x_i) / sum_i(u_ij^m)
        """

        # Raise memberships to power m to control fuzziness influence.
        um: np.ndarray = U**self.m

        # Numerator aggregates each feature weighted by fuzzy membership strength.
        numerator: np.ndarray = um.T @ X

        # Denominator is total fuzzy mass per cluster.
        denominator: np.ndarray = np.sum(um, axis=0, keepdims=True).T

        return numerator / np.maximum(denominator, EPSILON)

    def _compute_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between all samples and all centers."""

        # Broadcasting: (n,1,d) - (1,c,d) -> (n,c,d), then L2 norm over features.
        diff: np.ndarray = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances: np.ndarray = np.linalg.norm(diff, axis=2)
        return np.maximum(distances, EPSILON)

    def _update_membership(self, distances: np.ndarray) -> np.ndarray:
        """Update fuzzy memberships from distance ratios.

        Formula:
            u_ij = 1 / sum_k( (d_ij / d_ik)^(2/(m-1)) )
        """

        n_samples, n_clusters = distances.shape
        power: float = 2.0 / (self.m - 1.0)
        U_new: np.ndarray = np.zeros((n_samples, n_clusters), dtype=float)

        for i in range(n_samples):
            di: np.ndarray = distances[i]

            # If a sample exactly matches a center (distance approx 0), assign full
            # membership to those zero-distance centers and 0 to others.
            zero_mask: np.ndarray = di <= EPSILON
            if np.any(zero_mask):
                count: int = int(np.sum(zero_mask))
                U_new[i, zero_mask] = 1.0 / count
                continue

            # Compute pairwise ratio matrix (d_ij / d_ik)^power for fixed sample i.
            ratio_matrix: np.ndarray = (di[:, np.newaxis] / di[np.newaxis, :]) ** power

            # Denominator per j is sum_k of the ratio row for j.
            denom: np.ndarray = np.sum(ratio_matrix, axis=1)
            U_new[i] = 1.0 / np.maximum(denom, EPSILON)

        # Normalize rows to maintain the probabilistic constraint sum_j u_ij = 1.
        U_new /= np.maximum(U_new.sum(axis=1, keepdims=True), EPSILON)
        return U_new

    def _objective(self, U: np.ndarray, distances: np.ndarray) -> float:
        """Compute FCM objective value.

        Formula:
            J = sum_i sum_j (u_ij^m) * d_ij^2
        """

        return float(np.sum((U**self.m) * (distances**2)))

    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        """Fit FCM to data matrix X with shape (n_samples, n_features)."""

        X_arr: np.ndarray = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        n_samples: int = X_arr.shape[0]
        U: np.ndarray = self._initialize_membership(n_samples)
        self.history_J_ = []

        for iteration in range(1, self.max_iter + 1):
            centers: np.ndarray = self._compute_centers(X_arr, U)
            distances: np.ndarray = self._compute_distances(X_arr, centers)
            U_new: np.ndarray = self._update_membership(distances)
            J_val: float = self._objective(U_new, distances)
            self.history_J_.append(J_val)

            delta: float = float(np.max(np.abs(U_new - U)))
            U = U_new

            if delta < self.tol:
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter

        self.centers_ = self._compute_centers(X_arr, U)
        self.U_ = U
        self.labels_ = np.argmax(U, axis=1)
        return self

    def membership_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute soft membership matrix U for new samples."""

        if self.centers_ is None:
            raise ValueError("Model must be fitted before calling membership_matrix.")

        X_arr: np.ndarray = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")

        distances: np.ndarray = self._compute_distances(X_arr, self.centers_)
        return self._update_membership(distances)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard labels via argmax on fuzzy memberships."""

        U_new: np.ndarray = self.membership_matrix(X)
        return np.argmax(U_new, axis=1)

    def partition_coefficient(self) -> float:
        """Compute partition coefficient PC in [1/c, 1], where 1 is crisp."""

        if self.U_ is None:
            raise ValueError("Model must be fitted before computing PC.")
        n_samples: int = self.U_.shape[0]
        return float(np.sum(self.U_**2) / n_samples)

    def partition_entropy(self) -> float:
        """Compute partition entropy PE, where 0 indicates crisp memberships."""

        if self.U_ is None:
            raise ValueError("Model must be fitted before computing PE.")
        n_samples: int = self.U_.shape[0]
        U_safe: np.ndarray = np.clip(self.U_, LOG_EPSILON, 1.0)
        return float(-np.sum(U_safe * np.log(U_safe)) / n_samples)
