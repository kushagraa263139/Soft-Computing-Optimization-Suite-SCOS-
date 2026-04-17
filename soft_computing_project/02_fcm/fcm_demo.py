"""Demonstration of Fuzzy C-Means clustering on overlapping Gaussian blobs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from fcm import FuzzyCMeans

SEED: int = 42
N_SAMPLES: int = 600
N_FEATURES: int = 2
N_CLUSTERS: int = 3
BLOB_STD: float = 1.8
FCM_M: float = 2.0
FCM_MAX_ITER: int = 200
FCM_TOL: float = 1e-5
DPI: int = 150
AMBIGUITY_THRESHOLD: float = 0.6
FIG_SIZE_SCATTER: tuple[float, float] = (7.0, 6.0)
FIG_SIZE_HEATMAP: tuple[float, float] = (9.0, 4.5)
FIG_SIZE_CONV: tuple[float, float] = (8.0, 4.5)
SCATTER_SIZE: int = 20
SCATTER_SIZE_HARD: int = 22
CENTER_SIZE: int = 220
LINE_WIDTH: float = 2.0
GRID_ALPHA: float = 0.4
HEADING_LINE_WIDTH: int = 56


def save_and_show(fig: plt.Figure, output_path: Path) -> None:
    """Save figure to PNG and show it interactively."""

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.show()


def main() -> None:
    """Run FCM demo with diagnostics and KMeans comparison."""

    output_dir: Path = Path(__file__).resolve().parent

    X, _ = make_blobs(
        n_samples=N_SAMPLES,
        centers=N_CLUSTERS,
        n_features=N_FEATURES,
        cluster_std=BLOB_STD,
        random_state=SEED,
    )

    fcm = FuzzyCMeans(
        n_clusters=N_CLUSTERS,
        m=FCM_M,
        max_iter=FCM_MAX_ITER,
        tol=FCM_TOL,
        random_state=SEED,
    )
    fcm.fit(X)

    assert fcm.U_ is not None
    assert fcm.centers_ is not None
    assert fcm.labels_ is not None

    final_objective: float = fcm.history_J_[-1]
    pc: float = fcm.partition_coefficient()
    pe: float = fcm.partition_entropy()

    print("FCM Results")
    print("=" * HEADING_LINE_WIDTH)
    print(f"Final objective J: {final_objective:.6f}")
    print(f"Partition coefficient (PC): {pc:.6f}")
    print(f"Partition entropy (PE): {pe:.6f}")
    print(f"Iterations to converge: {fcm.n_iter_}")

    # Figure A: original unlabeled data.
    fig_a, ax_a = plt.subplots(figsize=FIG_SIZE_SCATTER)
    ax_a.scatter(X[:, 0], X[:, 1], s=SCATTER_SIZE, alpha=0.7, c="steelblue")
    ax_a.set_title("Original Overlapping Data")
    ax_a.set_xlabel("x1")
    ax_a.set_ylabel("x2")
    ax_a.grid(True, linestyle=":", alpha=GRID_ALPHA)
    save_and_show(fig_a, output_dir / "fcm_original_data.png")

    # Figure B: FCM hard labels + fuzzy centers.
    fig_b, ax_b = plt.subplots(figsize=FIG_SIZE_SCATTER)
    scatter_b = ax_b.scatter(
        X[:, 0],
        X[:, 1],
        c=fcm.labels_,
        cmap="viridis",
        s=SCATTER_SIZE_HARD,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.2,
    )
    ax_b.scatter(
        fcm.centers_[:, 0],
        fcm.centers_[:, 1],
        c="red",
        marker="X",
        s=CENTER_SIZE,
        edgecolors="white",
        linewidths=1.0,
        label="FCM Centers",
    )
    fig_b.colorbar(scatter_b, ax=ax_b, label="Hard Label")
    ax_b.set_title("FCM Clustering Result")
    ax_b.set_xlabel("x1")
    ax_b.set_ylabel("x2")
    ax_b.legend()
    ax_b.grid(True, linestyle=":", alpha=GRID_ALPHA)
    save_and_show(fig_b, output_dir / "fcm_cluster_result.png")

    # Figure C: membership matrix heatmap.
    fig_c, ax_c = plt.subplots(figsize=FIG_SIZE_HEATMAP)
    im = ax_c.imshow(fcm.U_, aspect="auto", cmap="viridis")
    fig_c.colorbar(im, ax=ax_c, label="Membership Degree")
    ax_c.set_title("FCM Membership Matrix U")
    ax_c.set_xlabel("Cluster Index")
    ax_c.set_ylabel("Sample Index")
    save_and_show(fig_c, output_dir / "fcm_membership_heatmap.png")

    # Figure D: objective convergence.
    fig_d, ax_d = plt.subplots(figsize=FIG_SIZE_CONV)
    ax_d.plot(fcm.history_J_, color="darkorange", linewidth=LINE_WIDTH)
    ax_d.set_title("FCM Objective Convergence")
    ax_d.set_xlabel("Iteration")
    ax_d.set_ylabel("Objective J")
    ax_d.grid(True, linestyle="--", alpha=GRID_ALPHA)
    save_and_show(fig_d, output_dir / "fcm_convergence.png")

    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=SEED)
    kmeans_labels: np.ndarray = kmeans.fit_predict(X)

    ambiguous_mask: np.ndarray = np.max(fcm.U_, axis=1) < AMBIGUITY_THRESHOLD
    disagree_mask: np.ndarray = fcm.labels_ != kmeans_labels

    print("\nKMeans Comparison")
    print("=" * HEADING_LINE_WIDTH)
    print(f"Number of ambiguous points (max membership < 0.6): {ambiguous_mask.sum()}")
    print(f"Number of FCM vs KMeans hard-label disagreements: {disagree_mask.sum()}")

    fig_e, ax_e = plt.subplots(figsize=FIG_SIZE_SCATTER)
    ax_e.scatter(
        X[:, 0],
        X[:, 1],
        c="lightgray",
        s=20,
        alpha=0.5,
        label="All Points",
    )
    ax_e.scatter(
        X[disagree_mask, 0],
        X[disagree_mask, 1],
        c="royalblue",
        s=40,
        alpha=0.8,
        label="FCM != KMeans",
    )
    ax_e.scatter(
        X[ambiguous_mask, 0],
        X[ambiguous_mask, 1],
        c="red",
        s=45,
        alpha=0.9,
        marker="o",
        edgecolors="white",
        linewidths=0.6,
        label="Ambiguous (max U < 0.6)",
    )
    ax_e.set_title("Agreement Analysis: FCM vs KMeans")
    ax_e.set_xlabel("x1")
    ax_e.set_ylabel("x2")
    ax_e.grid(True, linestyle=":", alpha=GRID_ALPHA)
    ax_e.legend(loc="best")
    save_and_show(fig_e, output_dir / "fcm_kmeans_disagreement.png")


if __name__ == "__main__":
    main()
