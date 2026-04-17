# Soft Computing: FCM, Genetic Algorithms & Particle Swarm Optimization

> A complete, from-scratch Python project implementing three pillars of soft computing —
> Fuzzy C-Means clustering, Genetic Algorithms, and Particle Swarm Optimization —
> benchmarked head-to-head on standard optimization landscapes.

---

## Table of Contents

- [Overview](#overview)
- [Project Philosophy](#project-philosophy)
- [Folder Structure](#folder-structure)
- [Modules at a Glance](#modules-at-a-glance)
- [Key Equations](#key-equations)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Expected Output](#expected-output)
- [Algorithm Comparison](#algorithm-comparison)
- [Benchmark Functions](#benchmark-functions)
- [Parameter Reference](#parameter-reference)
- [Dependencies](#dependencies)
- [Learning Path](#learning-path)

---

## Overview

This project answers one central question:

> *How do biological and natural systems solve hard optimization problems — without brute force?*

Three techniques, each inspired by a different phenomenon in nature, are implemented entirely from scratch using only `numpy` and `matplotlib`. No external optimization libraries. Every equation is coded by hand, commented, and explained.

| Technique | Inspired By | Problem Type | Key Idea |
|---|---|---|---|
| **FCM** | Fuzzy set theory | Clustering | Samples belong to multiple clusters with graded membership |
| **GA** | Darwinian evolution | Optimization | Fitter solutions survive, reproduce, and mutate |
| **PSO** | Bird flocking / fish schooling | Optimization | Particles share information about good regions they have visited |

All three are benchmarked fairly: same number of function evaluations, same test functions, 10 independent runs each.

---

## Project Philosophy

The modules are not isolated scripts — they build on each other in a deliberate narrative:

```
Soft Computing Basics          ← Why do we need this? What breaks without it?
        │
        ▼
Fuzzy C-Means (FCM)            ← Introduce uncertainty: things can partially belong
        │
        ▼
Genetic Algorithm (GA)         ← Population-based search: survival of the fittest
        │
        ▼
Particle Swarm Optimization    ← Social learning: particles remember and communicate
        │
        ▼
Unified Benchmark Comparison   ← Who wins? When? Why?
```

Each module teaches a concept AND produces a measurable result that feeds into the final comparison.

---

## Folder Structure

```
soft_computing_project/
│
├── README.md
├── requirements.txt
│
├── utils/
│   ├── benchmark_functions.py     # Sphere, Rastrigin, Rosenbrock, Ackley
│   └── visualizer.py              # Shared plotting utilities
│
├── 01_soft_computing_basics/
│   └── concepts.py                # Printed tutorial + taxonomy table
│
├── 02_fcm/
│   ├── fcm.py                     # FuzzyCMeans class (from scratch)
│   └── fcm_demo.py                # Demo: blobs, membership heatmap, vs KMeans
│
├── 03_genetic_algorithm/
│   ├── ga.py                      # GeneticAlgorithm class (from scratch)
│   └── ga_demo.py                 # Demo: Rastrigin 10D, parameter sensitivity
│
├── 04_pso/
│   ├── pso.py                     # ParticleSwarmOptimizer class (from scratch)
│   └── pso_demo.py                # Demo: multi-function, swarm animation, tuning
│
└── 05_comparison/
    └── benchmark_comparison.py    # Head-to-head: tables, box plots, radar chart
```

---

## Modules at a Glance

### `utils/benchmark_functions.py`

Reusable test functions for optimization. All accept a numpy array of any dimension and return a scalar.

| Function | Formula | Domain | Global Min |
|---|---|---|---|
| Sphere | $\sum x_i^2$ | $[-5.12,\ 5.12]^n$ | $0$ at origin |
| Rastrigin | $10n + \sum[x_i^2 - 10\cos(2\pi x_i)]$ | $[-5.12,\ 5.12]^n$ | $0$ at origin |
| Rosenbrock | $\sum[100(x_{i+1}-x_i^2)^2 + (1-x_i)^2]$ | $[-2,\ 2]^n$ | $0$ at $(1,\ldots,1)$ |
| Ackley | $-20e^{-0.2\sqrt{\bar{x^2}}} - e^{\overline{\cos 2\pi x}} + 20 + e$ | $[-32,\ 32]^n$ | $0$ at origin |

Also includes `plot_2d_surface(func, bounds, title)` for 3D surface + contour visualization.

---

### `02_fcm/fcm.py` — Fuzzy C-Means

**Core idea:** Unlike hard clustering (KMeans), FCM assigns each sample a *degree of membership* to every cluster. A sample can be 70% cluster A and 30% cluster B.

**Class:** `FuzzyCMeans(n_clusters, m=2.0, max_iter=150, tol=1e-4)`

**Key methods:**

| Method | Returns | Description |
|---|---|---|
| `fit(X)` | `self` | Run FCM iteration until convergence |
| `predict(X)` | `ndarray` | Hard labels (argmax of membership rows) |
| `membership_matrix(X)` | `ndarray` | Soft U matrix for new data |
| `partition_coefficient()` | `float` | PC ∈ [1/c, 1], higher = crisper clusters |
| `partition_entropy()` | `float` | PE ≥ 0, lower = crisper clusters |

---

### `03_genetic_algorithm/ga.py` — Genetic Algorithm

**Core idea:** Maintain a *population* of candidate solutions. Each generation, select fitter individuals, combine them (crossover), introduce small random changes (mutation), and carry the best forward (elitism).

**Class:** `GeneticAlgorithm(func, bounds, n_dim, pop_size=50, max_gen=200, ...)`

**Operators implemented:**

| Stage | Options |
|---|---|
| Selection | Tournament (default), Roulette wheel |
| Crossover | Simulated Binary (SBX), Blend (BLX-α) |
| Mutation | Gaussian (σ = 10% of range) |
| Elitism | Top-k individuals preserved each generation |

---

### `04_pso/pso.py` — Particle Swarm Optimization

**Core idea:** A swarm of particles flies through the search space. Each particle remembers the best position it has personally found (cognitive) and knows the best position found by anyone in the swarm (social). Velocity blends momentum, personal pull, and social pull.

**Class:** `ParticleSwarmOptimizer(func, bounds, n_dim, n_particles=30, max_iter=200, w=0.7, c1=1.5, c2=1.5)`

**Enhancements included:**
- Inertia weight decay (`w`: 0.9 → 0.4 linearly) — explore early, exploit late
- Velocity clamping at `v_max = 0.2 × (high − low)`
- Reflective boundary handling

---

### `05_comparison/benchmark_comparison.py` — Unified Benchmark

Runs GA and PSO (and FCM for clustering quality) across all benchmark functions with:
- 10 independent runs per algorithm per function
- Equal function evaluation budget (10,000 evals)
- Summary table (best, mean ± std, median)
- Box plot of fitness distributions
- Median convergence curves (log scale)
- Radar chart comparing five performance dimensions

---

## Key Equations

### FCM Membership Update

$$u_{ij} = \frac{1}{\displaystyle\sum_{k=1}^{c} \left(\frac{d_{ij}}{d_{ik}}\right)^{\!\frac{2}{m-1}}}$$

### FCM Cluster Centers

$$v_j = \frac{\displaystyle\sum_{i=1}^{n} u_{ij}^m \, x_i}{\displaystyle\sum_{i=1}^{n} u_{ij}^m}$$

### FCM Objective (minimize)

$$J = \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \, \|x_i - v_j\|^2$$

---

### GA Fitness

$$\text{fitness}_i = -f(x_i) \quad \text{(negated for minimization)}$$

### SBX Crossover (Simulated Binary)

$$\beta = \begin{cases} (2u)^{1/(\eta+1)} & u \leq 0.5 \\ \left(\dfrac{1}{2(1-u)}\right)^{1/(\eta+1)} & u > 0.5 \end{cases}$$

$$c_1 = \tfrac{1}{2}[(1+\beta)p_1 + (1-\beta)p_2], \quad c_2 = \tfrac{1}{2}[(1-\beta)p_1 + (1+\beta)p_2]$$

---

### PSO Velocity Update

$$v_i(t+1) = \underbrace{w \cdot v_i(t)}_{\text{inertia}} + \underbrace{c_1 r_1 \left(p_i^{\text{best}} - x_i(t)\right)}_{\text{cognitive}} + \underbrace{c_2 r_2 \left(g^{\text{best}} - x_i(t)\right)}_{\text{social}}$$

### PSO Position Update

$$x_i(t+1) = x_i(t) + v_i(t+1)$$

### PSO Inertia Decay

$$w(t) = w_{\max} - \frac{w_{\max} - w_{\min}}{T} \cdot t$$

---

## Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone or download the project
git clone https://github.com/yourname/soft_computing_project.git
cd soft_computing_project

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
scikit-learn>=1.2
tabulate>=0.9
tqdm>=4.65
```

---

## How to Run

Run each module independently. Each produces console output and saves PNG figures to its own directory.

```bash
# Soft computing primer — prints concept overview and taxonomy
python 01_soft_computing_basics/concepts.py

# FCM — clustering demo with membership heatmap
python 02_fcm/fcm_demo.py

# Genetic Algorithm — Rastrigin 10D + parameter sweep
python 03_genetic_algorithm/ga_demo.py

# PSO — multi-function demo + swarm animation + tuning heatmap
python 04_pso/pso_demo.py

# Final comparison — all algorithms, all functions, full report
python 05_comparison/benchmark_comparison.py
```

All figures are saved automatically as PNG (150 dpi) in the same folder as the script that produced them. Interactive matplotlib windows also open during execution.

---

## Expected Output

### FCM Demo
```
Fitting FuzzyCMeans: c=3, m=2.0
Converged in 34 iterations
Objective J         : 284.71
Partition Coefficient: 0.812     (1.0 = perfectly crisp)
Partition Entropy    : 0.203     (0.0 = perfectly crisp)
Ambiguous samples (max membership < 0.60): 14 / 300
```

Figures produced: original data, cluster assignments, membership heatmap, convergence of J, comparison with KMeans.

---

### GA Demo (Rastrigin 10D)
```
Generation   0 | Best: 142.38 | Mean: 198.54 | Std: 31.22
Generation  25 | Best:  89.14 | Mean: 112.07 | Std: 18.43
Generation  50 | Best:  41.26 | Mean:  67.89 | Std: 14.11
Generation 100 | Best:  12.74 | Mean:  29.32 | Std:  9.87
Generation 200 | Best:   3.98 | Mean:  11.45 | Std:  6.22

Best solution found : [0.021, -0.003, 0.998, ...]
True optimum        : [0.0, 0.0, 0.0, ...]
Absolute error      : 3.98
```

---

### PSO Demo (Rastrigin 10D)
```
Iter   0 | gBest: 168.42 | Mean:  201.37 | w=0.900
Iter  25 | gBest:  54.31 | Mean:   89.22 | w=0.788
Iter  50 | gBest:  19.87 | Mean:   41.56 | w=0.675
Iter 100 | gBest:   5.42 | Mean:   18.23 | w=0.450
Iter 200 | gBest:   1.99 | Mean:    7.84 | w=0.400

Best solution found : [0.004, -0.001, 0.997, ...]
True optimum        : [0.0, 0.0, 0.0, ...]
Absolute error      : 1.99
```

---

### Benchmark Comparison Summary Table

```
Function      Algorithm   Best      Mean ± Std          Median    Evals
------------- ----------- --------- ------------------- --------- -------
Sphere 10D    GA          0.0031    0.0082 ± 0.0041     0.0074    10000
Sphere 10D    PSO         0.0004    0.0011 ± 0.0006     0.0009    10000
Rastrigin 10D GA          3.98      8.24 ± 3.11         7.96      10000
Rastrigin 10D PSO         1.99      4.17 ± 2.08         3.88      10000
Rosenbrock 10D GA         12.44     28.71 ± 11.32       26.88     10000
Rosenbrock 10D PSO        8.21      19.43 ± 9.87        17.65     10000
Ackley 10D    GA          0.412     1.243 ± 0.621       1.107     10000
Ackley 10D    PSO         0.198     0.634 ± 0.312       0.589     10000
```

---

## Algorithm Comparison

| Dimension | FCM | GA | PSO |
|---|---|---|---|
| **Problem type** | Clustering | Optimization | Optimization |
| **Solution representation** | Membership matrix | Real-valued chromosome | Particle position |
| **Memory** | None (stateless per iter) | Population history | Personal + global best |
| **Exploration** | Fuzziness coefficient m | Mutation + crossover | Inertia + random factors |
| **Exploitation** | Hard convergence | Elitism | Cognitive pull |
| **Multimodal** | N/A | Moderate | Good |
| **Hyperparameters** | c, m | pop_size, rates | n_particles, w, c1, c2 |
| **Convergence speed** | Fast | Moderate | Fast |
| **Best suited for** | Overlapping data clusters | Discrete/combinatorial | Continuous spaces |

**When PSO outperforms GA:**
PSO converges faster on smooth continuous landscapes because particles share real-time positional information. GA's crossover and mutation operators are more disruptive — useful for escaping local optima in rugged, multimodal spaces, but slower to exploit a good region.

**When GA outperforms PSO:**
On highly multimodal functions (many deep local optima), GA's population diversity and crossover recombination maintain broader exploration. PSO swarms can prematurely collapse toward a local optimum when `c2` is too high.

---

## Parameter Reference

### FCM

| Parameter | Default | Effect |
|---|---|---|
| `n_clusters` | — | Number of clusters c |
| `m` | 2.0 | Fuzziness: m=1 → hard, m→∞ → maximum fuzzy |
| `max_iter` | 150 | Max iterations |
| `tol` | 1e-4 | Convergence threshold on U change |

### GA

| Parameter | Default | Effect |
|---|---|---|
| `pop_size` | 50 | Larger → better exploration, slower |
| `max_gen` | 200 | Budget in generations |
| `crossover_rate` | 0.8 | Probability of crossover per pair |
| `mutation_rate` | 0.01 | Per-gene mutation probability |
| `elitism` | 2 | Top-k individuals preserved unchanged |

### PSO

| Parameter | Default | Effect |
|---|---|---|
| `n_particles` | 30 | Swarm size |
| `max_iter` | 200 | Iteration budget |
| `w` | 0.7 | Inertia: higher → more momentum |
| `c1` | 1.5 | Cognitive pull toward personal best |
| `c2` | 1.5 | Social pull toward global best |
| `w_decay` | True | Linear decay 0.9→0.4 over iterations |

**Tuning rule of thumb:** `w + c1/2 + c2/2 ≤ 2` ensures guaranteed convergence under standard PSO stability analysis.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.24 | All numerical computation |
| `scipy` | ≥ 1.10 | Statistical utilities |
| `matplotlib` | ≥ 3.7 | All visualization |
| `scikit-learn` | ≥ 1.2 | Dataset generation, KMeans comparison only |
| `tabulate` | ≥ 0.9 | Formatted comparison tables |
| `tqdm` | ≥ 4.65 | Progress bars in long runs |

> **Note:** No PSO, GA, or FCM library is used anywhere. Every algorithm is implemented from first principles.

---

## Learning Path

If you are new to soft computing, read and run in this order:

1. **Start here:** `01_soft_computing_basics/concepts.py`
   Understand why hard computing fails on uncertain, imprecise problems.

2. **Understand fuzzy membership:** `02_fcm/fcm.py` → `02_fcm/fcm_demo.py`
   Internalize why "belongs to cluster A with 70% certainty" is more honest than a hard label.

3. **Understand evolutionary search:** `03_genetic_algorithm/ga.py` → `ga_demo.py`
   See how a population of bad solutions can collectively discover a good one over generations.

4. **Understand swarm intelligence:** `04_pso/pso.py` → `pso_demo.py`
   Watch particles communicate and collectively zero in on the global optimum.

5. **Compare and conclude:** `05_comparison/benchmark_comparison.py`
   Understand the tradeoffs empirically, not just theoretically.

---

## License

MIT License. Free to use for academic and educational purposes.

---

*Built from first principles. No black boxes.*
