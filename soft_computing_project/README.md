# Soft Computing Project: FCM, GA, and PSO

## Project Overview
This project is a complete educational and experimental soft computing toolkit
implemented from first principles in Python. It covers three core paradigms:

- Fuzzy C-Means (FCM) for soft clustering under uncertainty.
- Genetic Algorithms (GA) for evolutionary optimization.
- Particle Swarm Optimization (PSO) for swarm-based continuous search.

The codebase emphasizes clear theory-to-code mapping, reproducibility, and
benchmark-driven comparison.

## Motivation
Real-world optimization and learning tasks are often nonlinear, noisy,
multimodal, and computationally hard. Soft computing methods are designed for
exactly these conditions by trading strict exactness for robust,
high-quality approximate solutions.

## Folder Structure
```text
soft_computing_project/
├── README.md
├── requirements.txt
├── utils/
│   ├── benchmark_functions.py
│   └── visualizer.py
├── 01_soft_computing_basics/
│   └── concepts.py
├── 02_fcm/
│   ├── fcm.py
│   └── fcm_demo.py
├── 03_genetic_algorithm/
│   ├── ga.py
│   └── ga_demo.py
├── 04_pso/
│   ├── pso.py
│   └── pso_demo.py
└── 05_comparison/
    └── benchmark_comparison.py
```

## Installation
From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run
Run each module from the project root `soft_computing_project`.

1. Soft computing concepts tutorial:
```bash
python 01_soft_computing_basics/concepts.py
```

2. FCM implementation demo:
```bash
python 02_fcm/fcm_demo.py
```

3. GA benchmark and sensitivity demo:
```bash
python 03_genetic_algorithm/ga_demo.py
```

4. PSO benchmark, snapshots, and sensitivity demo:
```bash
python 04_pso/pso_demo.py
```

5. Full capstone benchmark comparison:
```bash
python 05_comparison/benchmark_comparison.py
```

## Key Equations
### Fuzzy C-Means
Center update:

$$
\mathbf{v}_j = \frac{\sum_{i=1}^{n} u_{ij}^{m}\,\mathbf{x}_i}{\sum_{i=1}^{n} u_{ij}^{m}}
$$

Membership update:

$$
u_{ij} = \frac{1}{\sum_{k=1}^{c}\left(\frac{d_{ij}}{d_{ik}}\right)^{\frac{2}{m-1}}}
$$

Objective:

$$
J = \sum_{i=1}^{n}\sum_{j=1}^{c} u_{ij}^{m}\,d_{ij}^{2}
$$

### Genetic Algorithm (minimization)
Fitness transform:

$$
\text{fitness}_i = -f(\mathbf{x}_i)
$$

### Particle Swarm Optimization
Velocity update:

$$
\mathbf{v}_i^{t+1} = w\mathbf{v}_i^{t}
+ c_1\mathbf{r}_1\odot(\mathbf{pbest}_i-\mathbf{x}_i^{t})
+ c_2\mathbf{r}_2\odot(\mathbf{gbest}-\mathbf{x}_i^{t})
$$

Position update:

$$
\mathbf{x}_i^{t+1} = \mathbf{x}_i^{t} + \mathbf{v}_i^{t+1}
$$

## Expected Outputs
When running demos, each script:

- Prints key metrics (best fitness, error, convergence information).
- Saves plots as PNG files (`dpi=150`) in the script directory.
- Opens figures interactively via matplotlib.

Typical results include:

- FCM: cluster visualization, membership heatmap, objective convergence,
  ambiguity vs KMeans.
- GA: convergence curves on Sphere/Rastrigin and population-size sensitivity.
- PSO: benchmark summaries, swarm trajectory snapshots, parameter heatmaps,
  inertia strategy comparison.
- Comparison: summary table, boxplots, median convergence subplots, radar chart,
  and written analytical discussion.

## Notes
- All core algorithms are implemented from scratch (no GA/PSO library used).
- Type hints and module docstrings are included across files.
- Constants are defined explicitly to avoid hidden magic numbers.
