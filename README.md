# FM-NSGA-II Classifier

Multi-class non-linear classifier based on **NSGA-II** with decision boundaries
represented as **closed parametric Fourier curves** in R^n.

## Core Idea (White-Box Topology)

Each class has an associated **manifold** (closed parametric curve) in $\mathbb{R}^D$. A point is classified into the class whose manifold "envelopes" it the most.

The parametric curve $\mathbf{f}(t)$ is represented via a truncated Fourier series for each dimension $d$:

```math
f_d(t) = a_{0,d} + \sum_{k=1}^{H} \left( a_{k,d} \cos(kt) + b_{k,d} \sin(kt) \right)
```
Where:
- $t \in [0, 2\pi)$ is the intrinsic parameter.
- $H$ is the number of harmonics.
- $a_{0,d}$ is the centroid of the manifold.
- $a_{k,d}, b_{k,d}$ control amplitude and phase.

### Co-evolved Feature Weights ($\mathbf{w}$)
To perform intrinsic feature selection in high dimensions, a weight vector $\mathbf{w} \in [0,1]^D$ is co-evolved within the chromosome. This induces a **weighted Euclidean distance**:
```math
d_w(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{d=1}^{D} w_d (x_d - y_d)^2}
```

### Radial Signed Distance ($D_S$)
To determine if a point $\mathbf{x}$ is inside the curve without relying on computationally unfeasible Ray-Casting in $D \ge 3$, the algorithm uses a radial comparison to the centroid $\mathbf{c}$:
```math
D_S(\mathbf{x}, M) = d_w(\mathbf{x}, \mathbf{c}) - d_w(\mathbf{f}(t^*), \mathbf{c})
```
Where $\mathbf{f}(t^*)$ is the projection of $\mathbf{x}$ onto the curve. If $D_S < 0$, the point is classified as **inside**.

### Multiclass (One-vs-Rest)
For $K$ classes, $K$ independent populations evolve manifolds. A new sample $\mathbf{x}$ is assigned to the class that yields the minimum signed distance:
```math
\hat{y} = \arg\min_{k \in \{1,\dots,K\}} D_S(\mathbf{x}, M_k)
```

NSGA-II optimizes **simultaneously**:
1. **Minimize classification error** (Accuracy)
2. **Minimize manifold complexity** (Occam's Razor: arc length + number of harmonics)

The chromosome has a **variable length** ($H$ varies), and mutation is ERC (Ephemeral Random Constant) combined with Polynomial/Gaussian noise.

## Project Structure

```
fm-nsga2/
├── include/
│   ├── utils.hpp          ← RNG, vector algebra, Golden Section
│   ├── dataset.hpp        ← CSV loading, normalization, split, one-vs-rest
│   ├── manifold.hpp       ← FourierManifold in R^n (algorithm core)
│   ├── objectives.hpp     ← Bi-objective functions
│   ├── operators.hpp      ← ERC mutation + GP-style crossover
│   ├── metrics.hpp        ← Accuracy, F1, macro-F1
│   └── nsga2.hpp          ← NSGA-II engine + FMClassifier (multi-class)
├── baselines/
│   ├── classifier_base.hpp ← IClassifier abstract interface
│   ├── knn.hpp             ← Euclidean k-NN
│   ├── naive_bayes.hpp     ← Gaussian Naive Bayes
│   ├── svm.hpp             ← SVM with RBF kernel (simplified SMO)
│   └── decision_tree.hpp   ← CART Tree (Gini impurity)
├── src/
│   └── main.cpp            ← Quick demo
├── experiments/
│   ├── benchmark.cpp       ← Full multi-dataset benchmark
│   └── visualize.py        ← 2D visualization and Pareto front
├── data/
│   └── generate_datasets.py ← Generates CSVs (sklearn)
└── CMakeLists.txt
```

## Compilation

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

## Usage

### 1. Generate datasets
```bash
cd build
python3 generate_datasets.py
```

### 2. Quick demo (Two Moons)
```bash
./fm_nsga2 data/moons_2d.csv
```

### 3. Full benchmark (all datasets)
```bash
./benchmark
```

### 4. Visualization
```bash
python3 visualize.py
```

## Benchmark Datasets

| Dataset | Dim | Classes | Samples | Type |
|---|---|---|---|---|
| Two Moons | 2 | 2 | 400 | Synthetic |
| Circles | 2 | 2 | 400 | Synthetic |
| Spirals | 2 | 2 | 400 | Synthetic |
| Iris | 4 | 3 | 150 | Real (UCI) |
| Wine | 13 | 3 | 178 | Real (UCI) |
| Breast Cancer | 30 | 2 | 569 | Real (UCI) |

## Configurable Parameters (NSGAConfig)

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 80 | Population size |
| `max_gen` | 200 | Maximum generations |
| `p_cross` | 0.9 | Crossover probability |
| `min_harmonics` | 1 | Minimum harmonics (N_min) |
| `max_harmonics` | 8 | Maximum harmonics (N_max) |
| `p_coef` | 0.12 | Probability to mutate each coefficient (ERC) |
| `p_add` | 0.06 | Probability to add a harmonic |
| `p_del` | 0.04 | Probability to delete a harmonic |
| `lambda` | 0.5 | Weight between arc length and harmonic count in objective 2 |

## References

- Deb, K. et al. (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II*. IEEE TEVC.
- Platt, J. (1998). *Sequential Minimal Optimization*. Advances in Kernel Methods.
- CART: Breiman et al. (1984). *Classification and Regression Trees*.
