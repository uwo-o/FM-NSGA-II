# FM-NSGA-II Classifier

Clasificador no-lineal multi-clase basado en **NSGA-II** con fronteras de decisión
representadas como **curvas de Fourier paramétricas cerradas** en R^n.

## Idea central

Cada clase tiene un **manifold** (curva cerrada) asociado. Un punto se clasifica
en la clase cuyo manifold lo "envuelve más" (mínima distancia firmada).

La curva se representa como:

```
γ_i(t) = c_i + Σ_{k=1}^{N} [ a_{ik}·cos(kt) + b_{ik}·sin(kt) ]
```

NSGA-II optimiza **simultáneamente**:
1. **Minimizar error de clasificación** (accuracy)
2. **Minimizar complejidad del manifold** (longitud de arco + número de armónicos)

El cromosoma tiene **longitud variable** (N varía), y la mutación es ERC.

## Estructura del proyecto

```
rm-nsga2/
├── include/
│   ├── utils.hpp          ← RNG, álgebra vectorial, Golden Section
│   ├── dataset.hpp        ← Carga CSV, normalización, split, one-vs-rest
│   ├── manifold.hpp       ← FourierManifold en R^n (núcleo del algoritmo)
│   ├── objectives.hpp     ← Funciones objetivo bi-objetivo
│   ├── operators.hpp      ← Mutación ERC + cruzamiento GP-style
│   ├── metrics.hpp        ← Accuracy, F1, macro-F1
│   └── nsga2.hpp          ← Motor NSGA-II + FMClassifier (multi-clase)
├── baselines/
│   ├── classifier_base.hpp ← Interfaz abstracta IClassifier
│   ├── knn.hpp             ← k-NN euclídeo
│   ├── naive_bayes.hpp     ← Gaussian Naive Bayes
│   ├── svm.hpp             ← SVM con kernel RBF (SMO simplificado)
│   └── decision_tree.hpp   ← Árbol CART (Gini impurity)
├── src/
│   └── main.cpp            ← Demo rápido
├── experiments/
│   ├── benchmark.cpp       ← Benchmark multi-dataset completo
│   └── visualize.py        ← Visualización 2D y frente de Pareto
├── data/
│   └── generate_datasets.py ← Genera CSVs (sklearn)
└── CMakeLists.txt
```

## Compilación

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

## Uso

### 1. Generar datasets
```bash
cd build
python3 generate_datasets.py
```

### 2. Demo rápido (Two Moons)
```bash
./fm_nsga2 data/moons_2d.csv
```

### 3. Benchmark completo (todos los datasets)
```bash
./benchmark
```

### 4. Visualización
```bash
python3 visualize.py
```

## Datasets de benchmark

| Dataset | Dim | Clases | Muestras | Tipo |
|---|---|---|---|---|
| Two Moons | 2 | 2 | 400 | Sintético |
| Circles | 2 | 2 | 400 | Sintético |
| Spirals | 2 | 2 | 400 | Sintético |
| Iris | 4 | 3 | 150 | Real (UCI) |
| Wine | 13 | 3 | 178 | Real (UCI) |
| Breast Cancer | 30 | 2 | 569 | Real (UCI) |

## Parámetros configurables (NSGAConfig)

| Parámetro | Default | Descripción |
|---|---|---|
| `pop_size` | 80 | Tamaño de la población |
| `max_gen` | 200 | Generaciones máximas |
| `p_cross` | 0.9 | Probabilidad de cruzamiento |
| `min_harmonics` | 1 | Mínimo de armónicos (N_min) |
| `max_harmonics` | 8 | Máximo de armónicos (N_max) |
| `p_coef` | 0.12 | Probabilidad de mutar cada coeficiente (ERC) |
| `p_add` | 0.06 | Probabilidad de agregar un harmónico |
| `p_del` | 0.04 | Probabilidad de eliminar un harmónico |
| `lambda` | 0.5 | Peso entre arco y conteo en objetivo 2 |

## Referencias

- Deb, K. et al. (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II*. IEEE TEVC.
- Platt, J. (1998). *Sequential Minimal Optimization*. Advances in Kernel Methods.
- CART: Breiman et al. (1984). *Classification and Regression Trees*.
