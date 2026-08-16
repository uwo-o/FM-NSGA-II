#!/usr/bin/env python3
"""
generate_datasets.py — Genera todos los datasets en CSV para el benchmark.
Requiere: pip install scikit-learn numpy

Uso:
    cd build/
    python3 generate_datasets.py
"""
import os, sys
import numpy as np

try:
    from sklearn.datasets import (
        make_moons, make_circles, make_classification, make_blobs,
        load_iris, load_wine, load_breast_cancer
    )
except ImportError:
    print("Error: pip install scikit-learn numpy")
    sys.exit(1)

os.makedirs("data", exist_ok=True)

def save_csv(path, X, y):
    """Guarda features + label como CSV (última columna = label)."""
    data = np.column_stack([X, y.astype(int)])
    np.savetxt(path, data, delimiter=",", fmt="%.8f",
               header=",".join([f"f{i}" for i in range(X.shape[1])] + ["label"]),
               comments="")
    n_cls = len(np.unique(y))
    print(f"  Guardado: {path}  ({len(X)} muestras, {X.shape[1]}D, {n_cls} clases)")

np.random.seed(42)

# ── Datasets 2D sintéticos ────────────────────────────────────
print("Generando datasets 2D:")

# Two Moons
X, y = make_moons(n_samples=400, noise=0.15, random_state=42)
save_csv("data/moons_2d.csv", X, y)

# Concentric Circles
X, y = make_circles(n_samples=400, noise=0.08, factor=0.5, random_state=42)
save_csv("data/circles_2d.csv", X, y)

# Doble espiral (generado manualmente)
def make_spirals(n=200, noise=0.05):
    n_half = n // 2
    theta = np.linspace(0, 3*np.pi, n_half)
    r = np.linspace(0.1, 1.0, n_half)
    X0 = np.column_stack([r * np.cos(theta) + np.random.randn(n_half)*noise,
                           r * np.sin(theta) + np.random.randn(n_half)*noise])
    X1 = np.column_stack([r * np.cos(theta + np.pi) + np.random.randn(n_half)*noise,
                           r * np.sin(theta + np.pi) + np.random.randn(n_half)*noise])
    X = np.vstack([X0, X1])
    y = np.array([0]*n_half + [1]*n_half)
    return X, y

X, y = make_spirals(400, noise=0.06)
save_csv("data/spirals_2d.csv", X, y)

# Linear Blobs (2 clusters linearly separable)
X, y = make_blobs(n_samples=400, centers=2, cluster_std=1.5, random_state=42)
save_csv("data/blobs_2d.csv", X, y)

# ── Datasets "Imposibles" (Stress Tests) ──────────────────────
print("\nGenerando datasets Imposibles:")

# 1. Chessboard (XOR / Parity Problem) en 2D
def make_chessboard(n_samples=400, grid_size=4, noise=0.05):
    # Genera puntos en [-1, 1] x [-1, 1]
    X = np.random.uniform(-1, 1, size=(n_samples, 2))
    # Calcula el índice de grilla
    x_idx = np.floor((X[:, 0] + 1) * grid_size / 2).astype(int)
    y_idx = np.floor((X[:, 1] + 1) * grid_size / 2).astype(int)
    y = (x_idx + y_idx) % 2
    # Añade ruido ligero
    X += np.random.randn(n_samples, 2) * noise
    return X, y

X_cb, y_cb = make_chessboard(400)
save_csv("data/chessboard_2d.csv", X_cb, y_cb)

# 2. Adversarial Noise (Maldición de la Dimensionalidad)
# Círculos concéntricos en 2D + 50 dimensiones de puro ruido
def make_adversarial_noise(n_samples=400, noise_dims=50):
    X_base, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    # Ruido uniforme en las otras 50 dimensiones
    X_noise = np.random.uniform(-1.5, 1.5, size=(n_samples, noise_dims))
    X = np.hstack([X_base, X_noise])
    return X, y

X_adv, y_adv = make_adversarial_noise(400, noise_dims=50)
save_csv("data/adversarial_52d.csv", X_adv, y_adv)

# ── Datasets reales (UCI) ─────────────────────────────────────
print("\nGenerando datasets reales:")

# Iris (4D, 3 clases)
iris = load_iris()
save_csv("data/iris.csv", iris.data, iris.target)

# Wine (13D, 3 clases)
wine = load_wine()
save_csv("data/wine.csv", wine.data, wine.target)

# Breast Cancer (30D, 2 clases)
bc = load_breast_cancer()
save_csv("data/breast_cancer.csv", bc.data, bc.target)

print("\n✓ Todos los datasets generados en data/")
print("  Ejecutar benchmark: ./benchmark")
