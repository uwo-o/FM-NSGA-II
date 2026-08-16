import os
import sys
import pandas as pd
import subprocess
import numpy as np

N_RUNS = 20
BUILD_DIR = "../build"
CSV_FILE = "benchmark_results.csv"

all_data = []

print(f"Iniciando evaluación de varianza: {N_RUNS} ejecuciones de benchmark...")

for i in range(N_RUNS):
    print(f"--- Ejecución {i+1}/{N_RUNS} ---")
    
    # Run the benchmark binary
    process = subprocess.run(["./benchmark"], cwd=BUILD_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        print(f"Error en ejecución {i+1}:")
        print(process.stderr)
        continue
    
    # Read the generated CSV
    csv_path = os.path.join(BUILD_DIR, CSV_FILE)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['run'] = i + 1
        all_data.append(df)
    else:
        print(f"Archivo {CSV_FILE} no encontrado en la iteración {i+1}")

if not all_data:
    print("No se pudo recolectar ningún dato.")
    sys.exit(1)

# Concatenate all runs
full_df = pd.concat(all_data, ignore_index=True)

# Group by dataset and classifier to calculate mean and std
grouped = full_df.groupby(['dataset', 'classifier']).agg(
    accuracy_mean=('accuracy', 'mean'),
    accuracy_std=('accuracy', 'std'),
    f1_mean=('f1', 'mean'),
    f1_std=('f1', 'std'),
    train_ms_mean=('train_ms', 'mean'),
    train_ms_std=('train_ms', 'std')
).reset_index()

# Handle NaNs for determinisitic algorithms (std will be NaN if all values are identical in some pandas versions, though here it'll be 0.0)
grouped.fillna(0.0, inplace=True)

print("\n=== RESULTADOS FINALES DE VARIANZA (10 RUNS) ===")
for ds in grouped['dataset'].unique():
    print(f"\nDataset: {ds}")
    ds_data = grouped[grouped['dataset'] == ds]
    for _, row in ds_data.iterrows():
        clf = row['classifier']
        acc = row['accuracy_mean']
        acc_std = row['accuracy_std']
        ms = row['train_ms_mean']
        ms_std = row['train_ms_std']
        print(f"  - {clf:15s}: Acc = {acc:.4f} ± {acc_std:.4f} | Time = {int(ms)}ms ± {int(ms_std)}ms")

# Save detailed results
out_path = os.path.join(BUILD_DIR, "variance_results.csv")
grouped.to_csv(out_path, index=False)
print(f"\nResultados con varianza exportados a: {out_path}")

# Overwrite benchmark_results.csv with means so visualizers still work but use stable data
stable_df = grouped.copy()
stable_df.rename(columns={
    'accuracy_mean': 'accuracy',
    'f1_mean': 'f1',
    'train_ms_mean': 'train_ms'
}, inplace=True)
stable_df = stable_df[['dataset', 'classifier', 'accuracy', 'f1', 'train_ms']]
stable_df.to_csv(os.path.join(BUILD_DIR, CSV_FILE), index=False)
print(f"Archivo {CSV_FILE} reescrito con los valores promedio para graficación automática.")
