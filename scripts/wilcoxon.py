import pandas as pd
from scipy.stats import wilcoxon

def run_tests():
    df = pd.read_csv("build/benchmark_results.csv")
    
    # Pivot table to have datasets as rows and classifiers as columns
    metrics = ["accuracy", "f1", "train_ms"]
    
    results = []
    
    # Define pairs to compare
    pairs = [
        ("SHM-NSGA-II", "AM-NSGA-II (Angular)"),
        ("GAM-NSGA-II", "AM-NSGA-II (Angular)"),
        ("SHM-NSGA-II", "SVM-RBF (C=1 γ=1.00)"),
        ("GAM-NSGA-II", "FM-NSGA-II (Fourier)")
    ]
    
    for metric in metrics:
        df_metric = df.pivot(index="dataset", columns="classifier", values=metric)
        
        for c1, c2 in pairs:
            if c1 in df_metric.columns and c2 in df_metric.columns:
                x = df_metric[c1].dropna()
                y = df_metric[c2].dropna()
                
                # We need matched pairs
                common_idx = x.index.intersection(y.index)
                x = x[common_idx]
                y = y[common_idx]
                
                if len(x) >= 5: # Need minimum samples for Wilcoxon
                    stat, p_val = wilcoxon(x, y, zero_method='wilcox', correction=False)
                    # Determine which is better (mean)
                    mean_x = x.mean()
                    mean_y = y.mean()
                    better = c1 if (mean_x > mean_y and metric != "train_ms") or (mean_x < mean_y and metric == "train_ms") else c2
                    
                    results.append({
                        "Metric": metric,
                        "Model A": c1,
                        "Model B": c2,
                        "p-value": p_val,
                        "Better (Mean)": better,
                        "Significant (a=0.05)": "Yes" if p_val < 0.05 else "No"
                    })
    
    res_df = pd.DataFrame(results)
    
    # Fix unicode character and underscores for LaTeX
    res_df = res_df.replace('SVM-RBF (C=1 γ=1.00)', 'SVM-RBF (C=1 $\\gamma$=1.00)', regex=False)
    res_df['Metric'] = res_df['Metric'].replace({'train_ms': 'Time (ms)', 'accuracy': 'Accuracy', 'f1': 'F1 Score'})
    
    # Save to LaTeX
    with open("paper/wilcoxon.tex", "w") as f:
        f.write("\\begin{table*}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Wilcoxon signed-rank test $p$-values across benchmark datasets.}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write(res_df.to_latex(index=False, float_format="%.4f"))
        f.write("}\n")
        f.write("\\label{tab:wilcoxon}\n")
        f.write("\\end{table*}\n")
        
    print(res_df.to_string())
    print("\nSaved LaTeX table to paper/wilcoxon.tex")

if __name__ == "__main__":
    run_tests()
