"""
forecast/src/eda.py
Quick EDA utilities — saves plots into outputs/eda/
"""
import matplotlib.pyplot as plt, seaborn as sns
import pandas as pd
from pathlib import Path

def correlation_heatmap(df: pd.DataFrame, cols, out_dir: Path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png")
    plt.close()

def pairplot(df: pd.DataFrame, cols, out_dir: Path):
    g = sns.pairplot(df[cols], diag_kind="kde",
                     plot_kws={"alpha": 0.3, "s": 10})
    g.fig.suptitle("Pairplot", y=1.02)
    g.savefig(out_dir / "pairplot.png")
    plt.close()