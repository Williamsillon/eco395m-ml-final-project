import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import io

def load_table(path="output/models_table.txt"):
    with open(path, "r") as f:
        lines = f.readlines()

    data_lines = [l for l in lines if "TABLE III" not in l and l.strip()]
    raw = "".join(data_lines)
    df = pd.read_fwf(io.StringIO(raw))
    df.columns = df.columns.str.strip()
    df["Model"] = df["Model"].str.strip()
    df["Accuracy [%]"] = pd.to_numeric(df["Accuracy [%]"], errors="coerce")
    df["ROC-AUC"] = pd.to_numeric(df["ROC-AUC"], errors="coerce")
    df["is_chronos"] = df["Model"].str.startswith("Chronos")
    df["short_label"] = df["Model"].str.replace("Chronos Embeddings + ", "C+ ", regex=False)
    return df

def plot_bar(df, metric, ylabel, title, out_path, xlim):
    df = df.sort_values(metric, ascending=True).reset_index(drop=True)
    colors = ["#d62728" if c else "#1f77b4" for c in df["is_chronos"]]

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(df["short_label"], df[metric], color=colors,
                   edgecolor="white", linewidth=0.5)

    for bar in bars:
        val = bar.get_width()
        label = f"{val:.3f}" if metric == "ROC-AUC" else f"{val:.1f}"
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8)

    ax.set_xlabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(xlim)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    baseline_patch = plt.Rectangle((0, 0), 1, 1, fc="#1f77b4", label="Baseline")
    chronos_patch  = plt.Rectangle((0, 0), 1, 1, fc="#d62728", label="Chronos Embeddings")
    ax.legend(handles=[baseline_patch, chronos_patch], fontsize=10, loc="lower right")

    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    df = load_table("output/models_table.txt")

    plot_bar(
        df,
        metric="Accuracy [%]",
        ylabel="Accuracy [%]",
        title="Model Accuracy Comparison",
        out_path="artifacts/accuracy_chart.png",
        xlim=(60, 97),
    )

    plot_bar(
        df,
        metric="ROC-AUC",
        ylabel="ROC-AUC",
        title="Model ROC-AUC Comparison",
        out_path="artifacts/roc_auc_chart.png",
        xlim=(0.6, 1.02),
    )