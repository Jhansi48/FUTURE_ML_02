"""
SupportSense NLP - Visualization & Diagnostic Plotting Module
Generates confusion matrices, class balance charts, model benchmark comparisons,
and feature importance charts.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

def plot_class_distributions(df: pd.DataFrame, figures_dir: str):
    """Plots category and priority frequency distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Category Distribution
    cat_counts = df["Category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    sns.barplot(data=cat_counts, y="Category", x="Count", ax=axes[0], palette="Blues_r")
    axes[0].set_title("Ticket Volume Distribution by Category", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Tickets")
    
    # Priority Distribution
    prio_counts = df["Priority"].value_counts().loc[["High", "Medium", "Low"]].reset_index()
    prio_counts.columns = ["Priority", "Count"]
    sns.barplot(data=prio_counts, x="Priority", y="Count", ax=axes[1], palette=["#e41a1c", "#ff7f00", "#4daf4a"])
    axes[1].set_title("Ticket Volume Distribution by Priority Level", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Number of Tickets")
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "dataset_class_distributions.png"), dpi=300)
    plt.close("all")

def plot_confusion_matrix_heatmap(cm: np.ndarray, labels: list, title: str, output_path: str):
    """Generates normalized confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Ground-Truth Label", fontsize=11)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")

def plot_model_benchmarks(cat_metrics: pd.DataFrame, prio_metrics: pd.DataFrame, output_path: str):
    """Plots side-by-side Macro F1 benchmark comparison for Category and Priority heads."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Category Benchmarks
    cat_df = cat_metrics.sort_values(by="Macro F1", ascending=True)
    axes[0].barh(cat_df["Model"], cat_df["Macro F1"], color="#3182bd", edgecolor="#08519c")
    axes[0].set_title("Head A: Ticket Category Classification (Macro F1)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Macro F1-Score")
    axes[0].set_xlim(0, 1.0)
    for i, v in enumerate(cat_df["Macro F1"]):
        axes[0].text(v + 0.01, i, f"{v:.4f}", va="center", fontweight="bold", fontsize=9)
        
    # Priority Benchmarks
    prio_df = prio_metrics.sort_values(by="Macro F1", ascending=True)
    axes[1].barh(prio_df["Model"], prio_df["Macro F1"], color="#e6550d", edgecolor="#a63603")
    axes[1].set_title("Head B: Ticket Priority Classification (Macro F1)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Macro F1-Score")
    axes[1].set_xlim(0, 1.0)
    for i, v in enumerate(prio_df["Macro F1"]):
        axes[1].text(v + 0.01, i, f"{v:.4f}", va="center", fontweight="bold", fontsize=9)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")
