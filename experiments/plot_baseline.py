import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

HISTORY_DIR = ROOT / "results" / "histories"

FIGURE_DIR = ROOT / "results" / "figures" / "baseline"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = ROOT / "results" / "tables"


histories = {}

for folder in sorted(HISTORY_DIR.glob("baseline_*")):

    history_path = folder / "history.csv"

    if history_path.exists():

        name = folder.name.split("_")[1]

        histories[name] = pd.read_csv(history_path)


def plot_curves(metric, title, ylabel, filename, log=False):

    plt.figure(figsize=(11, 6))

    for name, history in histories.items():

        plt.plot(
            history["epoch"],
            history[metric],
            marker="o",
            markersize=3,
            linewidth=2,
            label=name.upper()
        )

    if log:
        plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_bar(metric, title, ylabel, filename):

    results = pd.read_csv(
        TABLE_DIR / "baseline_comparison.csv"
    )

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        results["optimizer"].str.upper(),
        results[metric]
    )

    plt.bar_label(bars, fmt="%.2f")

    plt.title(title)
    plt.ylabel(ylabel)

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":

    plot_curves(
        "train_loss",
        "Training Loss Comparison",
        "Cross Entropy Loss",
        "training_loss.png",
        log=True
    )

    plot_curves(
        "val_loss",
        "Validation Loss Comparison",
        "Cross Entropy Loss",
        "validation_loss.png",
        log=True
    )

    plot_curves(
        "train_accuracy",
        "Training Accuracy Comparison",
        "Accuracy (%)",
        "training_accuracy.png"
    )

    plot_curves(
        "val_accuracy",
        "Validation Accuracy Comparison",
        "Accuracy (%)",
        "validation_accuracy.png"
    )

    plot_curves(
        "global_gradient_norm",
        "Gradient Norm Comparison",
        "Gradient L2 Norm",
        "gradient_norm.png",
        log=True
    )

    plot_curves(
        "batch_loss_std",
        "Training Stability Comparison",
        "Batch Loss Standard Deviation",
        "training_stability.png",
        log=True
    )

    plot_bar(
        "best_val_accuracy",
        "Best Validation Accuracy",
        "Accuracy (%)",
        "best_accuracy.png"
    )

    plot_bar(
        "final_val_accuracy",
        "Final Validation Accuracy",
        "Accuracy (%)",
        "final_accuracy.png"
    )

    plot_bar(
        "total_train_time_sec",
        "Training Time Comparison",
        "Time (seconds)",
        "training_time.png"
    )

    plot_bar(
        "peak_gpu_memory_mb",
        "GPU Memory Consumption",
        "Allocated Memory (MiB)",
        "gpu_memory.png"
    )

    print("Baseline figures generated successfully.")
    print(f"Saved to: {FIGURE_DIR}")