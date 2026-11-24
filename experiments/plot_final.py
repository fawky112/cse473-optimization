import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "results" / "tables"

FIGURE_DIR = ROOT / "results" / "figures" / "final"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


test = pd.read_csv(
    TABLE_DIR / "test_statistics.csv"
)

stability = pd.read_csv(
    TABLE_DIR / "stability_statistics.csv"
)

configs = pd.read_csv(
    TABLE_DIR / "best_configurations.csv"
)

results = test.merge(
    stability,
    on="optimizer"
)

results = results.merge(
    configs[[
        "optimizer",
        "lr",
        "batch_size",
        "momentum",
        "alpha",
        "betas"
    ]],
    on="optimizer"
)

results = results.sort_values(
    "mean_test_accuracy",
    ascending=False
)

results.to_csv(
    TABLE_DIR / "final_comparison.csv",
    index=False
)


def save_figure(filename):

    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_test_accuracy():

    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        results["optimizer"].str.upper(),
        results["mean_test_accuracy"],
        yerr=results["std_test_accuracy"],
        capsize=6,
        edgecolor="black"
    )

    plt.bar_label(
        bars,
        fmt="%.2f%%",
        padding=5
    )

    plt.ylim(90, 94)

    plt.ylabel("Test Accuracy (%)")

    plt.title(
        "Test Accuracy Across Three Random Seeds"
    )

    save_figure("test_accuracy.png")


def plot_validation_test():

    x = np.arange(len(results))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        results["mean_best_accuracy"],
        width,
        label="Validation"
    )

    plt.bar(
        x + width / 2,
        results["mean_test_accuracy"],
        width,
        label="Test"
    )

    plt.xticks(
        x,
        results["optimizer"].str.upper()
    )

    plt.ylim(90, 94)

    plt.ylabel("Accuracy (%)")
    plt.title("Validation vs Test Accuracy")
    plt.legend()

    save_figure("validation_vs_test.png")


def plot_classification_metrics():

    x = np.arange(len(results))
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width,
        results["mean_precision"],
        width,
        label="Precision"
    )

    plt.bar(
        x,
        results["mean_recall"],
        width,
        label="Recall"
    )

    plt.bar(
        x + width,
        results["mean_f1"],
        width,
        label="F1-score"
    )

    plt.xticks(
        x,
        results["optimizer"].str.upper()
    )

    plt.ylim(0.90, 0.94)

    plt.ylabel("Macro Score")
    plt.title("Final Classification Metrics")
    plt.legend()

    save_figure("classification_metrics.png")


def plot_tradeoff():

    plt.figure(figsize=(11, 7))

    for _, row in results.iterrows():

        plt.scatter(
            row["mean_training_time"],
            row["mean_test_accuracy"],
            s=row["mean_gpu_memory"] * 3,
            edgecolor="black",
            alpha=0.75
        )

        plt.annotate(
            row["optimizer"].upper(),
            (
                row["mean_training_time"],
                row["mean_test_accuracy"]
            ),
            xytext=(7, 7),
            textcoords="offset points"
        )

    plt.xlabel("Mean Training Time (seconds)")
    plt.ylabel("Mean Test Accuracy (%)")

    plt.title(
        "Accuracy-Time Trade-off\n"
        "Marker size represents peak GPU memory"
    )

    plt.margins(x=0.15, y=0.15)

    save_figure("accuracy_time_tradeoff.png")


def plot_bar(metric, ylabel, title, filename):

    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        results["optimizer"].str.upper(),
        results[metric],
        edgecolor="black"
    )

    plt.bar_label(
        bars,
        fmt="%.3f",
        padding=4
    )

    plt.ylabel(ylabel)
    plt.title(title)

    plt.margins(y=0.15)

    save_figure(filename)


if __name__ == "__main__":

    plot_test_accuracy()

    plot_validation_test()

    plot_classification_metrics()

    plot_tradeoff()

    plot_bar(
        "mean_test_loss",
        "Cross Entropy Loss",
        "Mean Test Loss",
        "test_loss.png"
    )

    plot_bar(
        "mean_training_time",
        "Time (seconds)",
        "Mean Training Time",
        "training_time.png"
    )

    plot_bar(
        "mean_gpu_memory",
        "Allocated Memory (MiB)",
        "Mean Peak GPU Memory",
        "gpu_memory.png"
    )

    plot_bar(
        "std_test_accuracy",
        "Standard Deviation (percentage points)",
        "Test Accuracy Repeatability",
        "test_stability.png"
    )

    plot_bar(
        "mean_convergence_time",
        "Time (seconds)",
        "Mean Time to Reach 85% Validation Accuracy",
        "convergence_time.png"
    )

    print("Final comparison table generated.")

    print("Final figures generated successfully.")

    print(f"Saved to: {FIGURE_DIR}")