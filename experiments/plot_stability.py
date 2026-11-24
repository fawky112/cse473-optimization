import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = ROOT / "results" / "figures" / "stability"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(
    ROOT / "results" / "tables" / "stability_statistics.csv"
)

results = results[
    results["successful_runs"] >= 2
].sort_values(
    "mean_best_accuracy",
    ascending=False
)


def plot_error_bar(mean, std, title, ylabel, filename):

    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        results["optimizer"].str.upper(),
        results[mean],
        yerr=results[std],
        capsize=6,
        edgecolor="black",
        alpha=0.8
    )

    labels = [
        f"{m:.3f} ± {s:.3f}"
        for m, s in zip(results[mean], results[std])
    ]

    plt.bar_label(
        bars,
        labels=labels,
        padding=8,
        fontsize=9
    )

    if "accuracy" in mean:
        plt.ylim(90, 95)
    else:
        plt.ylim(0, max(results[mean] + results[std]) * 1.15)

    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":

    plot_error_bar(
        "mean_best_accuracy",
        "std_best_accuracy",
        "Best Validation Accuracy - Mean ± SD",
        "Accuracy (%)",
        "best_accuracy_stability.png"
    )

    plot_error_bar(
        "mean_final_accuracy",
        "std_final_accuracy",
        "Final Validation Accuracy - Mean ± SD",
        "Accuracy (%)",
        "final_accuracy_stability.png"
    )

    plot_error_bar(
        "mean_training_time",
        "std_training_time",
        "Training Time - Mean ± SD",
        "Time (seconds)",
        "training_time_stability.png"
    )

    print("Stability figures generated successfully.")
    print(f"Saved to: {FIGURE_DIR}")