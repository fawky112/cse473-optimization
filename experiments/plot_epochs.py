import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = ROOT / "results" / "figures" / "epochs"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(
    ROOT / "results" / "tables" / "epoch_comparison.csv"
)


def plot_metric(metric, title, ylabel, filename):

    plt.figure(figsize=(11, 6))

    for optimizer, group in results.groupby("optimizer"):

        group = group.sort_values("epoch_budget")

        plt.plot(
            group["epoch_budget"],
            group[metric],
            marker="o",
            linewidth=2,
            label=optimizer.upper()
        )

    plt.xticks([5, 10, 15, 20])

    plt.xlabel("Number of Epochs")
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


if __name__ == "__main__":

    metrics = [
        (
            "best_val_accuracy",
            "Best Accuracy vs Epoch Budget",
            "Accuracy (%)",
            "best_accuracy.png"
        ),
        (
            "final_val_accuracy",
            "Final Accuracy vs Epoch Budget",
            "Accuracy (%)",
            "final_accuracy.png"
        ),
        (
            "final_train_loss",
            "Training Loss vs Epochs",
            "Cross Entropy Loss",
            "training_loss.png"
        ),
        (
            "final_val_loss",
            "Validation Loss vs Epochs",
            "Cross Entropy Loss",
            "validation_loss.png"
        ),
        (
            "accuracy_gap",
            "Generalization Gap vs Epochs",
            "Accuracy Gap (percentage points)",
            "generalization_gap.png"
        ),
        (
            "total_train_time_sec",
            "Computational Time vs Epochs",
            "Time (seconds)",
            "training_time.png"
        )
    ]

    for metric in metrics:

        plot_metric(*metric)

    print("Epoch analysis figures generated.")