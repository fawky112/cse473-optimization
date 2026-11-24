import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "results" / "tables"

FIGURE_DIR = ROOT / "results" / "figures" / "learning_rate"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


results = pd.read_csv(
    TABLE_DIR / "learning_rate_comparison.csv"
)


def plot_comparison(metric, title, ylabel, filename):

    plt.figure(figsize=(11, 6))

    for name, group in results.groupby("optimizer"):

        group = group.sort_values("lr")

        plt.plot(
            group["lr"],
            group[metric],
            marker="o",
            linewidth=2,
            label=name.upper()
        )

    plt.xscale("log")

    plt.xticks(
        [0.0001, 0.001, 0.01, 0.1],
        ["0.0001", "0.001", "0.01", "0.1"]
    )

    plt.xlabel("Learning Rate")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":

    plot_comparison(
        "best_val_accuracy",
        "Learning Rate Sensitivity",
        "Best Validation Accuracy (%)",
        "best_accuracy_vs_lr.png"
    )

    plot_comparison(
        "final_val_accuracy",
        "Final Accuracy vs Learning Rate",
        "Final Validation Accuracy (%)",
        "final_accuracy_vs_lr.png"
    )

    plot_comparison(
        "final_val_loss",
        "Validation Loss vs Learning Rate",
        "Cross Entropy Loss",
        "validation_loss_vs_lr.png"
    )

    plot_comparison(
        "convergence_epoch",
        "Convergence Speed vs Learning Rate",
        "Epoch to Reach 85% Accuracy",
        "convergence_vs_lr.png"
    )

    print("Learning rate figures generated.")