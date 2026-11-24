import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = ROOT / "results" / "figures" / "momentum"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(
    ROOT / "results" / "tables" / "momentum_comparison.csv"
)


def plot_metric(metric, title, ylabel, filename):

    plt.figure(figsize=(10, 6))

    for optimizer, group in results.groupby("optimizer"):

        group = group.sort_values("momentum")

        plt.plot(
            group["momentum"],
            group[metric],
            marker="o",
            linewidth=2,
            label=optimizer.upper()
        )

    plt.xlabel("Momentum Coefficient")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks([0.5, 0.9, 0.95, 0.99])

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

    plot_metric(
        "best_val_accuracy",
        "Best Accuracy vs Momentum",
        "Accuracy (%)",
        "best_accuracy_vs_momentum.png"
    )

    plot_metric(
        "final_val_accuracy",
        "Final Accuracy vs Momentum",
        "Accuracy (%)",
        "final_accuracy_vs_momentum.png"
    )

    plot_metric(
        "final_val_loss",
        "Validation Loss vs Momentum",
        "Cross Entropy Loss",
        "loss_vs_momentum.png"
    )

    plot_metric(
        "total_train_time_sec",
        "Training Time vs Momentum",
        "Time (seconds)",
        "time_vs_momentum.png"
    )

    print("Momentum figures generated successfully.")