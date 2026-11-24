import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = ROOT / "results" / "figures" / "batch_size"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(
    ROOT / "results" / "tables" / "batch_size_comparison.csv"
)


def plot_metric(metric, title, ylabel, filename):

    plt.figure(figsize=(11, 6))

    for optimizer, group in results.groupby("optimizer"):

        group = group.sort_values("batch_size")

        plt.plot(
            group["batch_size"],
            group[metric],
            marker="o",
            linewidth=2,
            label=optimizer.upper()
        )

    plt.xscale("log", base=2)

    plt.xticks(
        [32, 64, 128, 256],
        ["32", "64", "128", "256"]
    )

    plt.xlabel("Batch Size")
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
            "Best Accuracy vs Batch Size",
            "Accuracy (%)",
            "best_accuracy.png"
        ),
        (
            "final_val_accuracy",
            "Final Accuracy vs Batch Size",
            "Accuracy (%)",
            "final_accuracy.png"
        ),
        (
            "final_val_loss",
            "Validation Loss vs Batch Size",
            "Cross Entropy Loss",
            "validation_loss.png"
        ),
        (
            "total_train_time_sec",
            "Training Time vs Batch Size",
            "Time (seconds)",
            "training_time.png"
        ),
        (
            "peak_gpu_memory_mb",
            "GPU Memory vs Batch Size",
            "Allocated Memory (MiB)",
            "gpu_memory.png"
        ),
        (
            "images_per_second",
            "Training Throughput vs Batch Size",
            "Images/sec",
            "throughput.png"
        ),
        (
            "updates_to_target",
            "Optimizer Updates to Reach 85% Accuracy",
            "Number of Updates",
            "updates_to_target.png"
        ),
        (
            "convergence_time_sec",
            "Convergence Time vs Batch Size",
            "Time (seconds)",
            "convergence_time.png"
        )
    ]

    for metric in metrics:
        plot_metric(*metric)

    print("Batch size figures generated successfully.")