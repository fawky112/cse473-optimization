import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = ROOT / "results" / "figures" / "adaptive"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(
    ROOT / "results" / "tables" / "adaptive_comparison.csv"
)


def plot_parameter(data, parameter, title, filename):

    data = data.sort_values(parameter)

    x = range(len(data))

    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        data["best_val_accuracy"],
        marker="o",
        linewidth=2,
        label="Best Validation Accuracy"
    )

    plt.plot(
        x,
        data["final_val_accuracy"],
        marker="s",
        linewidth=2,
        label="Final Validation Accuracy"
    )

    plt.xticks(x, data[parameter].astype(str))

    plt.xlabel(parameter)
    plt.ylabel("Accuracy (%)")
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

    rmsprop = results[
        results["optimizer"] == "rmsprop"
    ]

    adam = results[
        results["optimizer"] == "adam"
    ]

    plot_parameter(
        rmsprop,
        "alpha",
        "Effect of Alpha on RMSprop",
        "rmsprop_alpha.png"
    )

    plot_parameter(
        adam[adam["beta2"] == 0.999],
        "beta1",
        "Effect of Beta1 on Adam",
        "adam_beta1.png"
    )

    plot_parameter(
        adam[adam["beta1"] == 0.9],
        "beta2",
        "Effect of Beta2 on Adam",
        "adam_beta2.png"
    )

    print("Adaptive optimizer figures generated.")