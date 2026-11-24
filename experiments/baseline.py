import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "baseline.json"

HISTORY_DIR = ROOT / "results" / "histories"

TABLE_DIR = ROOT / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


def run_baseline():

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    summaries = []

    total_experiments = len(config["optimizers"])

    for index, optimizer in enumerate(config["optimizers"], 1):

        run_name = (
            f"baseline_{optimizer}"
            f"_lr{config['lr']}"
            f"_bs{config['batch_size']}"
            f"_ep{config['epochs']}"
            f"_seed{config['seed']}"
        )

        summary_path = (
            HISTORY_DIR / run_name / "summary.json"
        )

        print("\n" + "=" * 75)

        print(
            f"Experiment {index}/{total_experiments}"
        )

        print(f"Optimizer: {optimizer.upper()}")

        print("=" * 75)

        if summary_path.exists():

            print("Experiment already recorded. Loading results.")

        else:

            command = [
                sys.executable,
                "-m",
                "src.train",

                "--optimizer", optimizer,
                "--lr", str(config["lr"]),
                "--momentum", str(config["momentum"]),
                "--alpha", str(config["alpha"]),
                "--beta1", str(config["beta1"]),
                "--beta2", str(config["beta2"]),

                "--batch-size", str(config["batch_size"]),
                "--epochs", str(config["epochs"]),
                "--seed", str(config["seed"]),
                "--workers", str(config["workers"]),

                "--target-accuracy",
                str(config["target_accuracy"]),

                "--run-name", run_name
            ]

            subprocess.run(
                command,
                cwd=ROOT,
                check=True
            )

        with open(summary_path, "r") as f:

            summary = json.load(f)

        summaries.append(summary)

        pd.DataFrame(summaries).to_csv(
            TABLE_DIR / "baseline_results.csv",
            index=False
        )

    results = pd.DataFrame(summaries)

    selected_columns = [
        "optimizer",
        "status",
        "completed_epochs",
        "best_epoch",
        "best_val_accuracy",
        "final_val_accuracy",
        "final_val_loss",
        "total_train_time_sec",
        "peak_gpu_memory_mb",
        "convergence_epoch",
        "convergence_time_sec"
    ]

    comparison = results[selected_columns]

    comparison.to_csv(
        TABLE_DIR / "baseline_comparison.csv",
        index=False
    )

    print("\n" + "=" * 100)

    print("BASELINE OPTIMIZER COMPARISON")

    print("=" * 100)

    print(
        comparison.to_string(index=False)
    )

    print("\nResults saved to:")

    print(TABLE_DIR / "baseline_results.csv")
    print(TABLE_DIR / "baseline_comparison.csv")


if __name__ == "__main__":

    run_baseline()