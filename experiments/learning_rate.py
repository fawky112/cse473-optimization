import json
import subprocess
import sys

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

HISTORY_DIR = ROOT / "results" / "histories"

TABLE_DIR = ROOT / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


def run_learning_rate_experiment():

    with open(ROOT / "configs" / "baseline.json") as f:

        config = json.load(f)

    with open(ROOT / "configs" / "learning_rate.json") as f:

        lr_config = json.load(f)

    learning_rates = lr_config["learning_rates"]

    optimizers = config["optimizers"]

    summaries = []

    total = len(optimizers) * len(learning_rates)

    experiment = 0

    for optimizer in optimizers:

        for lr in learning_rates:

            experiment += 1

            prefix = (
                "baseline"
                if lr == config["lr"]
                else "lr_sweep"
            )

            run_name = (
                f"{prefix}_{optimizer}"
                f"_lr{lr}"
                f"_bs{config['batch_size']}"
                f"_ep{config['epochs']}"
                f"_seed{config['seed']}"
            )

            run_dir = HISTORY_DIR / run_name

            summary_path = run_dir / "summary.json"

            print("\n" + "=" * 75)

            print(f"Experiment {experiment}/{total}")

            print(f"Optimizer: {optimizer.upper()}")

            print(f"Learning rate: {lr}")

            print("=" * 75)

            if summary_path.exists():

                print("Loading existing experiment.")

            else:

                if run_dir.exists():

                    raise RuntimeError(
                        f"Incomplete experiment: {run_dir}. "
                        "Back up this directory before retrying."
                    )

                command = [
                    sys.executable,
                    "-m",
                    "src.train",

                    "--optimizer", optimizer,
                    "--lr", str(lr),

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

            with open(summary_path) as f:

                summary = json.load(f)

            summary["run_name"] = run_name

            summaries.append(summary)

            pd.DataFrame(summaries).to_csv(
                TABLE_DIR / "learning_rate_results.csv",
                index=False
            )

    results = pd.DataFrame(summaries)

    columns = [
        "optimizer",
        "lr",
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

    comparison = results[columns]

    comparison.to_csv(
        TABLE_DIR / "learning_rate_comparison.csv",
        index=False
    )

    valid = results[
        (results["status"] == "completed")
        & results["best_val_accuracy"].notna()
    ]

    best = (
        valid.sort_values(
            ["best_val_accuracy", "final_val_accuracy"],
            ascending=False
        )
        .groupby("optimizer")
        .head(1)
    )

    best.to_csv(
        TABLE_DIR / "best_learning_rates.csv",
        index=False
    )

    print("\nLEARNING RATE COMPARISON\n")

    print(comparison.to_string(index=False))

    print("\nBEST LEARNING RATES\n")

    print(
        best[
            [
                "optimizer",
                "lr",
                "best_val_accuracy",
                "final_val_accuracy",
                "best_epoch"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":

    run_learning_rate_experiment()