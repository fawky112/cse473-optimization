import json
import math
import subprocess
import sys

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

HISTORY_DIR = ROOT / "results" / "histories"

TABLE_DIR = ROOT / "results" / "tables"


def run_batch_size_experiment():

    with open(ROOT / "configs" / "baseline.json") as f:

        config = json.load(f)

    with open(ROOT / "configs" / "batch_size.json") as f:

        batch_config = json.load(f)

    best_lrs = pd.read_csv(
        TABLE_DIR / "best_learning_rates.csv"
    )

    batch_sizes = batch_config["batch_sizes"]

    summaries = []

    total = len(best_lrs) * len(batch_sizes)

    experiment = 0

    for _, row in best_lrs.iterrows():

        optimizer = row["optimizer"]

        lr = float(row["lr"])

        for batch_size in batch_sizes:

            experiment += 1

            if batch_size == config["batch_size"]:

                run_name = row["run_name"]

            else:

                run_name = (
                    f"batch_sweep_{optimizer}"
                    f"_lr{lr}"
                    f"_bs{batch_size}"
                    f"_ep{config['epochs']}"
                    f"_seed{config['seed']}"
                )

            run_dir = HISTORY_DIR / run_name

            summary_path = run_dir / "summary.json"

            print("\n" + "=" * 75)

            print(f"Experiment {experiment}/{total}")

            print(f"Optimizer: {optimizer.upper()}")
            print(f"Learning rate: {lr}")
            print(f"Batch size: {batch_size}")

            print("=" * 75)

            if summary_path.exists():

                print("Loading existing experiment.")

            else:

                if run_dir.exists():

                    raise RuntimeError(
                        f"Incomplete experiment: {run_dir}"
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

                    "--batch-size", str(batch_size),
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

            summary["updates_per_epoch"] = math.ceil(
                50000 / batch_size
            )

            summary["completed_epoch_updates"] = (
                summary["updates_per_epoch"]
                * summary["completed_epochs"]
            )

            if summary["convergence_epoch"] is not None:

                summary["updates_to_target"] = (
                    summary["updates_per_epoch"]
                    * summary["convergence_epoch"]
                )

            else:

                summary["updates_to_target"] = None

            if summary["status"] == "completed":

                summary["images_per_second"] = (
                    50000 * summary["completed_epochs"]
                    / summary["total_train_time_sec"]
                )

            else:

                summary["images_per_second"] = None

            summaries.append(summary)

            pd.DataFrame(summaries).to_csv(
                TABLE_DIR / "batch_size_results.csv",
                index=False
            )

    results = pd.DataFrame(summaries)

    columns = [
        "optimizer",
        "lr",
        "batch_size",
        "status",
        "completed_epochs",
        "best_epoch",
        "best_val_accuracy",
        "final_val_accuracy",
        "final_val_loss",
        "total_train_time_sec",
        "peak_gpu_memory_mb",
        "convergence_epoch",
        "convergence_time_sec",
        "updates_per_epoch",
        "completed_epoch_updates",
        "updates_to_target",
        "images_per_second"
    ]

    comparison = results[columns]

    comparison.to_csv(
        TABLE_DIR / "batch_size_comparison.csv",
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
        TABLE_DIR / "best_batch_sizes.csv",
        index=False
    )

    print("\nBATCH SIZE COMPARISON\n")

    print(
        comparison.to_string(
            index=False,
            na_rep="N/A"
        )
    )

    print("\nBEST BATCH SIZES\n")

    print(
        best[
            [
                "optimizer",
                "lr",
                "batch_size",
                "best_val_accuracy",
                "final_val_accuracy",
                "peak_gpu_memory_mb"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":

    run_batch_size_experiment()