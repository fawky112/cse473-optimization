import json
import subprocess
import sys

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

HISTORY_DIR = ROOT / "results" / "histories"
TABLE_DIR = ROOT / "results" / "tables"

SEEDS = [42, 123, 2026]


def run_stability_experiment():

    configurations = pd.read_csv(
        TABLE_DIR / "best_configurations.csv"
    )

    summaries = []

    total = len(configurations) * len(SEEDS)

    experiment = 0

    for _, row in configurations.iterrows():

        original_run = row["run_name"]

        with open(
            HISTORY_DIR / original_run / "config.json"
        ) as f:

            config = json.load(f)

        optimizer = config["optimizer"]

        for seed in SEEDS:

            experiment += 1

            if seed == config["seed"]:

                run_name = original_run

            else:

                base_name = original_run.rsplit(
                    "_seed", 1
                )[0]

                run_name = (
                    f"stability_{base_name}"
                    f"_seed{seed}"
                )

            run_dir = HISTORY_DIR / run_name

            summary_path = run_dir / "summary.json"

            print("\n" + "=" * 75)

            print(f"Experiment {experiment}/{total}")
            print(f"Optimizer: {optimizer.upper()}")
            print(f"Learning rate: {config['lr']}")
            print(f"Batch size: {config['batch_size']}")
            print(f"Seed: {seed}")

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

                    "--lr", str(config["lr"]),

                    "--momentum",
                    str(config["momentum"]),

                    "--alpha",
                    str(config["alpha"]),

                    "--beta1",
                    str(config["betas"][0]),

                    "--beta2",
                    str(config["betas"][1]),

                    "--batch-size",
                    str(config["batch_size"]),

                    "--epochs",
                    str(config["epochs"]),

                    "--seed", str(seed),

                    "--workers",
                    str(config["num_workers"]),

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

            expected = {
                "seed": seed,
                "optimizer": optimizer,
                "lr": config["lr"],
                "batch_size": config["batch_size"],
                "epochs": config["epochs"],
                "momentum": config["momentum"],
                "alpha": config["alpha"],
                "betas": config["betas"]
            }

            for key, value in expected.items():

                if summary.get(key) != value:

                    raise RuntimeError(
                        f"Configuration mismatch: "
                        f"{run_name}, {key}"
                    )

            summary["run_name"] = run_name

            summaries.append(summary)

            pd.DataFrame(summaries).to_csv(
                TABLE_DIR / "stability_results.csv",
                index=False
            )

    results = pd.DataFrame(summaries)

    columns = [
        "optimizer",
        "seed",
        "lr",
        "batch_size",
        "status",
        "best_epoch",
        "best_val_accuracy",
        "final_val_accuracy",
        "final_val_loss",
        "total_train_time_sec",
        "peak_gpu_memory_mb",
        "convergence_epoch",
        "convergence_time_sec"
    ]

    results[columns].to_csv(
        TABLE_DIR / "stability_comparison.csv",
        index=False
    )

    successful = results[
        (results["status"] == "completed")
        & (results["completed_epochs"] == results["epochs"])
    ]

    statistics = successful.groupby("optimizer").agg(

        successful_runs=("seed", "count"),

        mean_best_accuracy=(
            "best_val_accuracy", "mean"
        ),

        std_best_accuracy=(
            "best_val_accuracy", "std"
        ),

        min_best_accuracy=(
            "best_val_accuracy", "min"
        ),

        max_best_accuracy=(
            "best_val_accuracy", "max"
        ),

        mean_final_accuracy=(
            "final_val_accuracy", "mean"
        ),

        std_final_accuracy=(
            "final_val_accuracy", "std"
        ),

        mean_final_loss=(
            "final_val_loss", "mean"
        ),

        mean_training_time=(
            "total_train_time_sec", "mean"
        ),

        std_training_time=(
            "total_train_time_sec", "std"
        ),

        mean_gpu_memory=(
            "peak_gpu_memory_mb", "mean"
        ),

        mean_convergence_time=(
            "convergence_time_sec", "mean"
        ),

        reached_target_runs=(
            "convergence_epoch", "count"
        )

    )

    statistics = statistics.reindex(
        results["optimizer"].unique()
    )

    statistics["successful_runs"] = (
        statistics["successful_runs"]
        .fillna(0)
        .astype(int)
    )

    statistics["total_runs"] = (
        results.groupby("optimizer")["seed"].count()
    )

    statistics["failed_runs"] = (
        statistics["total_runs"]
        - statistics["successful_runs"]
    )

    new_seeds = successful[
        successful["seed"] != 42
    ]

    statistics["new_seeds_mean_accuracy"] = (
        new_seeds.groupby("optimizer")[
            "best_val_accuracy"
        ].mean()
    )

    statistics = statistics.reset_index()

    statistics = statistics.sort_values(
        "mean_best_accuracy",
        ascending=False
    )

    statistics.to_csv(
        TABLE_DIR / "stability_statistics.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("MULTIPLE-SEED STABILITY ANALYSIS")
    print("=" * 100)

    print(
        statistics.round(4).to_string(
            index=False,
            na_rep="N/A"
        )
    )


if __name__ == "__main__":

    run_stability_experiment()