import json
import subprocess
import sys

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

HISTORY_DIR = ROOT / "results" / "histories"

TABLE_DIR = ROOT / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


def run_momentum_experiment():

    with open(ROOT / "configs" / "baseline.json") as f:

        config = json.load(f)

    with open(ROOT / "configs" / "momentum.json") as f:

        momentum_config = json.load(f)

    summaries = []

    optimizers = momentum_config["optimizers"]

    momentum_values = momentum_config["momentum_values"]

    total = len(optimizers) * len(momentum_values)

    experiment = 0

    for optimizer in optimizers:

        for momentum in momentum_values:

            experiment += 1

            if momentum == config["momentum"]:

                run_name = (
                    f"baseline_{optimizer}"
                    f"_lr{config['lr']}"
                    f"_bs{config['batch_size']}"
                    f"_ep{config['epochs']}"
                    f"_seed{config['seed']}"
                )

            else:

                run_name = (
                    f"momentum_sweep_{optimizer}"
                    f"_m{momentum}"
                    f"_lr{config['lr']}"
                    f"_bs{config['batch_size']}"
                    f"_ep{config['epochs']}"
                    f"_seed{config['seed']}"
                )

            run_dir = HISTORY_DIR / run_name

            summary_path = run_dir / "summary.json"

            print("\n" + "=" * 75)

            print(f"Experiment {experiment}/{total}")
            print(f"Optimizer: {optimizer.upper()}")
            print(f"Momentum: {momentum}")

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
                    "--momentum", str(momentum),

                    "--alpha", str(config["alpha"]),
                    "--beta1", str(config["beta1"]),
                    "--beta2", str(config["beta2"]),

                    "--batch-size",
                    str(config["batch_size"]),

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
                TABLE_DIR / "momentum_results.csv",
                index=False
            )

    results = pd.DataFrame(summaries)

    columns = [
        "optimizer",
        "momentum",
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
        TABLE_DIR / "momentum_comparison.csv",
        index=False
    )

    print("\nMOMENTUM COMPARISON\n")

    print(
        comparison.to_string(
            index=False,
            na_rep="N/A"
        )
    )


if __name__ == "__main__":

    run_momentum_experiment()