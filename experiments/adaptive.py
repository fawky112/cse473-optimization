import json
import subprocess
import sys

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

HISTORY_DIR = ROOT / "results" / "histories"

TABLE_DIR = ROOT / "results" / "tables"


def run_adaptive_experiment():

    with open(ROOT / "configs" / "baseline.json") as f:
        config = json.load(f)

    with open(ROOT / "configs" / "adaptive.json") as f:
        adaptive = json.load(f)

    lr = adaptive["lr"]

    batch_size = adaptive["batch_size"]

    default_alpha = config["alpha"]
    default_beta1 = config["beta1"]
    default_beta2 = config["beta2"]

    experiments = []

    for alpha in adaptive["alpha_values"]:

        experiments.append(
            ("rmsprop", alpha, default_beta1, default_beta2)
        )

    for beta1 in adaptive["beta1_values"]:

        experiments.append(
            ("adam", default_alpha, beta1, default_beta2)
        )

    for beta2 in adaptive["beta2_values"]:

        experiments.append(
            ("adam", default_alpha, default_beta1, beta2)
        )

    experiments = list(dict.fromkeys(experiments))

    summaries = []

    for index, experiment in enumerate(experiments, 1):

        optimizer, alpha, beta1, beta2 = experiment

        is_default = (
            alpha == default_alpha
            and beta1 == default_beta1
            and beta2 == default_beta2
        )

        if is_default:

            run_name = (
                f"lr_sweep_{optimizer}"
                f"_lr{lr}"
                f"_bs{batch_size}"
                f"_ep{config['epochs']}"
                f"_seed{config['seed']}"
            )

        else:

            run_name = (
                f"adaptive_{optimizer}"
                f"_lr{lr}"
                f"_a{alpha}"
                f"_b1{beta1}"
                f"_b2{beta2}"
                f"_bs{batch_size}"
                f"_ep{config['epochs']}"
                f"_seed{config['seed']}"
            )

        run_dir = HISTORY_DIR / run_name

        summary_path = run_dir / "summary.json"

        print("\n" + "=" * 75)

        print(f"Experiment {index}/{len(experiments)}")
        print(f"Optimizer: {optimizer.upper()}")
        print(f"Alpha: {alpha}")
        print(f"Beta1: {beta1}")
        print(f"Beta2: {beta2}")

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
                "--alpha", str(alpha),
                "--beta1", str(beta1),
                "--beta2", str(beta2),

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

        summary["beta1"] = beta1
        summary["beta2"] = beta2

        summaries.append(summary)

        pd.DataFrame(summaries).to_csv(
            TABLE_DIR / "adaptive_results.csv",
            index=False
        )

    results = pd.DataFrame(summaries)

    columns = [
        "optimizer",
        "lr",
        "alpha",
        "beta1",
        "beta2",
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
        TABLE_DIR / "adaptive_comparison.csv",
        index=False
    )

    print("\nADAPTIVE OPTIMIZER COMPARISON\n")

    print(
        comparison.to_string(
            index=False,
            na_rep="N/A"
        )
    )


if __name__ == "__main__":

    run_adaptive_experiment()