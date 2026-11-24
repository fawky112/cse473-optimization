from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "results" / "tables"

HISTORY_DIR = ROOT / "results" / "histories"

EPOCH_BUDGETS = [5, 10, 15, 20]


def select_best_configurations():

    files = [
        "learning_rate_results.csv",
        "momentum_results.csv",
        "batch_size_results.csv",
        "adaptive_results.csv"
    ]

    results = pd.concat(
        [
            pd.read_csv(TABLE_DIR / filename)
            for filename in files
        ],
        ignore_index=True
    )

    results = results.drop_duplicates(
        subset=["run_name"]
    )

    results = results[
        (results["status"] == "completed")
        & (results["completed_epochs"] == 20)
        & results["best_val_accuracy"].notna()
    ]

    best = (
        results.sort_values(
            [
                "best_val_accuracy",
                "final_val_accuracy",
                "final_val_loss"
            ],
            ascending=[False, False, True]
        )
        .groupby("optimizer")
        .head(1)
    )

    best.to_csv(
        TABLE_DIR / "best_configurations.csv",
        index=False
    )

    return best


def run_epoch_analysis():

    configurations = select_best_configurations()

    summaries = []

    for _, config in configurations.iterrows():

        optimizer = config["optimizer"]

        run_name = config["run_name"]

        history = pd.read_csv(
            HISTORY_DIR / run_name / "history.csv"
        )

        history = history.sort_values("epoch")

        print("\n" + "=" * 65)

        print("Optimizer:", optimizer.upper())
        print("Run:", run_name)

        print("=" * 65)

        for budget in EPOCH_BUDGETS:

            subset = history[
                history["epoch"] <= budget
            ]

            if len(subset) != budget:

                raise RuntimeError(
                    f"Incomplete history: {run_name}"
                )

            final = subset.iloc[-1]

            best = subset.loc[
                subset["val_accuracy"].idxmax()
            ]

            reached = subset[
                subset["val_accuracy"] >= 85.0
            ]

            if reached.empty:

                convergence_epoch = None

                convergence_time = None

            else:

                convergence_epoch = int(
                    reached.iloc[0]["epoch"]
                )

                elapsed = subset[
                    subset["epoch"] <= convergence_epoch
                ]

                convergence_time = (
                    elapsed["train_time_sec"].sum()
                    + elapsed["val_time_sec"].sum()
                )

            summary = {
                "optimizer": optimizer,
                "run_name": run_name,
                "lr": config["lr"],
                "batch_size": config["batch_size"],
                "epoch_budget": budget,

                "best_epoch": int(best["epoch"]),

                "best_val_accuracy":
                    best["val_accuracy"],

                "final_train_accuracy":
                    final["train_accuracy"],

                "final_val_accuracy":
                    final["val_accuracy"],

                "final_train_loss":
                    final["train_loss"],

                "final_val_loss":
                    final["val_loss"],

                "accuracy_gap":
                    final["train_accuracy"]
                    - final["val_accuracy"],

                "loss_gap":
                    final["val_loss"]
                    - final["train_loss"],

                "total_train_time_sec":
                    subset["train_time_sec"].sum(),

                "peak_gpu_memory_mb":
                    subset["peak_gpu_memory_mb"].max(),

                "convergence_epoch":
                    convergence_epoch,

                "convergence_time_sec":
                    convergence_time
            }

            summaries.append(summary)

            print(
                f"Epochs: {budget:>2} | "
                f"Best Acc: "
                f"{best['val_accuracy']:.2f}% | "
                f"Final Acc: "
                f"{final['val_accuracy']:.2f}% | "
                f"Val Loss: "
                f"{final['val_loss']:.4f}"
            )

    results = pd.DataFrame(summaries)

    results.to_csv(
        TABLE_DIR / "epoch_comparison.csv",
        index=False
    )

    print("\nEPOCH COMPARISON\n")

    columns = [
        "optimizer",
        "epoch_budget",
        "best_epoch",
        "best_val_accuracy",
        "final_val_accuracy",
        "final_train_loss",
        "final_val_loss",
        "accuracy_gap",
        "total_train_time_sec"
    ]

    print(
        results[columns].to_string(
            index=False,
            na_rep="N/A"
        )
    )


if __name__ == "__main__":

    run_epoch_analysis()