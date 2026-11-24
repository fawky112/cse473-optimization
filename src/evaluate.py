import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from src.data import get_datasets, CLASSES
from src.model import LeakyGELUCNN
from src.utils import get_device, synchronize_device


ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "results" / "tables"
CHECKPOINT_DIR = ROOT / "checkpoints"

FIGURE_DIR = (
    ROOT / "results" / "figures" / "evaluation"
)

REPORT_DIR = (
    ROOT / "results" / "classification_reports"
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@torch.inference_mode()
def evaluate_model(model, loader, device):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    total_samples = 0

    synchronize_device(device)

    start_time = time.perf_counter()

    for images, labels in loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)

        loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)

        total_samples += labels.size(0)

        all_labels.extend(labels.cpu().tolist())

        all_predictions.extend(
            predictions.cpu().tolist()
        )

    synchronize_device(device)

    elapsed_time = time.perf_counter() - start_time

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    metrics = {
        "test_loss": total_loss / total_samples,

        "test_accuracy": 100 * accuracy_score(
            y_true, y_pred
        ),

        "macro_precision": precision_score(
            y_true, y_pred,
            average="macro",
            labels=list(range(10)),
            zero_division=0
        ),

        "macro_recall": recall_score(
            y_true, y_pred,
            average="macro",
            labels=list(range(10)),
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            labels=list(range(10)),
            zero_division=0
        ),

        "evaluation_time_sec": elapsed_time,

        "evaluation_time_ms_per_image": (
            1000 * elapsed_time / total_samples
        ),

        "images_per_second": (
            total_samples / elapsed_time
        )
    }

    return metrics, y_true, y_pred


def save_classification_results(
    y_true,
    y_pred,
    optimizer,
    seed
):

    run_name = f"{optimizer}_seed{seed}"

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(10)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / f"{run_name}_report.csv"
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(10))
    )

    pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES
    ).to_csv(
        REPORT_DIR / f"{run_name}_confusion.csv"
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASSES
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=True
    )

    ax.set_title(
        f"{optimizer.upper()} - Seed {seed}"
        "\nFashion-MNIST Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{run_name}_confusion.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def run_evaluation():

    device = get_device()

    _, _, test_dataset = get_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True
    )

    runs = pd.read_csv(
        TABLE_DIR / "stability_results.csv"
    )

    runs = runs[
        (runs["status"] == "completed")
        & (runs["completed_epochs"] == runs["epochs"])
    ]

    summaries = []

    for index, (_, run) in enumerate(
        runs.iterrows(), start=1
    ):

        optimizer = run["optimizer"]
        seed = int(run["seed"])
        run_name = run["run_name"]

        checkpoint_path = (
            CHECKPOINT_DIR
            / run_name
            / "best_model.pt"
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                checkpoint_path
            )

        model = LeakyGELUCNN().to(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        del checkpoint

        metrics, y_true, y_pred = evaluate_model(
            model,
            test_loader,
            device
        )

        summary = {
            "optimizer": optimizer,
            "seed": seed,
            "run_name": run_name,
            "lr": run["lr"],
            "training_batch_size": run["batch_size"],
            "best_epoch": run["best_epoch"],
            "best_val_accuracy": run["best_val_accuracy"],
            **metrics
        }

        summaries.append(summary)

        save_classification_results(
            y_true,
            y_pred,
            optimizer,
            seed
        )

        print(
            f"[{index}/{len(runs)}] "
            f"{optimizer.upper():<10} | "
            f"Seed: {seed:<5} | "
            f"Test Acc: {metrics['test_accuracy']:.2f}% | "
            f"F1: {metrics['macro_f1']:.4f} | "
            f"Loss: {metrics['test_loss']:.4f}"
        )

        pd.DataFrame(summaries).to_csv(
            TABLE_DIR / "test_results.csv",
            index=False
        )

        del model

    results = pd.DataFrame(summaries)

    statistics = results.groupby("optimizer").agg(

        evaluated_runs=("seed", "count"),

        mean_test_accuracy=("test_accuracy", "mean"),

        std_test_accuracy=("test_accuracy", "std"),

        mean_test_loss=("test_loss", "mean"),

        mean_precision=("macro_precision", "mean"),

        mean_recall=("macro_recall", "mean"),

        mean_f1=("macro_f1", "mean"),

        std_f1=("macro_f1", "std"),

        mean_evaluation_time=(
            "evaluation_time_sec", "mean"
        ),

        mean_time_per_image=(
            "evaluation_time_ms_per_image", "mean"
        )

    ).reset_index()

    statistics = statistics.sort_values(
        "mean_test_accuracy",
        ascending=False
    )

    statistics.to_csv(
        TABLE_DIR / "test_statistics.csv",
        index=False
    )

    print("\nFINAL TEST EVALUATION\n")

    print(
        statistics.round(4).to_string(
            index=False
        )
    )


if __name__ == "__main__":

    run_evaluation()