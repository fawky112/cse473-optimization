import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data import ROOT, get_dataloaders
from src.model import LeakyGELUCNN
from src.optimizers import get_optimizer, OPTIMIZER_NAMES

from src.utils import (
    set_seed,
    get_device,
    synchronize_device,
    reset_gpu_memory,
    get_peak_gpu_memory
)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    batch_losses = []

    parameters = [
        (name, p)
        for name, p in model.named_parameters()
        if p.requires_grad
    ]

    gradient_norms = {
        name: 0.0
        for name, p in parameters
    }

    gradient_rms = {
        name: 0.0
        for name, p in parameters
    }

    global_gradient_norms = []

    synchronize_device(device)
    start_time = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        norms = torch.stack([
            p.grad.detach().norm()
            for name, p in parameters
        ])

        batch_correct = (
            outputs.argmax(dim=1) == labels
        ).sum()

        statistics = torch.cat([
            loss.detach().reshape(1),
            batch_correct.reshape(1),
            norms
        ]).tolist()

        loss_value = statistics[0]
        batch_correct_value = statistics[1]
        norm_values = statistics[2:]

        if not all(
            math.isfinite(value)
            for value in statistics
        ):

            raise FloatingPointError(
                f"Non-finite loss or gradient "
                f"at batch {batch_idx}"
            )

        optimizer.step()

        batch_size = labels.size(0)

        running_loss += loss_value * batch_size

        correct += batch_correct_value

        total += batch_size

        batch_losses.append(loss_value)

        for (name, p), norm in zip(
            parameters,
            norm_values
        ):

            gradient_norms[name] += norm

            gradient_rms[name] += (
                norm / math.sqrt(p.numel())
            )

        global_gradient_norms.append(
            math.sqrt(
                sum(norm ** 2 for norm in norm_values)
            )
        )

        if (batch_idx + 1) % 100 == 0:

            print(
                f"Batch {batch_idx + 1:>3}/{len(loader)} "
                f"| Loss: {loss_value:.4f} "
                f"| Running Acc: {100 * correct / total:.2f}%"
            )

    synchronize_device(device)

    epoch_time = time.perf_counter() - start_time

    num_batches = len(batch_losses)

    metrics = {
        "train_loss": running_loss / total,
        "train_accuracy": 100 * correct / total,
        "train_time_sec": epoch_time,
        "batch_loss_std": float(np.std(batch_losses)),
        "max_batch_loss": max(batch_losses),
        "global_gradient_norm": float(
            np.mean(global_gradient_norms)
        ),
        "max_gradient_norm": max(
            global_gradient_norms
        )
    }

    for name, p in parameters:

        metrics[f"grad_norm_{name}"] = (
            gradient_norms[name] / num_batches
        )

        metrics[f"grad_rms_{name}"] = (
            gradient_rms[name] / num_batches
        )

    return metrics


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    synchronize_device(device)
    start_time = time.perf_counter()

    for images, labels in loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):

            raise FloatingPointError(
                "Non-finite validation loss"
            )

        running_loss += loss_value * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    synchronize_device(device)

    validation_time = (
        time.perf_counter() - start_time
    )

    return {
        "val_loss": running_loss / total,
        "val_accuracy": 100 * correct / total,
        "val_time_sec": validation_time
    }


def train(
    optimizer_name="sgd",
    lr=0.01,
    momentum=0.9,
    alpha=0.99,
    betas=(0.9, 0.999),
    batch_size=128,
    epochs=20,
    seed=42,
    num_workers=4,
    target_accuracy=85.0,
    run_name=None
):

    set_seed(seed)

    device = get_device()

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed
    )

    model = LeakyGELUCNN().to(device)

    optimizer = get_optimizer(
        name=optimizer_name,
        model=model,
        lr=lr,
        momentum=momentum,
        alpha=alpha,
        betas=betas
    )

    criterion = nn.CrossEntropyLoss()

    if run_name is None:

        run_name = (
            f"{optimizer_name}"
            f"_lr{lr}"
            f"_m{momentum}"
            f"_a{alpha}"
            f"_b1{betas[0]}"
            f"_b2{betas[1]}"
            f"_bs{batch_size}"
            f"_ep{epochs}"
            f"_seed{seed}"
        )

    history_dir = (
        ROOT / "results" / "histories" / run_name
    )

    checkpoint_dir = (
        ROOT / "checkpoints" / run_name
    )

    history_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "optimizer": optimizer_name,
        "architecture": "LeakyGELUCNN",
        "dataset": "Fashion-MNIST",
        "lr": lr,
        "momentum": momentum,
        "alpha": alpha,
        "betas": list(betas),
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "num_workers": num_workers,
        "target_accuracy": target_accuracy,
        "precision": "FP32",
        "device": str(device),
        "pytorch_version": torch.__version__,
        "parameters": sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )
    }

    if device.type == "cuda":

        config["gpu"] = torch.cuda.get_device_name(0)

    with open(history_dir / "config.json", "w") as f:

        json.dump(config, f, indent=4)

    history = []

    best_accuracy = -1.0
    best_epoch = None

    total_train_time = 0.0
    total_validation_time = 0.0

    peak_gpu_memory = 0.0

    convergence_epoch = None
    convergence_time = None

    status = "completed"
    failure_reason = None

    print("\n" + "=" * 65)
    print("Training:", run_name)
    print("Device:", device)
    print("Optimizer:", optimizer_name.upper())
    print("Learning rate:", lr)
    print("Batch size:", batch_size)
    print("Epochs:", epochs)
    print("=" * 65)

    for epoch in range(1, epochs + 1):

        print(f"\nEpoch {epoch}/{epochs}")

        reset_gpu_memory(device)

        try:

            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )

            val_metrics = validate(
                model,
                val_loader,
                criterion,
                device
            )

        except FloatingPointError as error:

            status = "diverged"
            failure_reason = str(error)

            print(f"\nTraining stopped: {error}")

            break

        gpu_memory = get_peak_gpu_memory(device)

        if gpu_memory is not None:

            peak_gpu_memory = max(
                peak_gpu_memory,
                gpu_memory
            )

        total_train_time += (
            train_metrics["train_time_sec"]
        )

        total_validation_time += (
            val_metrics["val_time_sec"]
        )

        metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "peak_gpu_memory_mb": gpu_memory,
            "cumulative_train_time_sec": total_train_time
        }

        history.append(metrics)

        pd.DataFrame(history).to_csv(
            history_dir / "history.csv",
            index=False
        )

        val_accuracy = metrics["val_accuracy"]

        if (
            convergence_epoch is None
            and val_accuracy >= target_accuracy
        ):

            convergence_epoch = epoch

            convergence_time = (
                total_train_time
                + total_validation_time
            )

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "config": config
                },
                checkpoint_dir / "best_model.pt"
            )

        print("-" * 65)

        print(
            f"Train Loss: {metrics['train_loss']:.4f} "
            f"| Train Acc: {metrics['train_accuracy']:.2f}%"
        )

        print(
            f"Val Loss:   {metrics['val_loss']:.4f} "
            f"| Val Acc:   {metrics['val_accuracy']:.2f}%"
        )

        print(
            f"Gradient Norm: "
            f"{metrics['global_gradient_norm']:.6f}"
        )

        print(
            f"Training Time: "
            f"{metrics['train_time_sec']:.2f}s"
        )

        if gpu_memory is not None:

            print(
                f"Peak GPU Memory: {gpu_memory:.2f} MiB"
            )

        print(
            f"Best Validation Accuracy: "
            f"{best_accuracy:.2f}%"
        )

    summary = {
        **config,
        "status": status,
        "failure_reason": failure_reason,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": (
            best_accuracy if history else None
        ),
        "final_train_loss": (
            history[-1]["train_loss"] if history else None
        ),
        "final_val_loss": (
            history[-1]["val_loss"] if history else None
        ),
        "final_val_accuracy": (
            history[-1]["val_accuracy"] if history else None
        ),
        "total_train_time_sec": total_train_time,
        "total_validation_time_sec": total_validation_time,
        "peak_gpu_memory_mb": (
            peak_gpu_memory
            if device.type == "cuda" and history
            else None
        ),
        "convergence_epoch": convergence_epoch,
        "convergence_time_sec": convergence_time
    }

    with open(history_dir / "summary.json", "w") as f:

        json.dump(summary, f, indent=4)

    print("\n" + "=" * 65)
    print("Training status:", status)
    print("Best epoch:", best_epoch)
    print("Best validation accuracy:", summary["best_val_accuracy"])
    print("Total training time:", total_train_time)
    print("Convergence epoch:", convergence_epoch)
    print("Results saved to:", history_dir)
    print("=" * 65)

    return summary


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZER_NAMES,
        default="sgd"
    )

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.99)

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=85.0
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None
    )

    args = parser.parse_args()

    train(
        optimizer_name=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        alpha=args.alpha,
        betas=(args.beta1, args.beta2),
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        num_workers=args.workers,
        target_accuracy=args.target_accuracy,
        run_name=args.run_name
    )