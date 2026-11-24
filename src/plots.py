import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

from src.data import ROOT, DATA_DIR, SPLIT_DIR, CLASSES


FIGURE_DIR = ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_dataset_samples():

    dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=False
    )

    train_indices = np.load(SPLIT_DIR / "train_indices.npy")
    labels = dataset.targets.numpy()

    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for class_id, ax in enumerate(axes.flat):

        indices = train_indices[
            labels[train_indices] == class_id
        ]

        index = rng.choice(indices)

        image, label = dataset[int(index)]

        ax.imshow(image, cmap="gray")
        ax.set_title(CLASSES[label])
        ax.axis("off")

    fig.suptitle("Fashion-MNIST Dataset Samples", fontsize=16)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "dataset_samples.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_class_distribution():

    dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=False
    )

    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=False
    )

    labels = dataset.targets.numpy()

    train_indices = np.load(SPLIT_DIR / "train_indices.npy")
    val_indices = np.load(SPLIT_DIR / "val_indices.npy")

    train_counts = np.bincount(
        labels[train_indices], minlength=10
    )

    val_counts = np.bincount(
        labels[val_indices], minlength=10
    )

    test_counts = np.bincount(
        test_dataset.targets.numpy(), minlength=10
    )

    x = np.arange(len(CLASSES))
    width = 0.25

    plt.figure(figsize=(15, 6))

    plt.bar(x - width, train_counts, width, label="Training")
    plt.bar(x, val_counts, width, label="Validation")
    plt.bar(x + width, test_counts, width, label="Testing")

    plt.xticks(x, CLASSES, rotation=30)

    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.title("Fashion-MNIST Class Distribution")

    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "class_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_normalization():

    dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=False
    )

    stats = np.load(SPLIT_DIR / "normalization.npz")

    mean = float(stats["mean"])
    std = float(stats["std"])

    train_indices = np.load(SPLIT_DIR / "train_indices.npy")

    rng = np.random.default_rng(42)

    indices = rng.choice(
        train_indices, size=1000, replace=False
    )

    pixels = (
        dataset.data[indices].numpy().astype(np.float32) / 255.0
    )

    normalized = (pixels - mean) / std

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(pixels.flatten(), bins=100)
    axes[0].set_title("Before Normalization")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(normalized.flatten(), bins=100)
    axes[1].set_title("After Normalization")
    axes[1].set_xlabel("Normalized Pixel Intensity")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "normalization_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":

    plot_dataset_samples()
    plot_class_distribution()
    plot_normalization()

    print("Dataset visualization completed.")
    print(f"Figures saved to: {FIGURE_DIR}")