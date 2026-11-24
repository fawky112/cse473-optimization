from pathlib import Path

import torch
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "splits"

SEED = 42
BATCH_SIZE = 128

CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def prepare_dataset():

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    labels = train_dataset.targets.numpy()

    indices = np.arange(len(train_dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=10000,
        random_state=SEED,
        stratify=labels
    )

    np.save(SPLIT_DIR / "train_indices.npy", train_indices)
    np.save(SPLIT_DIR / "val_indices.npy", val_indices)

    train_pixels = (
        train_dataset.data[train_indices]
        .float() / 255.0
    )

    mean = train_pixels.mean().item()
    std = train_pixels.std(unbiased=False).item()

    np.savez(
        SPLIT_DIR / "normalization.npz",
        mean=mean,
        std=std
    )

    print("\nDataset Preparation Completed")

    print(f"Training images:   {len(train_indices)}")
    print(f"Validation images: {len(val_indices)}")
    print(f"Testing images:    {len(test_dataset)}")

    print(f"Training mean: {mean:.6f}")
    print(f"Training std:  {std:.6f}")

    print("\nClass Distribution")
    print(f"{'Class':<15}{'Train':>10}{'Val':>10}{'Test':>10}")

    test_labels = test_dataset.targets.numpy()

    for i, name in enumerate(CLASSES):

        train_count = np.sum(labels[train_indices] == i)
        val_count = np.sum(labels[val_indices] == i)
        test_count = np.sum(test_labels == i)

        print(
            f"{name:<15}"
            f"{train_count:>10}"
            f"{val_count:>10}"
            f"{test_count:>10}"
        )


def get_datasets():

    train_indices = np.load(
        SPLIT_DIR / "train_indices.npy"
    )

    val_indices = np.load(
        SPLIT_DIR / "val_indices.npy"
    )

    stats = np.load(
        SPLIT_DIR / "normalization.npz"
    )

    mean = float(stats["mean"])
    std = float(stats["std"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    full_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform
    )

    train_dataset = Subset(
        full_dataset,
        train_indices.tolist()
    )

    val_dataset = Subset(
        full_dataset,
        val_indices.tolist()
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=4,
    seed=SEED
):

    train_dataset, val_dataset, test_dataset = get_datasets()

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_args
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_args
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_args
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    prepare_dataset()

    train_loader, val_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))

    print("\nDataLoader Test")

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Testing batches:", len(test_loader))