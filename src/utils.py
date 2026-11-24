import random
import numpy as np
import torch


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def synchronize_device(device):

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_gpu_memory(device):

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_gpu_memory(device):

    if device.type == "cuda":

        return (
            torch.cuda.max_memory_allocated(device)
            / 1024**2
        )

    return None