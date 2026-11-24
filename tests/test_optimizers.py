import torch
import torch.nn as nn

from src.model import LeakyGELUCNN
from src.optimizers import get_optimizer, OPTIMIZER_NAMES


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)

images = torch.randn(16, 1, 28, 28).to(device)
labels = torch.randint(0, 10, (16,)).to(device)

criterion = nn.CrossEntropyLoss()


for name in OPTIMIZER_NAMES:

    torch.manual_seed(42)

    model = LeakyGELUCNN().to(device)

    optimizer = get_optimizer(
        name,
        model,
        lr=0.01
    )

    optimizer.zero_grad(set_to_none=True)

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    with torch.no_grad():

        new_loss = criterion(
            model(images),
            labels
        )

    assert torch.isfinite(new_loss).item()

    print(
        f"{name.upper():<12} | "
        f"Initial Loss: {loss.item():.6f} | "
        f"Updated Loss: {new_loss.item():.6f}"
    )

print("\nAll optimizers passed.")