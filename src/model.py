import torch
import torch.nn as nn


class LeakyGELUCNN(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.block1 = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(negative_slope=0.01),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(negative_slope=0.01),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 7 * 7, 128),

            nn.GELU(),

            nn.Linear(128, 64),

            nn.GELU(),

            nn.Linear(64, num_classes)
        )

        self.initialize_weights()


    def initialize_weights(self):

        for layer in self.modules():

            if isinstance(layer, nn.Conv2d):

                nn.init.kaiming_normal_(
                    layer.weight,
                    a=0.01,
                    mode="fan_in",
                    nonlinearity="leaky_relu"
                )

                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

            elif isinstance(layer, nn.Linear):

                nn.init.xavier_normal_(layer.weight)

                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.classifier(x)

        return x


    def predict_proba(self, x):

        logits = self.forward(x)

        return torch.softmax(logits, dim=1)


if __name__ == "__main__":

    torch.manual_seed(42)

    model = LeakyGELUCNN()

    print(model)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    x = torch.randn(128, 1, 28, 28)

    output = model(x)

    print("\nModel Summary")

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)

    print(
        "FP32 parameter memory:",
        total_parameters * 4 / 1024**2,
        "MiB"
    )