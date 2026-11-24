import torch.optim as optim


OPTIMIZER_NAMES = [
    "sgd",
    "momentum",
    "nesterov",
    "adagrad",
    "rmsprop",
    "adam"
]


def get_optimizer(
    name,
    model,
    lr=0.01,
    momentum=0.9,
    alpha=0.99,
    betas=(0.9, 0.999)
):

    name = name.lower()
    params = model.parameters()

    if name == "sgd":

        return optim.SGD(
            params,
            lr=lr
        )

    elif name == "momentum":

        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum
        )

    elif name == "nesterov":

        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=True
        )

    elif name == "adagrad":

        return optim.Adagrad(
            params,
            lr=lr,
            eps=1e-10
        )

    elif name == "rmsprop":

        return optim.RMSprop(
            params,
            lr=lr,
            alpha=alpha,
            eps=1e-8
        )

    elif name == "adam":

        return optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=1e-8
        )

    raise ValueError(f"Unknown optimizer: {name}")