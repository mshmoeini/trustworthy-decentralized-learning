"""Fashion-MNIST dataset utilities."""

from pathlib import Path

from torchvision import datasets, transforms

FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)


def load_fashion_mnist(
    data_dir: str | Path,
) -> tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """Download and return separate normalized train and test datasets."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ]
    )
    root = str(Path(data_dir))
    train_dataset = datasets.FashionMNIST(
        root=root,
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = datasets.FashionMNIST(
        root=root,
        train=False,
        transform=transform,
        download=True,
    )
    return train_dataset, test_dataset
