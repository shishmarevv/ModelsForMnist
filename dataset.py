import numpy as np

from torch import Generator
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

def get_mnist_dataset():
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True, 
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset

def get_loaders(batch_size=32, val_ratio=0.2, seed=42):

    whole_dataset, test_dataset = get_mnist_dataset()

    train_dataset, val_dataset = random_split(whole_dataset, [1- val_ratio, val_ratio], generator=Generator().manual_seed(seed))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader


def get_cross_validation_loaders(batch_size=32, folds=5, seed=42):
    whole_dataset, test_dataset = get_mnist_dataset()

    idxs = list(range(len(whole_dataset)))

    np.random.seed(seed)

    np.random.shuffle(idxs)

    fold_size = len(whole_dataset) // folds
    fold_indices = [idxs[i:i + fold_size] for i in range(0, len(idxs), fold_size)]


    for i in range(folds):
        val_idx = fold_indices[i]
        train_idx = np.concatenate([fold_indices[j] for j in range(folds) if j != i])

        train_dataset = Subset(whole_dataset, train_idx)
        val_dataset = Subset(whole_dataset, val_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        yield train_loader, val_loader, test_loader
    