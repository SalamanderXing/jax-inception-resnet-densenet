import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils import data
import nargsort

print(nargsort)
def get_data(*,train_batch_size: int, test_batch_size: int, dataset_path: str):

    train_dataset = CIFAR10(root=dataset_path, train=True, download=True)
    DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)
    # Transformations applied on each image => bring them into a numpy array
    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255.0 - DATA_MEANS) / DATA_STD
        return img

    # We need to stack the batch elements
    def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)

    test_transform = image_to_numpy
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            image_to_numpy,
        ]
    )
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(
        root=dataset_path, train=True, transform=train_transform, download=True
    )
    val_dataset = CIFAR10(
        root=dataset_path, train=True, transform=test_transform, download=True
    )
    train_set, _ = torch.utils.data.random_split(
        train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42)
    )
    _, val_set = torch.utils.data.random_split(
        val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42)
    )

    # Loading the test set
    test_set = CIFAR10(
        root=dataset_path, train=False, transform=test_transform, download=True
    )

    # We define a set of data loaders that we can use for training and validation
    train_loader = data.DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=numpy_collate,
        num_workers=8,
        persistent_workers=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
        num_workers=4,
        persistent_workers=True,
    )
    return train_loader, val_loader, test_loader
