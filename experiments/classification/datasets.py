import os

import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision import transforms as T


def random90(x):
    return torch.rot90(x, int(torch.randint(0, 4, (1,)).item()), [1, 2])


def getDataset(args):
    # Define transformations
    if "cifar" in args.dataset:
        # Small size images.
        tr_train = T.Compose(
            [
                T.Resize(args.resolution),
                T.RandomCrop(args.resolution, padding=args.padding),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                random90 if args.rot else lambda x: x,
                T.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )
        tr_test = T.Compose(
            [
                T.Resize(args.resolution),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )
    else:
        # ImageNet-style preprocessing.
        tr_train = T.Compose(
            [
                T.RandomResizedCrop(args.resolution),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        tr_test = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(args.resolution),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    # Load dataset
    if args.dataset == "cifar10":
        x_train = datasets.CIFAR10(
            os.environ["DATA_DIR"], train=True, transform=tr_train, download=True
        )
        x_test = datasets.CIFAR10(
            os.environ["DATA_DIR"], train=False, transform=tr_test, download=True
        )
        args.classes = x_train.classes
    elif args.dataset == "cifar100":
        x_train = datasets.CIFAR100(
            os.environ["DATA_DIR"], train=True, transform=tr_train, download=True
        )
        x_test = datasets.CIFAR100(
            os.environ["DATA_DIR"], train=False, transform=tr_test, download=True
        )
        args.classes = x_train.classes
    elif args.dataset == "flowers102":
        x_train = datasets.Flowers102(
            os.environ["DATA_DIR"], split="train", transform=tr_train, download=True
        )
        x_val = datasets.Flowers102(
            os.environ["DATA_DIR"], split="val", transform=tr_train, download=True
        )
        x_train = torch.utils.data.ConcatDataset([x_train, x_val])  # type: ignore
        x_test = datasets.Flowers102(
            os.environ["DATA_DIR"], split="test", transform=tr_test, download=True
        )
        args.classes = torch.arange(102)
    else:
        raise AssertionError("Invalid value for args.dataset: ", args.dataset)

    # Define training subset.
    num_train = len(x_train)
    split = int(args.split * num_train)
    train_idx = torch.randperm(num_train)[:split].numpy()
    train_sampler = SubsetRandomSampler(train_idx)

    # Dataloaders.
    trainloader = DataLoader(
        x_train,
        batch_size=args.bs,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    testloader = DataLoader(
        x_test, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True
    )

    return trainloader, testloader
