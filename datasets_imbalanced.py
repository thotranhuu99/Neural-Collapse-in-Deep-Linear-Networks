import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, EMNIST
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Sequence

class CustomSubsetRandomSampler(SubsetRandomSampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, indices: Sequence[int], generator=None) -> None:
        super().__init__(indices, generator)
    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

def make_dataset(dataset_name, data_dir, num_data, batch_size=128, SOTA=False):
    if dataset_name == 'cifar10':
        print('Dataset: CIFAR10.')
        trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]))

        testset = CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]))
        num_classes = 10
    elif dataset_name == 'mnist':
        print('Dataset: MNIST.')
        trainset = MNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))

        testset = MNIST(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))
        num_classes = 10
    elif dataset_name == 'emnist':
        print('Dataset: EMNIST.')
        trainset = EMNIST(root=data_dir, split="letters", train=True, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1722,), (0.3309,))
            ]))

        testset = EMNIST(root=data_dir, split="letters", train=False, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1722,), (0.3309,))
            ]))
        num_classes = 26
    else:
        raise ValueError

    total_sample_size = sum(num_data)
    cnt_dict = dict()
    total_cnt = 0
    indices = []
    for i in range(len(trainset)):

        if total_cnt == total_sample_size:
            break

        if dataset_name == "emnist":
            label = trainset[i][1] - 1
        else:
            label = trainset[i][1]
        if label not in cnt_dict:
            cnt_dict[label] = 1
            total_cnt += 1
            indices.append(i)
        else:
            if cnt_dict[label] == num_data[label]:
                continue
            else:
                cnt_dict[label] += 1
                total_cnt += 1
                indices.append(i)

    train_indices = torch.tensor(indices)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, sampler=CustomSubsetRandomSampler(train_indices), num_workers=1,
        shuffle=False)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    return trainloader, testloader, num_classes


