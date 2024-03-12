import os

import torch
import numpy as np
from torchvision import datasets, transforms
from sampling import mnist_extr_noniid, cifar_extr_noniid, cifar100_extr_noniid, tiny_extr_noniid
from dataset import DatasetSplit, tiny


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def train_val_test_image(dataset, idxs, dataset_test, idxs_test):
    np.random.shuffle(idxs)
    idxs_train = idxs[:int(0.9 * len(idxs))]
    idxs_val = idxs[int(0.9 * len(idxs)):]
    idxs_test = idxs_test

    train_set = DatasetSplit(dataset, idxs_train)
    val_set = DatasetSplit(dataset, idxs_val)
    test_set = DatasetSplit(dataset_test, idxs_test)
    return idxs_val, train_set, val_set, test_set


def get_dataset_mnist_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = './data/mnist/'
    os.makedirs(data_dir, exist_ok=True)
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test


def get_dataset_cifar10_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = './data/cifar/'
    os.makedirs(data_dir, exist_ok=True)
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test


def get_dataset_cifar100_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = './data/cifar100/'
    os.makedirs(data_dir, exist_ok=True)
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                      transform=apply_transform)

    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                     transform=apply_transform)
    # Chose euqal splits for every user
    user_groups_train, user_groups_test = cifar100_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test


def get_dataset_tiny_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = './data/tiny_imagenet'
    os.makedirs(data_dir, exist_ok=True)
    CIFAR_MEAN = [0.5, 0.5, 0.5]
    CIFAR_STD = [0.5, 0.5, 0.5]
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_dataset = tiny(data_dir, train=True, transform=transform_train, download=True)
    test_dataset = tiny(data_dir, train=False, transform=transform_test, download=True)
    # Chose euqal splits for every user
    user_groups_train, user_groups_test = tiny_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test