import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import datasets
import random
import transforms

import pickle
import os

known_class = -1
init_percent = -1

class CustomMNISTDataset_train(Dataset):

    def __init__(self, root="./data/mnist", train=True, download=True, transform=None, invalidList=None):

        self.mnist_dataset = datasets.MNIST(root, train=train, download=download, transform=transform)
        self.targets = self.mnist_dataset.targets

        if invalidList is not None:
            targets = np.array(self.mnist_dataset.targets)
            targets[targets >= known_class] = known_class
            self.mnist_dataset.targets = targets.tolist()

    def __getitem__(self, index):
        data_point, label = self.mnist_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(self.mnist_dataset)



class CustomMNISTDataset_test(Dataset):
    mnist_dataset = None
    targets = None

    @classmethod
    def load_dataset(cls, root="./data/mnist", train=False, download=True, transform=None):
        cls.mnist_dataset = datasets.MNIST(root, train=train, download=download, transform=transform)
        cls.targets = cls.mnist_dataset.targets

    def __init__(self):
        if CustomMNISTDataset_test.mnist_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        data_point, label = CustomMNISTDataset_test.mnist_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomMNISTDataset_test.mnist_dataset)

"""This code defines a class constructor that sets up data loaders for a machine learning task. The data loaders 
handle loading of data from a dataset, applying transformations to the data, and batching the data for training or 
testing a model.

Here's a general overview of what the code does:

1. **Data Transformation**: It first defines a series of transformations to be applied to the images in the dataset. 
These transformations include converting the images to grayscale, converting them to PyTorch tensors, and normalizing 
them.

2. **Memory Management**: It sets a flag for whether to pin memory or not. Pinning memory can speed up data transfer 
between CPU and GPU.

3. **Data Indexing**: It checks if there's a list of invalid indices. If there is, it adds these indices to the list 
of labeled indices. This could be used to exclude certain data points from the training process.

4. **Dataset and DataLoader Setup**: It creates a custom MNIST dataset with the defined transformations and invalid 
list. Depending on the flags `is_filter` and `is_mini`, it sets up different ways to split the data into labeled and 
unlabeled sets. It then creates PyTorch DataLoaders for the training and testing datasets. The DataLoaders handle 
batching of the data and shuffling.

5. **Data Filtering**: It defines methods to filter the data based on the class labels. This could be used in a 
scenario where you have known and unknown classes, and you want to separate the data accordingly.

The code is designed to be flexible to different scenarios, such as whether to use a GPU, whether to filter the data, 
and whether to use a smaller version of the dataset."""
class MNIST(object):
    def __init__(self, batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train=None,
                 labeled_ind_train=None, invalidList=None):
        transform = transforms.Compose([
            # Convert the images to grayscale with 3 output channels
            #transforms.Grayscale(num_output_channels=3),
            # Convert the images to PyTorch tensors
            transforms.ToTensor(),
            # Normalize the images with mean and standard deviation
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        pin_memory = True if use_gpu else False

        if invalidList is not None:
            labeled_ind_train = labeled_ind_train + invalidList

        trainset = CustomMNISTDataset_train(transform=transform, invalidList=invalidList)

        if unlabeled_ind_train == None and labeled_ind_train == None:
            if is_mini:
                labeled_ind_train, unlabeled_ind_train = self.filter_known_unknown_10percent(trainset)
                self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train
            else:
                labeled_ind_train = self.filter_known_unknown(trainset)
                self.labeled_ind_train = labeled_ind_train
        else:
            self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train

        if is_filter:
            print("openset here!")
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(labeled_ind_train),
            )
            unlabeledloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(unlabeled_ind_train),
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        CustomMNISTDataset_test.load_dataset(transform=transform)
        testset = CustomMNISTDataset_test()
        filter_ind_test = self.filter_known_unknown(testset)
        self.filter_ind_test = filter_ind_test

        if is_filter:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(filter_ind_test),
            )
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        self.trainloader = trainloader
        if is_filter: self.unlabeledloader = unlabeledloader
        self.testloader = testloader
        self.num_classes = known_class

    def filter_known_unknown(self, dataset):
        filter_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
        return filter_ind

    def filter_known_unknown_10percent(self, dataset):
        filter_ind = []
        unlabeled_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
            else:
                unlabeled_ind.append(i)

        # 随机选
        random.shuffle(filter_ind)
        labeled_ind = filter_ind[:len(filter_ind) * init_percent // 10000]
        unlabeled_ind = unlabeled_ind + filter_ind[len(filter_ind) * init_percent // 10000:]
        return labeled_ind, unlabeled_ind
