import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset, Subset, TensorDataset
import numpy as np
import loader.cifarfs_splits as cifarfs_splits


dataset_dict = {
    "omniglot": lambda c: get_omniglot(config=c),
    "mnist": lambda c: get_mnist(config=c),
    "cifar10": lambda c: get_cifar10(config=c),
    "cifar100": lambda c: get_cifar100(config=c),
    "cifarfs": lambda c: get_cifarfs(config=c),
}

def load_data(train_data, test_data, batch_size=100):
    loaders = {
        "train": torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1),
        "test": torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    }

    return loaders

def get_data_loader(data, batch_size=100):
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
    loader.image_size = data[0][0].size()[1]
    loader.channels = data[0][0].size()[0]

    return loader


def get_data(config):
    return dataset_dict[config.dataset](config)



def get_mnist(config):
    train_data = datasets.MNIST(
        root = config.data_dir,
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )

    test_data = datasets.MNIST(
        root = config.data_dir, 
        train = False, 
        transform = ToTensor()
    )
    
    train_data = Subset(train_data, range(len(train_data)))

    return train_data, test_data


def get_cifar10(config):
    training_set = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        transform=ToTensor(),
        download=True
    )       
    
    testing_set = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        transform=ToTensor(),
        download=True
    )       
    
    return training_set, testing_set

def get_cifar100(config):
    training_set = datasets.CIFAR100(
        root=config.data_dir,
        train=True,
        transform=ToTensor(),
        download=True
    )

    testing_set = datasets.CIFAR100(
        root=config.data_dir,
        train=False,
        transform=ToTensor(),
        download=True
    ) 

    return training_set, testing_set

class CustomCifarDataset(torch.utils.data.Dataset):
    def __init__(self, data, data_targets):
        self.data = data
        self.targets = data_targets
        self.transform = ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), torch.from_numpy(self.targets)[idx]
    

def get_cifarfs(config):
    train_data, test_data = get_cifar100(config)

    all_data =  np.concatenate((train_data.data, test_data.data), axis=0)
    all_targets = np.concatenate((train_data.targets, test_data.targets), axis=0)

    train_idx = [train_data.class_to_idx[v] for v in cifarfs_splits.train]
    test_idx = [train_data.class_to_idx[v] for v in cifarfs_splits.test]
    val_idx = [train_data.class_to_idx[v] for v in cifarfs_splits.validation]
    
    train_data_split = all_data[train_idx]
    train_target_split = all_targets[train_idx]
    test_data_split = all_data[test_idx]
    test_target_split = all_targets[test_idx]
    val_data_split = all_data[val_idx]
    val_target_split = all_targets[val_idx]

    train_split = CustomCifarDataset(train_data_split, train_target_split)
    test_split = CustomCifarDataset(test_data_split, test_target_split)
    val_split = CustomCifarDataset(val_data_split, val_target_split)

    return train_split, test_split, val_split


def get_omniglot(config, target_alphabets=[]):
    background_set = datasets.Omniglot(
        root = config.data_dir,
        background = True,                         
        transform = ToTensor(), 
        download = True,            
    )

    evaluation_set = datasets.Omniglot(
        root = config.data_dir,
        background = False, 
        transform = ToTensor(),
        download=True,
    )

    if target_alphabets:
        characters = [k for (k, v) in enumerate(background_set._characters) if string_contains(v, target_alphabets)]

        background_data = Subset(background_set, characters)
        evaluation_data = Subset(evaluation_set, characters)
    else:
        background_data = background_set
        evaluation_data = evaluation_set

    return background_data, evaluation_data

def string_contains(string, set):
    contains = False
    for entry in set:
        contains = contains or string.startswith(entry)
    
    return contains


if __name__ == '__main__':
    train, test = get_omniglot(0, ["Arcadian", "Armenian"])
    print(0)




