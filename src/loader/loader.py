import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Subset, TensorDataset
import numpy as np
import random
import loader.cifarfs_splits as cifarfs_splits
import loader.cifar10fs_splits as cifar10fs_splits
from learn2learn.vision.datasets import FC100


dataset_dict = {
    "omniglot": lambda c: get_omniglot(config=c),
    "mnist": lambda c: get_mnist(config=c),
    "cifar10": lambda c: get_cifar10(config=c),
    "cifar100": lambda c: get_cifar100(config=c),
    "cifarfs": lambda c: get_cifarfs_as_classification(config=c)
}

transforms_dict = {
    "toTensor": ToTensor(),
    "resize224_flip": transforms.Compose([
                        transforms.Resize(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]),
    "resize224": transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]),
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
    loader.unique_targets = torch.unique(data.targets)

    return loader


def get_data(config):
    return dataset_dict[config.dataset](config)



def get_mnist(config):
    training_set = datasets.MNIST(
        root=config.data_dir,
        train=True,
        transform=transforms_dict[config.train_transforms],
        download=True
    )       
    
    testing_set = datasets.MNIST(
        root=config.data_dir,
        train=False,
        transform=transforms_dict[config.test_transforms],
        download=True
    )  

    training_set.targets = torch.from_numpy(np.array(training_set.targets))
    testing_set.targets = torch.from_numpy(np.array(testing_set.targets))
    
    train_split_size = int(len(training_set) * 0.8)
    val_split_size = int(len(training_set) * 0.2)

    idx = range(len(training_set))
    random.seed(25437890)
    train_split_idx = random.sample(idx, k=train_split_size)
    remaining_idx = [i for i in idx if i not in train_split_idx]
    val_split_idx = random.sample(remaining_idx, k=val_split_size)

    train_split = [training_set.data[i] for i in train_split_idx]
    train_targets = [int(training_set.targets[i].item()) for i in train_split_idx]
    val_split = [training_set.data[i] for i in val_split_idx]
    val_targets = [int(training_set.targets[i].item()) for i in val_split_idx]

    train = CustomCifarDataset(train_split, train_targets)
    val = CustomCifarDataset(val_split, val_targets)

    train.targets = torch.tensor(train.targets, dtype=torch.int32)
    val.targets = torch.tensor(val.targets)

    return train, val, testing_set


def get_cifar10(config):
    training_set = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        transform=transforms_dict[config.train_transforms],
        download=True
    )       
    
    testing_set = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        transform=transforms_dict[config.test_transforms],
        download=True
    )       
    
    training_set.targets = torch.from_numpy(np.array(training_set.targets))
    testing_set.targets = torch.from_numpy(np.array(testing_set.targets))
    
    train_split_size = int(len(training_set) * 0.8)
    val_split_size = int(len(training_set) * 0.2)

    idx = range(len(training_set))
    random.seed(25437890)
    train_split_idx = random.sample(idx, k=train_split_size)
    remaining_idx = [i for i in idx if i not in train_split_idx]
    val_split_idx = random.sample(remaining_idx, k=val_split_size)

    train_split = [training_set.data[i] for i in train_split_idx]
    train_targets = [int(training_set.targets[i].item()) for i in train_split_idx]
    val_split = [training_set.data[i] for i in val_split_idx]
    val_targets = [int(training_set.targets[i].item()) for i in val_split_idx]

    train = CustomCifarDataset(train_split, train_targets)
    val = CustomCifarDataset(val_split, val_targets)

    train.targets = torch.tensor(train.targets, dtype=torch.int32)
    val.targets = torch.tensor(val.targets)

    return train, val, testing_set

def get_omniglot(config, target_alphabets=[]):
    background_set = datasets.Omniglot(
        root = config.data_dir,
        background = True,                         
        transform = transforms_dict[config.train_transforms], 
        download = True,            
    )

    evaluation_set = datasets.Omniglot(
        root = config.data_dir,
        background = False, 
        transform = transforms_dict[config.test_transforms],
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

def string_contains(string: str, set: list[str]):
    contains = False
    for entry in set:
        contains = contains or string.startswith(entry)
    
    return contains

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
    
    training_set.targets = torch.from_numpy(np.array(training_set.targets))
    testing_set.targets = torch.from_numpy(np.array(testing_set.targets))

    return training_set, testing_set


def get_cifarfs_as_classification(config):
    train_split, val_split, _ = get_cifarfs(config)

    all_data =  np.concatenate((train_split.data, val_split.data), axis=0)
    all_targets = torch.from_numpy(np.concatenate((train_split.targets, val_split.targets), axis=0))

    train_split_size = int(len(all_data) * 0.64)
    val_split_size = int(len(all_data) * 0.16)
    test_split_size = int(len(all_data) * 0.2)

    idx = range(len(all_data))
    random.seed(25437890)
    train_split_idx = random.sample(idx, k=train_split_size)
    remaining_idx = [i for i in idx if i not in train_split_idx]
    val_split_idx = random.sample(remaining_idx, k=val_split_size)
    remaining_idx = [i for i in remaining_idx if i not in val_split_idx]
    test_split_idx = random.sample(remaining_idx, k=test_split_size)

    train_data = [all_data[i] for i in train_split_idx]
    train_targets = [int(all_targets[i].item()) for i in train_split_idx]
    val_data = [all_data[i] for i in val_split_idx]
    val_targets = [int(all_targets[i].item()) for i in val_split_idx]
    test_data = [all_data[i] for i in test_split_idx]
    test_targets = [int(all_targets[i].item()) for i in test_split_idx]

    train = CustomCifarDataset(train_data, train_targets)
    val = CustomCifarDataset(val_data, val_targets)
    test = CustomCifarDataset(test_data, test_targets)

    train.targets = torch.tensor(train.targets)
    val.targets = torch.tensor(val.targets)
    test.targets = torch.tensor(test.targets)

    return train, val, test
    


#________________________ FEW-SHOT LAND_________________________________

fs_dataset_dict = {
    "cifarfs": lambda c: get_cifarfs(config=c),
    "cifar10": lambda c: get_cifar10_fs(config=c),
    "fc100": lambda c: get_FC100(config=c)
}

def get_fs_data(config):
    return fs_dataset_dict[config.dataset](config)

class CustomCifarDataset(torch.utils.data.Dataset):
    def __init__(self, data, data_targets):
        self.data = data
        self.targets = data_targets
        self.transform = ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.targets[idx]
    

def get_cifarfs(config):
    train_data, test_data = get_cifar100(config)

    all_data =  np.concatenate((train_data.data, test_data.data), axis=0)
    all_targets = torch.from_numpy(np.concatenate((train_data.targets, test_data.targets), axis=0))

    idx_to_class = {v : k for k,v in train_data.class_to_idx.items()}

    train_idx = [i for i, v in enumerate(all_targets) if idx_to_class[v.item()] in cifarfs_splits.train]
    test_idx = [i for i, v in enumerate(all_targets) if idx_to_class[v.item()] in cifarfs_splits.test]
    val_idx = [i for i, v in enumerate(all_targets) if idx_to_class[v.item()] in cifarfs_splits.validation]
    
    train_data_split = all_data[train_idx]
    train_target_split = all_targets[train_idx]
    test_data_split = all_data[test_idx]
    test_target_split = all_targets[test_idx]
    val_data_split = all_data[val_idx]
    val_target_split = all_targets[val_idx]

    train_split = CustomCifarDataset(train_data_split, train_target_split)
    test_split = CustomCifarDataset(test_data_split, test_target_split)
    val_split = CustomCifarDataset(val_data_split, val_target_split)

    return train_split, val_split, test_split 

def get_cifar10_fs(config):
    train_data, test_data = get_cifar10(config)

    all_data =  np.concatenate((train_data.data, test_data.data), axis=0)
    all_targets = torch.from_numpy(np.concatenate((train_data.targets, test_data.targets), axis=0))

    idx_to_class = {v : k for k,v in train_data.class_to_idx.items()}

    train_idx = [i for i, v in enumerate(all_targets) if idx_to_class[v.item()] in cifar10fs_splits.train]
    test_idx = [i for i, v in enumerate(all_targets) if idx_to_class[v.item()] in cifar10fs_splits.test]
    val_idx = [i for i, v in enumerate(all_targets) if idx_to_class[v.item()] in cifar10fs_splits.validation]
    
    train_data_split = all_data[train_idx]
    train_target_split = all_targets[train_idx]
    test_data_split = all_data[test_idx]
    test_target_split = all_targets[test_idx]
    val_data_split = all_data[val_idx]
    val_target_split = all_targets[val_idx]

    train_split = CustomCifarDataset(train_data_split, train_target_split)
    test_split = CustomCifarDataset(test_data_split, test_target_split)
    val_split = CustomCifarDataset(val_data_split, val_target_split)

    return train_split, val_split, test_split 


def get_FC100(config):
    try:
        training_set = FC100(root=config.data_dir, mode='train', transform=transforms_dict[config.train_transforms], download=True)
        validation_set = FC100(root=config.data_dir, mode='validation', transform=transforms_dict[config.train_transforms], download=True)
        test_set = FC100(root=config.data_dir, mode='test', transform=transforms_dict[config.test_transforms], download=True)
    except Exception:
        training_set = FC100(root=config.data_dir, mode='train', transform=transforms_dict[config.train_transforms], download=True)
        validation_set = FC100(root=config.data_dir, mode='validation', transform=transforms_dict[config.train_transforms], download=True)
        test_set = FC100(root=config.data_dir, mode='test', transform=transforms_dict[config.test_transforms], download=True)

    training_set.data = training_set.images
    validation_set.data = validation_set.images
    test_set.data = test_set.images

    training_set.targets = torch.from_numpy(np.array(training_set.labels))
    validation_set.targets = torch.from_numpy(np.array(validation_set.labels))
    test_set.targets = torch.from_numpy(np.array(test_set.labels))

    return training_set, validation_set, test_set

def get_CUB200(config):
    pass


# Create loaders for each class in support data 
# with batch size of shots.
# Creates loader for queries which contain the rest of the data
def k_shot_loaders(support_data, shots, query_batch_size=100):
    all_indexs_to_remove = []
    support_loaders = []
    targets = torch.unique(support_data.targets)
    
    for target in targets:
        subset_indexs = [j for j, x in enumerate(support_data.targets) if x == target][:shots]
        all_indexs_to_remove.extend(subset_indexs)
        subset_data = Subset(support_data, subset_indexs)
        subset_data.targets = torch.tensor([target])
        support_loader = get_data_loader(subset_data, batch_size=shots)
        support_loaders.append(support_loader)
        # support_loaders.append(torch.utils.data.DataLoader(subset_data, batch_size=shots, shuffle=True, num_workers=1))

    indexs_to_keep = [i for i in range(len(support_data.data)) if i not in all_indexs_to_remove]
    query_data = Subset(support_data, indexs_to_keep)
    query_data.targets = targets
    
    # query_loader = torch.utils.data.DataLoader(query_data, batch_size=query_batch_size, shuffle=True, num_workers=1)
    query_loader = get_data_loader(query_data, query_batch_size)

    return support_loaders, query_loader

if __name__ == '__main__':
    get_FC100()
    # train, test = get_omniglot(0, ["Arcadian", "Armenian"])
    # print(0)




