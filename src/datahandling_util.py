import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset, Subset

def get_data(dir="./data"):
    train_data = datasets.MNIST(
        root = dir,
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )

    test_data = datasets.MNIST(
        root = dir, 
        train = False, 
        transform = ToTensor()
    )

    return train_data, test_data

def load_data(train_data, test_data, batch_size=100):
    loaders = {
        "train": torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1),
        "test": torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    }

    return loaders

    
# Remove classes from train and test data and add them to support data
def split_data(train_data, test_data, targets):
    idx_to_remove = [i for i, x in enumerate(train_data.targets) if x in targets]
    idx_to_remove_test = [i for i, x in enumerate(test_data.targets) if x in targets]

    idx_to_keep = [i for i, x in enumerate(train_data.targets) if x not in targets]
    idx_to_keep_test = [i for i, x in enumerate(test_data.targets) if x not in targets]

    removed = Subset(train_data.data, idx_to_remove)
    removed.targets = Subset(train_data.targets, idx_to_remove)

    test_removed = Subset(test_data.data, idx_to_remove_test)
    test_removed.targets = Subset(test_data.targets, idx_to_remove_test)

    support_data = ConcatDataset([removed, test_removed])
    
    train_data.data = Subset(train_data.data, idx_to_keep)
    train_data.targets = Subset(train_data.targets, idx_to_keep)

    test_data.data = Subset(test_data.data, idx_to_keep_test)
    test_data.targets = Subset(test_data.targets, idx_to_keep_test)

    return train_data, test_data, support_data

# Create k loaders for each class in support data 
# with batch size of shots
def k_shot_loaders(support_data, k, shots):
    loaders = []
    targets = set(support_data.targets)

    for i in range(k):
        target = targets[i]
        idx_targets = [j for j, x in enumerate(support_data.targets) if x == target]
        subset_data = Subset(support_data.data, idx_targets)
        loaders.append(torch.utils.data.DataLoader(subset_data, batch_size=shots, shuffle=True, num_workers=1))

    return loaders

