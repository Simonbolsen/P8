import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset, Subset, TensorDataset

class SupportDataset(torch.utils.data.Dataset):
    def __init__(self, support_data, support_data_targets):
        self.data = support_data
        self.targets = support_data_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

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

    support_data_concat = torch.cat((train_data.data[idx_to_remove], 
                                     test_data.data[idx_to_remove_test]), 0)
    
    support_targets_concat = torch.cat((train_data.targets[idx_to_remove], 
                                        test_data.targets[idx_to_remove_test]), 0)
    
    support_dataset = SupportDataset(support_data_concat, support_targets_concat)
    
    train_data.data = train_data.data[idx_to_keep]
    train_data.targets = train_data.targets[idx_to_keep]

    test_data.data = test_data.data[idx_to_keep_test]
    test_data.targets = test_data.targets[idx_to_keep_test]

    return train_data, test_data, support_dataset

# Create loaders for each class in support data 
# with batch size of shots
def k_shot_loaders(support_data, shots):
    loaders = []
    targets = torch.unique(support_data.targets)
    
    for i in range(k):
        target = targets[i]
        idx_targets = [j for j, x in enumerate(support_data.targets) if x == target]
        subset_data = Subset(support_data, idx_targets)
        loaders.append(torch.utils.data.DataLoader(subset_data, batch_size=shots, shuffle=True, num_workers=1))

    return loaders

