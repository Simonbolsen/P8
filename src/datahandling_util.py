import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

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