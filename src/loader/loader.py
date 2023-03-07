import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset, Subset, TensorDataset


dataset_dict = {
    "omniglot": lambda c: load_omniglot(config=c, target_alphabets=["Arcadian", "Armenian"]),
    "mnist": lambda c: load_mnist(config=c),
    "cifar10": lambda c: load_cifar10(config=c)
}

def load_data(config):
    return dataset_dict[config.dataset](config)



def load_mnist(config):
    pass

def load_cifar10(config):
    pass

def load_omniglot(config, target_alphabets=[]):
    background_set = datasets.Omniglot(
        root = "./data", #config.data_dir,
        background = True,                         
        transform = ToTensor(), 
        download = True,            
    )

    evaluation_set = datasets.Omniglot(
        root = "./data", #config.data_dir,
        background = False, 
        transform = ToTensor(),
        download=True,
    )

    if (len(target_alphabets) > 0):
        characters = [k for (k, v) in enumerate(background_set._characters) if string_contains(v, target_alphabets)]

        background_loader = torch.utils.data.DataLoader(Subset(background_set, characters), batch_size=config.batch_size, shuffle=True, num_workers=1)
        evaluation_loader = torch.utils.data.DataLoader(Subset(evaluation_set, characters), batch_size=config.batch_size, shuffle=True, num_workers=1)
    else:
        background_loader = torch.utils.data.DataLoader(background_set, characters, batch_size=config.batch_size, shuffle=True, num_workers=1)
        evaluation_loader = torch.utils.data.DataLoader(evaluation_set, characters, batch_size=config.batch_size, shuffle=True, num_workers=1)

    return {"train": background_loader, "test": evaluation_loader}

def string_contains(string, set):
    contains = False
    for entry in set:
        contains = contains or string.startswith(entry)
    
    return contains



if __name__ == '__main__':
    train, test = load_omniglot(0, ["Arcadian", "Armenian"])
    print(0)




