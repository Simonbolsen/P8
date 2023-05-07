from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import numpy as np
import logging
import os
import torch

class Kuzushiji(Dataset):

    mirrors = ["http://codh.rois.ac.jp/kmnist/dataset/k49/"]

    resources = [
        "k49-train-imgs.npz",
        "k49-train-labels.npz",
        "k49-test-imgs.npz",
        "k49-test-labels.npz",
    ]
    
    def __init__(self, root, download=False, train=True, transform=None):
        if root.startswith("'") and root.endswith("'"):
            root = root[1:-1]
            
        self.root = root
        self.train = train
        self.transform = transform

        if download:
            self._download()
        
        if not self._check_exists():
            logging.error("Dataset not downloaded")
        else:
            self.data, self.targets = self._load_data()
            
    def _download(self):
        if self._check_exists():
            print("Files already downloaded")
            return
        
        for resource in self.resources:
            url = self.mirrors[0] + resource
            download_url(url, self.root, resource)
    
    def _check_exists(self):
        for resource in self.resources:
            path_to_file = os.path.join(self.root, resource)
            if not os.path.isfile(path_to_file):
                return False
            
        return True

    def _load_data(self):
        image_file = f"k49-{'train' if self.train else 'test'}-imgs.npz"
        data = np.load(os.path.join(self.root, image_file))['arr_0']

        label_file = f"k49-{'train' if self.train else 'test'}-labels.npz"
        targets = np.load(os.path.join(self.root, label_file))['arr_0']

        return torch.from_numpy(data), torch.from_numpy(targets)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return (self.data[index], self.targets[index])

