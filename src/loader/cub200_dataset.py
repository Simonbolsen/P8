import torch
import tarfile
import os
import io
from skimage import io, transform
import pandas as pd
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url
from torch.utils.data import ConcatDataset, Subset, TensorDataset, Dataset
import logging

class Cub200(Dataset):
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    folder_name = "CUB_200_2011"
    filename = folder_name + ".tgz"
    
    checksum = "97eceeb196236b17998738112f37df78"

    def __init__(self, root, download=False, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if download:
            self._download()
        
        if not self._check_exists():
            logging.error("Dataset not downloaded")
            

    def _load_meta(self):
        folder_path = os.path.join(self.root, self.folder_name)
        images = pd.read_csv(os.path.join(folder_path, "images.txt"), sep=" ", names=["img_id", "img_path"])
        image_labels = pd.read_csv(os.path.join(folder_path, "image_class_labels.txt"), sep=" ", names=["img_id", "target"])
        train_test_split = pd.read_csv(os.path.join(folder_path, "train_test_split.txt"), sep=" ", names=["img_id", "is_training"])

        image_data = images.merge(image_labels, on="img_id")
        self.image_data = image_data.merge(train_test_split, on="img_id")
        self.image_data[['img_folder', 'img_name']] = self.image_data['img_path'].str.split('/', 1, expand=True)

        if self.train:
            self.image_data = self.image_data[self.image_data['is_training'] == 1]
        else:
            self.image_data = self.image_data[self.image_data['is_training'] == 0]        

    def _check_exists(self):
        try:
            self._load_meta()
        except Exception:
            return False

        for _, row in self.image_data.iterrows():
            image_path = os.path.join(self.root, self.folder_name, 'images', row['img_folder'], row['img_name'])
            if not os.path.isfile(image_path):
                return False

        return True

    def _download(self):
        
        if (self._check_exists()):
            print("Files already downloaded")
            return
        
        download_url(self.url, self.root, self.filename, self.checksum)

        file_path = os.path.join(self.root, self.filename)
        with tarfile.open(file_path, "r:gz") as file:
            file.extractall(path=self.root)

    def __getitem__(self, index):
        data_point = self.image_data.iloc[index]

        img_path = os.path.join(self.root, self.folder_name, 'images', data_point['img_folder'], data_point['img_name'])
        img = self._load_img(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, data_point['target'] - 1


    def _load_img(self, img_path):
        return io.imread(img_path)

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')
    data = Cub200(root=path, download=True, transform=ToTensor())
    res = data[0]
    print("")



