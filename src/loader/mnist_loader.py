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
from PIL import Image
import struct
import sys

from array import array
import csv
import png

# Largely inspired by: https://github.com/myleott/mnist_png/blob/master/convert_mnist_to_png.py
class FashionMNIST(Dataset):
    folder_name = "FashionMNIST"
    files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz", 
            "t10k-images-idx3-ubyte.gz", 
            "t10k-labels-idx1-ubyte.gz"
        ]

    def __init__(self, root, download=False, train=True, transform=ToTensor()):
        if root.startswith("'") and root.endswith("'"):
            root = root[1:-1]
        self.root = root
        self.train = train
        self.transform = transform

        if download:
            self._download()
        
        if not self._check_exists():
            logging.error("Dataset not downloaded")

        self.load_raw_data()
        if not self._check_pngs_exist():
            logging.error("Files not extracted")
            self.save_as_pngs()

        self.load_data()

    def load_data(self):
        dir_name = "training" if self.train else "testing"
        folder_path = os.path.join(self.root, self.folder_name, dir_name)
        images = pd.read_csv(os.path.join(folder_path, "meta_data.csv"), sep=",", names=["i", "training", "folder", "target", "img_name", "full_name"])
        self.data = images
        
        self.targets = self.data.target

    def load_raw_data(self):
        if self.train:
            fname_img = os.path.join(self.root, self.folder_name, "raw", 'train-images-idx3-ubyte')
            fname_lbl = os.path.join(self.root, self.folder_name, "raw", 'train-labels-idx1-ubyte')
        else:
            fname_img = os.path.join(self.root, self.folder_name, "raw", 't10k-images-idx3-ubyte')
            fname_lbl = os.path.join(self.root, self.folder_name, "raw", 't10k-labels-idx1-ubyte')

        flbl = open(fname_lbl, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = array("b", flbl.read())
        flbl.close()

        fimg = open(fname_img, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = array("B", fimg.read())
        fimg.close()

        self.size = size
        self.rows = rows
        self.cols = cols
        self.data = img
        self.labels = lbl



    def save_as_pngs(self):
        entries = []
        dir_name = "training" if self.train else "testing"
        output_dirs = [
            os.path.join(self.root, self.folder_name, dir_name, str(i))
            for i in range(10)
        ]
        for dir in output_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
        logging.error("Extracting files")
        # write data
        for (i, label) in enumerate(self.labels):
            output_filename = os.path.join(output_dirs[label], str(i if self.train else i + 60000) + ".png")
            entries.append({"i": i, "training": self.train, "folder": output_dirs[label], "target": label, "img_name": str(i if self.train else i + 60000) + ".png", "full_name": output_filename})
            #print("writing " + output_filename)
            with open(output_filename, "wb") as h:
                w = png.Writer(self.cols, self.rows, greyscale=True)
                data_i = [
                    self.data[ (i*self.rows*self.cols + j*self.cols) : (i*self.rows*self.cols + (j+1)*self.cols) ]
                    for j in range(self.rows)
                ]
                w.write(h, data_i)

        with open(os.path.join(self.root, self.folder_name, dir_name, "meta_data.csv"), "w", newline='') as f:
            writer = csv.DictWriter(f, entries[0].keys())
            writer.writerows(entries)

    def _check_pngs_exist(self):
        dir_name = "training" if self.train else "testing"
        folder_name = os.path.join(self.root, self.folder_name, dir_name)
        num_per_class = 6000 if self.train else 1000
        
        if not os.path.isfile(os.path.join(folder_name, "meta_data.csv")):
            return False

        for i in range(10):
            sub_dir_path = os.path.join(folder_name, str(i))
            if not os.path.isdir(sub_dir_path):
                return False
            if len([name for name in os.listdir(sub_dir_path) if os.path.isfile(os.path.join(sub_dir_path, name))]) != num_per_class:
                return False
            
        return True

    def _check_exists(self):
        files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz", 
            "t10k-images-idx3-ubyte.gz", 
            "t10k-labels-idx1-ubyte.gz"
        ]
        
        for file_name in files:
            image_path = os.path.join(self.root, self.folder_name, "raw", file_name)
            if not os.path.isfile(image_path):
                return False

        return True

    def _download(self):
        
        if (self._check_exists()):
            print("Files already downloaded")
            return
        
        datasets.FashionMNIST(
            root=self.root,
            train=True,
            download=True
        )       
        
        datasets.FashionMNIST(
            root=self.root,
            train=False,
            download=True
        ) 

    def __getitem__(self, index):
        data_point = self.data.iloc[index]

        img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data', self.folder_name, 'training' if self.train else 'testing', str(data_point['target']), data_point['img_name'])

        img = self._load_img(img_path)
        img = Image.fromarray(img).convert('L')
        img = Image.merge('RGB', (img,img,img))

        if self.transform is not None:
            img = self.transform(img)

        assert data_point.target == self.targets[index]
        return img, self.targets[index]

    def __len__(self):
        return len(self.data)

    def _load_img(self, img_path):
        return io.imread(img_path)

class MNIST(FashionMNIST):
    folder_name = "MNIST"

    def _download(self):
        if (self._check_exists()):
            print("Files already downloaded")
            return
        
        datasets.MNIST(
            root=self.root,
            train=True,
            download=True
        )       
        
        datasets.MNIST(
            root=self.root,
            train=False,
            download=True
        ) 
    


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')
    train_data = FashionMNIST(root=path, download=True, transform=ToTensor())
    test_data = FashionMNIST(root=path, download=True, train=False, transform=ToTensor())
    train_data[0]
    print("")
