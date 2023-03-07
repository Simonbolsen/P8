import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datahandling_util import get_data, load_data
from model_loader import load_pretrained
from trainer import train_model
from torchvision import models, transforms
import torch
from functools import partial
from torch import nn
from torch import optim
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import nn_util
import embedding_model as emb_model
import Plotting.plotting_util as plot
import math

def main():
    
    #Hyperparameters, model and data are configured. This can be messed with to see how it affects the model
    model, input_size = load_pretrained("resnet18", num_classes=10)

    train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    train_data, test_data = get_data()

    train_data.transform = train_transforms
    test_data.transform = test_transform

    loaders = load_data(train_data, test_data)


    optimiser = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn_util.simple_dist_loss
    target_class_map = { i:i for i in range(10) }
    epochs = 10

    train_model(model, loaders, loss_func, optimiser, epochs)

def train_all():
    all_models = models.list_models(module="torchvision")
    for model in all_models:
        print("Training: ", model)
        train_model(model)

if __name__ == '__main__':
    main()

