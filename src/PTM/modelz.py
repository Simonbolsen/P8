from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import re
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def fine_tune_pretrained(model_name, model_constructor, num_classes, feature_extract=False):
    """
    Fine-tune a pretrained model
    :param model_name: name of the pretrained model requested
    :param model_constructor: constructor for the model
    :param num_classes: number of classes to classify
    :param feature_extract: True if only the last layer is to be trained
    :return: model
    """
    model = None
    input_size = 0

    if model_name == "resnet":
        model = model_constructor(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        "gm"

    return model, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

fine_tune_pretrained("resnet18", models.resnet18, 10, True)