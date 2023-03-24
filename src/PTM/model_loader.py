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

def load_pretrained(model_name, num_classes, embedding_dim_count, image_size, img_channels, device, feature_extract=False):
    """
    Fine-tune a pretrained model
    :param model_name: name of the pretrained model requested
    ['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'densenet121', 'densenet161', 'densenet169',
    'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
    'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'googlenet', 'inception_v3',
    'maxvit_t', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
    'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf',
    'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf',
    'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d',
    'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0',
    'squeezenet1_1', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
    'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']
    :param model_constructor: constructor for the model
    :param num_classes: number of classes to classify
    :param embedding_dim_count: The number of dimensions for the embedding
    :param feature_extract: True if only the last layer is to be trained
    :return: model
    """
    
    model = models.get_model(model_name)
    model.num_of_classes = num_classes
    input_size = 0

    #split the model name upon first non letter encountered
    split_name = re.split(r'(\d+)', model_name, 1)
    set_parameter_requires_grad(model, feature_extract)
    
    # for param in model.parameters():
    #     print (param.data)

    if split_name[0] == "resnet":
        model.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, embedding_dim_count)
        model.embeddings = nn.Embedding(num_classes, embedding_dim_count)
        
        def forward(x):
            x = model._forward_impl(x)
            y = model.embeddings(torch.tensor(range(num_classes), device=device))
            return torch.cat((x, y), dim=0)
        model.forward = forward

        input_size = 224

    elif model_name == "alexnet":
        model.features[0] = nn.Conv2d(img_channels, 64, kernel_size=11, stride=4, padding=2)

        #Output stems from the sixth layer in the dense net at the end called classifier
        model.classifier[6] = nn.Linear(4096, embedding_dim_count)
        model.embeddings = nn.Embedding(num_classes, embedding_dim_count)

        def forward(x):
            x = model.features(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            x = model.classifier(x)
            y = model.embeddings(torch.tensor(range(num_classes), device=device))
            return torch.cat((x, y), dim=0)
        model.forward = forward

        input_size = 224
    
    elif model_name == "densenet":
        "Nicht implementiert"

    return model, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# load_pretrained("resnet18", 10)