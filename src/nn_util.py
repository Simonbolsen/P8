from torch import nn
import math as Math
import torch
import numpy as np
import random

def create_n_linear_layers(n, first_in, size):
    """Create a set of linear layers.
    All layers have the same amount of input and output
    features except the first layer.
    
    Args:
        n (int): amount of layers
        first_in (int): input features for the first layer
        size (int): input and output features for the remaining (n - 1) layers

    Returns:
        List: the linear layers
    """
    layers = []
    
    layers.append(nn.Linear(first_in, size))
    
    for _ in range(1, n):
        layer = nn.Linear(size, size)
        layers.append(layer)
        
    return nn.Sequential(*layers)
    

def conv_layer(input, output, kernel_size, stride):
    """Create a 2D convolution layer

    Args:
        input (int): in dimensions
        output (int): out dimensions
        kernel_size (int): kernel size
        stride (int): stride

    Returns:
        nn.Sequential: the 2D convolutional layer
    """
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size=kernel_size, stride=stride, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU()
    )

def conv_out_size(kernel_size, stride, padding, input_size):
    return Math.floor((input_size + 2 * padding - kernel_size) / stride) + 1

def conv_final_out_size(conv_layers, kernel_size, stride, padding, input_size):
    if (conv_layers < 1):
        return input_size
    return conv_final_out_size(conv_layers-1, kernel_size, stride, padding, conv_out_size(kernel_size, stride, padding, input_size))

def get_final_layers_size(picture_size, previous_layer_size):
    return picture_size * picture_size * previous_layer_size

def simple_dist_loss(output, target, num_of_classes, device):
    acc_loss = torch.tensor(0.0, requires_grad=True, device=device)
    acc_loss_div = torch.zeros(output.shape, device=device, dtype=torch.float)

    for i, output_embedding in enumerate(output[:-num_of_classes]):
        actual_index = target[i].item() - num_of_classes
        actual_embedding = output[actual_index]

        diff = output_embedding - actual_embedding
        squared_dist = (diff).pow(2).sum(0)
        squared_dist_div = diff

        acc_loss_div[i] = squared_dist_div
        acc_loss_div[actual_index] = acc_loss_div[actual_index] - squared_dist_div

        acc_loss = acc_loss + squared_dist

    return acc_loss, acc_loss_div



