from torch import nn
import math as Math
import torch
import numpy as np
import random

def conv_layer(input, output, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size=kernel_size, stride=stride),
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

def simple_dist_loss(output, target, num_of_classes, target_class_map, device):
    acc_loss = torch.tensor(0.0, requires_grad=True, device=device)
    acc_loss_div = torch.zeros(output.shape, device=device, dtype=torch.float)

    for i, output_embedding in enumerate(output[:-num_of_classes]):
        actual_index = target_class_map[target[i].item()] - num_of_classes
        actual_embedding = output[actual_index]

        diff = output_embedding - actual_embedding
        squared_dist = (diff).pow(2).sum(0)
        squared_dist_div = diff

        acc_loss_div[i] = squared_dist_div
        acc_loss_div[actual_index] = acc_loss_div[actual_index] - squared_dist_div

        acc_loss = acc_loss + squared_dist

    return acc_loss, acc_loss_div



def move_away_from_other_near_classes_output_loss(predicted_embeddings:list[list[float]], target_labels:list[int], class_embeddings:list[list[float]], device: torch.device):
    loss = torch.tensor(0.0, requires_grad=True, device=device)

    for predicted_embedding, target_label in zip(predicted_embeddings, target_labels):
        dist = torch.linalg.norm(predicted_embedding - class_embeddings[target_label]).pow(2)
        loss = loss + dist

    return loss


def move_away_from_other_near_classes_class_loss(predicted_embeddings:list[list[float]], target_labels:list[int], class_embeddings:list[list[float]], device: torch.device):
    def transform(x): return 1/x
    def get_push_from_other_classes(self_label):
        self_embedding = class_embeddings[self_label]
        push_amount = 0
        for other_label in range(class_embeddings):
            if self_label == other_label:
                continue
            other_embedding = class_embeddings[other_label]

            push_amount = push_amount + transform(torch.linalg.norm(self_embedding - other_embedding))
        
        return push_amount



    loss = torch.tensor(0.0, requires_grad=True, device=device)

    for predicted_embedding, target_label in zip(predicted_embeddings, target_labels):
        dist = torch.linalg.norm(predicted_embedding - class_embeddings[target_label]).pow(2)
        push_from_other_classes = get_push_from_other_classes(target_label)

        loss = loss + dist + push_from_other_classes

    return loss



# def get_class_embeddings(res, number_of_classes):
#     return res[-number_of_classes:]
