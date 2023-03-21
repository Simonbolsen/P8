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

def simple_dist_loss(output_embds, class_embeds, targets, device):
    acc_loss = torch.tensor(0.0, requires_grad=True, device=device)
    acc_loss_div = torch.zeros(output_embds.shape, device=device, dtype=torch.float)
    
    # num_of_classes = len(class_embeds)

    for i, output_embedding in enumerate(output_embds):
        actual_index = targets[i]
        actual_embedding = class_embeds[actual_index]

        diff = output_embedding - actual_embedding
        squared_dist = (diff).pow(2).sum(0)
        squared_dist_div = diff

        acc_loss_div[i] = squared_dist_div
        acc_loss_div[actual_index] = acc_loss_div[actual_index] - squared_dist_div

        acc_loss = acc_loss + squared_dist

    return acc_loss#, acc_loss_div

def comparison_dist_loss(output_embeddings, class_embeddings, targets, device):
    loss = torch.tensor(0.0, requires_grad=True, device=device)
    #ddx_loss = torch.zeros(output.shape, device=device, dtype=torch.float)
    num_of_classes = len(class_embeddings)

    for output_embedding, target in zip(output_embeddings, targets):
        # actual_index = target_class_map[targets[i].item()] - num_of_classes#abusing negative indecies
        target_class_embedding = class_embeddings[target]

        diff_actual = output_embedding - target_class_embedding
        squared_dist_actual = (diff_actual).pow(2).sum(0)

        other_embeddings = class_embeddings[torch.arange(num_of_classes) != target]

        diff = output_embedding.unsqueeze(0) - other_embeddings
        squared_distances = torch.sum(diff**2, dim=1)
        losses = [torch.exp(squared_dist_actual / (distance + squared_dist_actual)) for distance in squared_distances.flatten()]
        loss = loss + sum(losses)

        #for ii, class_embedding in enumerate(class_embeddings):
            #if actual_index != ii:
                #diff_class = output_embedding - class_embedding
                #squared_dist_class = (diff_class).pow(2).sum(0)

                #total_squared_dist = squared_dist_actual + squared_dist_class                
                #loss = loss + torch.exp(squared_dist_actual / total_squared_dist)
                #scale_value = loss_value / (total_squared_dist * total_squared_dist)

                #actual_ddx_loss = diff_actual * squared_dist_class * scale_value
                #class_ddx_loss = diff_class * squared_dist_actual * scale_value

                #ddx_loss[i] = ddx_loss[i] + (actual_ddx_loss - class_ddx_loss)
                #ddx_loss[actual_index] = ddx_loss[actual_index] - actual_class_embedding
                #ddx_loss[ii-num_of_classes] = ddx_loss[ii-num_of_classes] + class_ddx_loss

                #loss = loss + loss_value

    return loss #, ddx_loss



def _move_away_from_other_near_classes_class_loss(proximity_multiplier, predicted_embeddings:list[list[float]], target_labels:list[int], class_embeddings:list[list[float]], device: torch.device):
    def proximity(x): return proximity_multiplier / (x + 1)
    def get_push_from_other_classes(self_label):
        self_embedding = class_embeddings[self_label]
        other_embeddings = class_embeddings[torch.arange(len(class_embeddings), device=device) != self_label]
        
        distances = torch.cdist(self_embedding.unsqueeze(0), other_embeddings)
        transformed_distances = torch.tensor([proximity(distance) for distance in distances.flatten()], device=device)
        push_amount = transformed_distances.sum()

        return push_amount

    unique_labels = torch.unique(target_labels)
    push_from_other_classes = {}
    loss = torch.tensor(0.0, requires_grad=True, device=device)

    for label in unique_labels:
        label = label.item()
        push_from_other_classes[label] = get_push_from_other_classes(label)

    for predicted_embedding, target_label in zip(predicted_embeddings, target_labels):
        dist = torch.linalg.norm(predicted_embedding - class_embeddings[target_label]).pow(2)
        push_from_class = push_from_other_classes[target_label.item()]

        loss = loss + dist + push_from_class

    return loss


def dist_and_proximity_loss(proximity_multiplier:float or int):
    return lambda output_embds, class_embeds, targets, device: _move_away_from_other_near_classes_class_loss(
            proximity_multiplier = proximity_multiplier,
            predicted_embeddings = output_embds,
            target_labels = targets,
            class_embeddings = class_embeds,
            device = device
        )

def get_loss_function(config):
    loss_functions = {
        "simple-dist" : simple_dist_loss,
        "class-push" : dist_and_proximity_loss,
        "comp-dist-loss" : comparison_dist_loss
    }

    loss_func = loss_functions[config["loss_func"]]

    if config.loss_func == "class-push":
        loss_func = loss_func(config["prox_mult"])
            
    return loss_func