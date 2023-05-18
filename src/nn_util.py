from torch import nn
import math as Math
import torch
import logging


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
    layers.append(nn.ReLU())

    for _ in range(1, n):
        layer = nn.Linear(size, size)
        layers.append(layer)
        layers.append(nn.ReLU())

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
        nn.ReLU(),
    )


def conv_out_size(kernel_size, stride, padding, input_size):
    return Math.floor((input_size + 2 * padding - kernel_size) / stride) + 1


def conv_final_out_size(conv_layers, kernel_size, stride, padding, input_size):
    if conv_layers < 1:
        return input_size
    return conv_final_out_size(
        conv_layers - 1,
        kernel_size,
        stride,
        padding,
        conv_out_size(kernel_size, stride, padding, input_size),
    )


def get_final_layers_size(picture_size, previous_layer_size):
    return picture_size * picture_size * previous_layer_size


def simple_dist_loss(output_embds, class_embeds, targets, device):
    acc_loss = torch.tensor(0.0, requires_grad=True, device=device)
    acc_loss_div = torch.zeros(torch.Size([output_embds.size()[0] + class_embeds.size()[0], output_embds.size()[1]]), device=device, dtype=torch.float)

    # # num_of_classes = len(class_embeds)
    actual_embeds = class_embeds[targets]
    # # class_indexs = targets + output_embds.size()[0]
    diffs = output_embds - actual_embeds
    squared_dists = torch.norm(diffs, dim=1, p=2) ** 2
    # #squared_dist_divs = diffs
    acc_loss = squared_dists.sum()

    # for i, output_embedding in enumerate(output_embds):
    #     actual_index = targets[i]
    #     actual_embedding = class_embeds[actual_index]
    #     actual_index = actual_index + output_embds.size()[0]

    #     diff = output_embedding - actual_embedding
    #     squared_dist = (diff).pow(2).sum(0)
    #     squared_dist_div = diff

    #     acc_loss_div[i] = squared_dist_div
    #     acc_loss_div[actual_index] = acc_loss_div[actual_index] - squared_dist_div

    #     acc_loss = acc_loss + squared_dist

    return acc_loss, None#, acc_loss_div


def pnp_get_avg_center(class_embeds): return torch.sum(class_embeds, dim = 0) / class_embeds.size()[0]

def push_n_pull_loss(q, output_embds, class_embeds, targets, device):
    acc_loss = torch.tensor(0.0, requires_grad=True, device=device)
    avg_center = pnp_get_avg_center(class_embeds)
    
    actual_embeds = class_embeds[targets]
    class_embed_diffs = output_embds - actual_embeds
    avg_center_diffs = output_embds - avg_center

    # ce_loss =  torch.sum(class_embed_diffs**2, dim = [0,1])
    ce_loss =  torch.mean(class_embed_diffs**2)
    ac_loss = torch.mean(avg_center_diffs**2)
    acc_loss = ce_loss - q * ac_loss

    return acc_loss, None

def pnp_hyperparam(q):
    return lambda output_embeds, class_embeds, targets, device : push_n_pull_loss(q, output_embeds, class_embeds, targets, device)

def cone_loss(p, q, output_embeds, class_embeds, targets, device):
    acc_loss = torch.tensor(0.0, requires_grad=True, device=device)
    grad = torch.zeros(torch.Size([output_embeds.size()[0] + class_embeds.size()[0], output_embeds.size()[1]]), device=device, dtype=torch.float)

    class_center = torch.zeros([output_embeds.size()[1]], device=device, dtype=torch.float)
    for i, class_embedding in enumerate(class_embeds):
        class_center = class_center + class_embedding

    class_center = class_center / len(class_embeds)

    r = torch.sqrt(torch.tensor(1 - q * q))

    for i, output_embedding in enumerate(output_embeds):
        actual_index = targets[i]
        normalized_class_from_center = torch.nn.functional.normalize(class_embeds[actual_index] - class_center, dim=0)
        normalized_output_from_center = torch.nn.functional.normalize(output_embedding - class_center, dim=0)

        diff = output_embedding - class_embeds[actual_index]

        d = torch.dot(normalized_output_from_center, normalized_class_from_center)
        a = torch.nn.functional.normalize(normalized_class_from_center * d - normalized_output_from_center, dim = 0)
        scale = (1.00001 - d)**p

        actual_index = actual_index + output_embeds.size()[0]
        output_grad = scale * (-q * normalized_class_from_center - r * a)
        grad[i] = output_grad
        grad[actual_index] = grad[actual_index] - diff / len(output_embeds)

        #acc_loss = acc_loss + (diff).pow(2).sum(0)

    print(f"{torch.norm(grad[0])}, {torch.norm(grad[-1])}")
    print(f"{torch.norm(output_embeds[0])}, {torch.norm(class_embeds[-1])}")
    print("")
    return acc_loss, grad

#Doesn't work, sorry!
def cone_non_grad(output_embeds, class_embeds, targets, device):
    acc_loss = torch.tensor(0.0, requires_grad=True, device=device)
    grad = torch.zeros(torch.Size([output_embeds.size()[0] + class_embeds.size()[0], output_embeds.size()[1]]), device=device, dtype=torch.float)
    for i, output_embedding in enumerate(output_embeds):
        actual_index = targets[i]
        output_from_center = output_embedding
        normalized_class_from_center = torch.nn.functional.normalize(class_embeds[actual_index], dim=0)
        orthogonal_cfc = output_from_center - normalized_class_from_center * torch.dot(normalized_class_from_center, output_from_center)
        acc_loss = acc_loss - torch.dot(output_from_center, normalized_class_from_center) + torch.norm(orthogonal_cfc) 
    return acc_loss, None

def cone_loss_hyperparam(p=0, q=0.68):
    return lambda output_embeds, class_embeds, targets, device: cone_loss(p, q, output_embeds, class_embeds, targets, device)

def comparison_dist_loss(output_embeddings, class_embeddings, targets, device):
    loss = torch.tensor(0.0, requires_grad=True, device=device)
    # ddx_loss = torch.zeros(output.shape, device=device, dtype=torch.float)
    num_of_classes = len(class_embeddings)

    for output_embedding, target in zip(output_embeddings, targets):
        # actual_index = target_class_map[targets[i].item()] - num_of_classes#abusing negative indecies
        target_class_embedding = class_embeddings[target]

        diff_actual = output_embedding - target_class_embedding
        squared_dist_actual = (diff_actual).pow(2).sum(0)

        other_embeddings = class_embeddings[torch.arange(num_of_classes) != target]

        diff = output_embedding.unsqueeze(0) - other_embeddings
        squared_distances = torch.sum(diff**2, dim=1)
        losses = [
            squared_dist_actual / (distance + squared_dist_actual)
            for distance in squared_distances.flatten()
        ]
        loss = loss + sum(losses)

        # for ii, class_embedding in enumerate(class_embeddings):
        # if actual_index != ii:
        # diff_class = output_embedding - class_embedding
        # squared_dist_class = (diff_class).pow(2).sum(0)

        # total_squared_dist = squared_dist_actual + squared_dist_class
        # loss = loss + torch.exp(squared_dist_actual / total_squared_dist)
        # scale_value = loss_value / (total_squared_dist * total_squared_dist)

        # actual_ddx_loss = diff_actual * squared_dist_class * scale_value
        # class_ddx_loss = diff_class * squared_dist_actual * scale_value

        # ddx_loss[i] = ddx_loss[i] + (actual_ddx_loss - class_ddx_loss)
        # ddx_loss[actual_index] = ddx_loss[actual_index] - actual_class_embedding
        # ddx_loss[ii-num_of_classes] = ddx_loss[ii-num_of_classes] + class_ddx_loss

        # loss = loss + loss_value

    return loss, None  # , ddx_loss


def _move_away_from_other_near_classes_class_loss(
    proximity_multiplier,
    predicted_embeddings: list[list[float]],
    target_labels: list[int],
    class_embeddings: list[list[float]],
    device: torch.device,
):
    def proximity(x):
        return proximity_multiplier / (x + 1)

    def get_push_from_other_classes(self_label):
        self_embedding = class_embeddings[self_label]
        other_embeddings = class_embeddings[
            torch.arange(len(class_embeddings), device=device) != self_label
        ]

        distances = torch.cdist(self_embedding.unsqueeze(0), other_embeddings)
        transformed_distances = proximity(distances.flatten())
        # transformed_distances = torch.tensor(
        #     [proximity(distance) for distance in distances.flatten()], device=device, requires_grad=True
        # )
        push_amount = transformed_distances.sum()

        return push_amount

    target_labels = torch.tensor(target_labels)
    unique_labels = torch.unique(target_labels)
    # todo: set to the maximum number of classes we allow
    # currently at 1000
    push_from_other_classes = torch.zeros(1000, device=device)
    loss = torch.tensor(0.0, requires_grad=True, device=device)

    for label in unique_labels:
        # label = label.item()
        push_from_other_classes[label] = get_push_from_other_classes(label)

    target_embeds = class_embeddings[target_labels]
    diffs = predicted_embeddings - target_embeds
    dists = torch.norm(diffs, dim=1, p=2) ** 2
    pushes_from_classes_sum = push_from_other_classes[target_labels].sum()
    distance_sum = dists.sum()
    loss = distance_sum + pushes_from_classes_sum
    # print( f"dist sum: {distance_sum}, prox: {pushes_from_classes_sum.item()} \r" , end="")

    # for predicted_embedding, target_label in zip(predicted_embeddings, target_labels):
    #     dist = torch.linalg.norm(
    #         predicted_embedding - class_embeddings[target_label]
    #     ).pow(2)
    #     push_from_class = push_from_other_classes[target_label.item()]

    #     loss = loss + dist + push_from_class

    return loss, None


def dist_and_proximity_loss(proximity_multiplier: float or int):
    return lambda output_embds, class_embeds, targets, device: _move_away_from_other_near_classes_class_loss(
        proximity_multiplier=proximity_multiplier,
        predicted_embeddings=output_embds,
        target_labels=targets,
        class_embeddings=class_embeds,
        device=device,
    )

def cosine_loss(output_embeds, class_embeds, targets, device):
    target_embds = class_embeds[targets]
    dotproducts = (output_embeds * target_embds).sum(dim=1)
    
    embeds_norms = torch.norm(output_embeds, dim=1)
    target_norms = torch.norm(target_embds, dim=1)
    
    loss = (1 - (dotproducts / (embeds_norms * target_norms))).sum()
    
    return loss, None

def one_hot_cone_loss(res, labels, device):
    acc_loss = torch.tensor(0.0, requires_grad=True, device=device)
    class_embeddings = torch.nn.functional.one_hot(labels)
    for i, output_embedding in enumerate(res):
        actual_index = labels[i]
        #output_from_center = output_embedding
        #normalized_class_from_center = torch.zeros( device=device) #torch.nn.functional.normalize(class_embeds[actual_index], dim=0)
        #orthogonal_cfc = output_from_center - normalized_class_from_center * torch.dot(normalized_class_from_center, output_from_center)
        acc_loss = acc_loss - output_embedding[actual_index] + torch.norm(output_embedding[actual_index] - output_embedding[actual_index] * class_embeddings[i]) 
    return acc_loss

emc_loss_functions = {
    "simple-dist": simple_dist_loss,
    "pnp-loss": pnp_hyperparam,
    "class-push": dist_and_proximity_loss,
    "comp-dist-loss": comparison_dist_loss,
    "cone-loss": cone_loss_hyperparam,
    "cosine-loss": cosine_loss
}


def get_emc_loss_function(args, config):
    loss_func = emc_loss_functions[args.loss_func]

    if args.loss_func == "class-push":
        logging.debug(f'class-push loss: using prox mult: {config["prox_mult"]}')
        return loss_func(config["prox_mult"])
    if args.loss_func == "cone-loss":
        logging.debug(f'cone loss: using values p: {config["p"]}, q: {config["q"]}')
        return loss_func(config["p"], config["q"])
    if args.loss_func == "pnp-loss":
        logging.debug(f"pnp loss: using value q: {config['q']}")
        return loss_func(config["q"])

    return loss_func

pure_loss_functions = {
    "cross_entropy": lambda res, labels, _ : nn.CrossEntropyLoss()(res, labels),
    "cone_loss": one_hot_cone_loss
}

def get_pure_loss_function(args, config):
    return pure_loss_functions[args.loss_func]
