import embedding_model as emb_model
from torch import optim
from loader.loader import get_data_loader, k_shot_loaders
from training_utils import train
from ray import tune
import torch
import sys

def train_few_shot(config, 
                   train_data, 
                   few_shot_data, 
                   validation_data, 
                   loss_func, device, ray_tune = True):
    train_loader = get_data_loader(train_data, config["batch_size"])
    fs_sup_loader, fs_query_load = k_shot_loaders(few_shot_data, config["shots"])
    val_sup_load, val_query_load = k_shot_loaders(validation_data, config["shots"])     
    
    img_size = train_loader.image_size
    img_channels = train_loader.image_channels
    
    model = emb_model.Convnet(device, config["lr"], 
                              config["d"], 
                              config["num_of_classes"], 
                              config["channels"],
                              config["k_size"],
                              config["stride"],
                              img_channels, img_size, config["linear_n"], config["linear_size"]).to(device)
    
    optimiser = optim.Adam(model.parameters(), lr=config["lr"])
    
    max_epochs = config["num_of_epochs"]
    
    last_accuracy = 0
    for epoch in range(max_epochs):
        train(model, train_loader, optimiser, loss_func, max_epochs, epoch, device)
        correct, total = few_shot_eval(model, fs_sup_loader, fs_query_load)
        accuracy = sum(correct) / sum(total)
        if ray_tune:
            tune.report(accuracy = accuracy)
    
    print("doing final evaluation...")
    val_acc = few_shot_eval(model, val_sup_load, val_query_load)
    # todo : report final acc
    # tune.report(val_acc = val_acc)

def few_shot_eval(model, support_loaders, query_loader, device):
    # Test the model
    model.eval()
    with torch.no_grad():
        # Get the targets we have not seen before
        few_shot_targets = find_few_shot_targets(support_loaders)
        num_of_new_classes = len(few_shot_targets)

        new_class_embeddings = []
        correct = [0] * num_of_new_classes
        total = [0] * num_of_new_classes

        new_class_embeddings = get_few_shot_embeddings(support_loaders, model, device)
        
        # average embeddings for class
        new_class_embeddings = [sum(item) / len(item) for item in new_class_embeddings]

        # ensure lengths but 
        # assume the order is preserved
        assert len(new_class_embeddings) == len(few_shot_targets)

        # do evaluation
        for images, labels in query_loader:
            # todo: remove hardcoded shape
            images = images.view(-1, 1, 28, 28).float().to(device)
            test_output = model(images)

            for i, output_embedding in enumerate(test_output[:-model.num_of_classes]):
                closest_target_index = find_closest_embedding(output_embedding, new_class_embeddings)
                predicted_target = few_shot_targets[closest_target_index]

                if predicted_target == labels[i].item():
                    correct[closest_target_index] += 1
                total[closest_target_index] += 1

        return correct, total
    
def find_few_shot_targets(support_loaders):
    few_shot_targets = []
    for loader in support_loaders:
        for _, labels in loader:
            few_shot_targets.append(labels[0].item())
            break
    return few_shot_targets

def get_few_shot_embeddings(support_loaders, model, device):
    new_class_embeddings = []

    for loaders in support_loaders:
        for images, _ in loaders:
            # todo: remove hardcode shape
            # ensure correct shape
            images = images.view(-1, 1, 28, 28).float().to(device)
            few_shot_output = model(images)
            new_class_embeddings.append(few_shot_output[:-model.num_of_classes])
    
    return new_class_embeddings

def find_closest_embedding(query, class_embeddings):
    smallest_sqr_dist = sys.maxsize
    closest_target_index = 0
    for i, embedding in enumerate(class_embeddings):
        squared_dist = (embedding - query).pow(2).sum(0)
        if squared_dist < smallest_sqr_dist:
            smallest_sqr_dist = squared_dist
            closest_target_index = i
    
    return closest_target_index



