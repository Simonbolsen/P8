import ray
import embedding_model as emb_model
from torch import optim
from loader.loader import get_data_loader, k_shot_loaders
from training_utils import train
from ray import tune
import torch
import sys
from nn_util import get_loss_function
from PTM.model_loader import load_pretrained
from ray import tune
import json

def get_few_shot_loaders(config, train_data, few_shot_data):
    train_loader = get_data_loader(train_data, config["batch_size"])
    fs_sup_loader, fs_query_load = k_shot_loaders(few_shot_data, config["shots"])
    
    return train_loader, fs_sup_loader, fs_query_load

def setup_few_shot_pretrained(config, model_name, train_data, few_shot_data, device, args, ray_tune = True):
    train_loader, fs_sup_loader, fs_query_loaders = get_few_shot_loaders(config, ray.get(train_data), ray.get(few_shot_data))
    loss_func = get_loss_function(args)
    num_of_classes = len(train_loader.unique_targets)
    model, _ = load_pretrained(model_name, num_of_classes, 
                            config["d"], train_loader.image_size, 
                            train_loader.channels, device)
    model.to(device)
    
    train_few_shot(config, train_loader, fs_sup_loader, fs_query_loaders, 
                model, loss_func, device, ray_tune)

def train_few_shot(config, train_loader, fs_sup_loaders, fs_query_load, 
                   model, loss_func, device, ray_tune):
    
    optimiser = optim.Adam(model.parameters(), lr=config["lr"])
    
    max_epochs = config["max_epochs"]
    
    last_acc = 0
    for epoch in range(max_epochs):
        print("training...")
        train(model, train_loader, optimiser, loss_func, max_epochs, epoch, device)
        print("evaluating...")
        last_acc = few_shot_eval(model, fs_sup_loaders, fs_query_load, device)
        if ray_tune:
            tune.report(accuracy = last_acc)
        else:
            print(f"Validation accuracy: {last_acc}")
    
    save_few_shot_embedding_result(train_loader, fs_sup_loaders, fs_query_load, model, config, last_acc, device)

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
            img_size = query_loader.image_size
            channels = query_loader.channels
            images = images.view(-1, channels, img_size, img_size).float().to(device)
            
            test_output = model(images)

            for i, output_embedding in enumerate(test_output[:-model.num_of_classes]):
                closest_target_index = find_closest_embedding(output_embedding, new_class_embeddings)
                predicted_target = few_shot_targets[closest_target_index]

                if predicted_target == labels[i].item():
                    correct[closest_target_index] += 1
                total[closest_target_index] += 1

        return sum(correct) / sum(total)
    
def find_few_shot_targets(support_loaders):
    few_shot_targets = []
    for loader in support_loaders:
        for _, labels in loader:
            few_shot_targets.append(labels[0].item())
            break
    return few_shot_targets

def get_few_shot_embeddings(support_loaders, model, device):
    new_class_embeddings = []
    for loader in support_loaders:
        img_size = loader.image_size
        channels = loader.channels
       
        for images, _ in loader:
            images = images.view(-1, channels, img_size, img_size).float().to(device)
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


def save_few_shot_embedding_result(train_loader, support_loaders, query_loader, model, config, accuracy, device):
    train_embeddings = []
    val_support_embeddings = []
    val_query_embeddings = []

    model.eval()

    train_labels = []
    val_support_labels = []
    val_query_labels = []

    for images, labels in train_loader: 
        train_embeddings.extend(model.model(images.to(device)).tolist())
        train_labels.extend(labels.tolist())
    
    for support_loader in support_loaders:
        for images, labels in support_loader: 
            val_support_embeddings.extend(model.model(images.to(device)).tolist())
            val_support_labels.extend(labels.tolist())

    for images, labels in query_loader:
        val_query_embeddings.extend(model.model(images.to(device)).tolist())
        val_query_labels.extend(labels.tolist())

    new_class_embeds = []
    few_shot_embeds = get_few_shot_embeddings(support_loaders, model, device)
    for few_shot_embed in few_shot_embeds:
        new_class_embeds.append(few_shot_embed.tolist())

    embeddings = {"train_embeddings": train_embeddings, "train_labels": train_labels, 
                  "val_support_embeddings": val_support_embeddings, "val_support_labels": val_support_labels,
                  "val_query_embeddings": val_query_embeddings, "val_query_labels": val_query_labels,
                  "new_class_embeddings": new_class_embeds,
                  "class_embeddings": model.get_embeddings().tolist(), "accuracy" : accuracy, "config" : config}

    with open('embeddingData/few_shot_test_data.json', 'w') as outfile:
        json.dump(json.dumps(embeddings), outfile)


""" {
    "train_embeddings", "train_labels", val_support_embeddings, val_support_labels, 
    val_query_embeddings, val_query_labels, "class_embeddings", "new_class_embeddings", 
    accuracy, config
}"""