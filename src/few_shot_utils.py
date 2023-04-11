import ray
import embedding_model as emb_model
from torch import optim
from loader.loader import get_data_loader, k_shot_loaders
from training_utils import find_closest_embedding, train_emc
from ray import tune
import torch
from nn_util import get_emc_loss_function
from PTM.model_loader import load_pretrained
from ray import tune
import logging
import embedding_util as eu
import json_util as ju

def get_few_shot_loaders(config, train_data, few_shot_data):
    train_loader = get_data_loader(train_data, config["batch_size"])
    fs_sup_loaders, fs_query_loader = k_shot_loaders(few_shot_data, config["shots"])
    
    return train_loader, fs_sup_loaders, fs_query_loader

def setup_few_shot_custom_model(config, train_data_ptr, few_shot_data_ptr, device, args, ray_tune):
    train_loader, fs_sup_loaders, fs_query_loader = get_few_shot_loaders(config, ray.get(train_data_ptr), ray.get(few_shot_data_ptr))
    logging.debug(f"support loaders: {len(fs_sup_loaders)}")
    logging.debug(f"few shot support loader batches: {len(fs_sup_loaders[0])}")
    logging.debug(f"few shot querry loader batches: {len(fs_query_loader)}")

    loss_func = get_emc_loss_function(args, config)
    num_of_classes = train_loader.unique_targets 
    image_channels = train_loader.channels
    image_size = train_loader.image_size
    model = emb_model.Convnet(device, config["lr"], config["d"],
                              num_of_classes, config["channels"], config["kernel_size"],
                              config["stride"], image_channels, image_size, config["linear_layers"],
                              config["linear_size"]).to(device)
    
    train_few_shot(config, train_loader, fs_sup_loaders, fs_query_loader, 
                   model, loss_func, device, ray_tune)
    
def setup_few_shot_pretrained(config, train_data, few_shot_data, device, args, ray_tune):
    train_loader, fs_sup_loaders, fs_query_loader = get_few_shot_loaders(config, ray.get(train_data), ray.get(few_shot_data))
    logging.debug(f"support loaders: {len(fs_sup_loaders)}")
    logging.debug(f"few shot support loader batches: {len(fs_sup_loaders[0])}")
    logging.debug(f"few shot querry loader batches: {len(fs_query_loader)}")
    loss_func = get_emc_loss_function(args, config)
    num_of_classes = len(train_loader.unique_targets)
    model, _ = load_pretrained(config["model_name"], num_of_classes, 
                            config["d"], train_loader.image_size, 
                            train_loader.channels, device, feature_extract=config["feature_extract"], train_layers=config["train_layers"])
    model.to(device)
   
    train_few_shot(config, train_loader, fs_sup_loaders, fs_query_loader, 
                model, loss_func, device, ray_tune)

def train_few_shot(config, train_loader, fs_sup_loaders, fs_query_load, 
                   model, loss_func, device, ray_tune):
    
    # logging.debug("extracting support images...")
    print("extracing support images...")
    support_images = extract_support_images(fs_sup_loaders)
    
    optimiser = optim.Adam(model.parameters(), lr=config["lr"])
    
    max_epochs = config["max_epochs"]

    snapshot_embeddings = []
    
    last_acc = 0
    for epoch in range(max_epochs):
        print("training...")
        train_emc(model, train_loader, optimiser, loss_func, max_epochs, epoch, device)
        last_acc = few_shot_eval(model, fs_sup_loaders, fs_query_load, support_images, device)
        if ray_tune:
            tune.report(accuracy = last_acc)
        else:
            print(f"Validation accuracy: {last_acc}")

        if not ray_tune:
            # TODO: make this faster using previous calculated support_images
            snapshot_embeddings.append(eu.get_few_shot_embedding_result(train_loader, fs_sup_loaders, fs_query_load, model, config, last_acc, device))
    
    if not ray_tune:
        print("==> saving embeddings..")
        ju.save_to_json('embeddingData', 'few_shot_test_data.json', snapshot_embeddings)

def extract_support_images(fs_sup_loaders):
    batches = []
    for i, loader in enumerate(fs_sup_loaders):
        logging.debug("extracting support for loader: %s", i)
        images, _ = next(iter(loader))
        batches.append(images)
        
    return batches

def few_shot_eval(model, support_loaders, query_loader, support_images, device):
    # Test the model
    model.eval()
    with torch.no_grad():
        print("evaluating")
        # Get the targets we have not seen before
        logging.debug("calling find few shot targets")
        few_shot_targets = find_few_shot_targets(support_loaders)
        logging.debug("done find few shot targets")
        num_of_new_classes = len(few_shot_targets)

        new_class_embeddings = []
        correct = [0] * num_of_new_classes
        total = [0] * num_of_new_classes

        logging.debug("calculating new embeddings...")
        new_class_embeddings = eu.get_few_shot_embeddings(support_images, model, device)
        
        # average embeddings for class
        new_class_embeddings = [sum(item) / len(item) for item in new_class_embeddings]

        # ensure lengths but 
        # assume the order is preserved
        assert len(new_class_embeddings) == len(few_shot_targets)

        # do evaluation
        print("evaluating predictions...")
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
        few_shot_targets.extend(loader.unique_targets)
    return few_shot_targets



""" {
    "train_embeddings", "train_labels", val_support_embeddings, val_support_labels, 
    val_query_embeddings, val_query_labels, "class_embeddings", "new_class_embeddings", 
    accuracy, config
}"""