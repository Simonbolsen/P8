import logging
import sys
import file_util as fu
import torch
import os

def extract_support_images(fs_sup_loaders):
    batches = []
    for i, loader in enumerate(fs_sup_loaders):
        logging.debug("extracting support for loader: %s", i)
        images, _ = next(iter(loader))
        batches.append(images)
        
    return batches

def get_few_shot_embeddings(support_images, model, device):
    new_class_embeddings = []
    
    image_size = support_images[0].size()[2]
    channels = support_images[0].size()[1]
    
    logging.debug("image_size: ", image_size)
    logging.debug("channels ", channels)
        
    for support_batch in support_images:
        images = support_batch.view(-1, channels, image_size, image_size).float().to(device)
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


def get_few_shot_embedding_result(train_loader, support_loaders, query_loader, model, config, accuracy, device):
    print("saving few shot embedding results")
    train_embeddings = []
    val_support_embeddings = []
    val_query_embeddings = []

    model.eval()

    train_labels = []
    val_support_labels = []
    val_query_labels = []

    class_embeds = []
    first_it = True

    for images, labels in train_loader: 
        if (first_it):
            first_it = False
            class_embeds = model(images.to(device)).tolist()[-len(train_loader.unique_targets):]
        train_embeddings.extend(model(images.to(device)).tolist())
        train_labels.extend(labels.tolist())
    
    for support_loader in support_loaders:
        for images, labels in support_loader: 
            val_support_embeddings.extend(model(images.to(device)).tolist())
            val_support_labels.extend(labels.tolist())

    for images, labels in query_loader:
        val_query_embeddings.extend(model(images.to(device)).tolist())
        val_query_labels.extend(labels.tolist())

    new_class_embeds = []
    extracted_images = extract_support_images(support_loaders)
    few_shot_embeds = get_few_shot_embeddings(extracted_images, model, device)
    few_shot_embeds = [sum(item) / len(item) for item in few_shot_embeds]
    for few_shot_embed in few_shot_embeds:
        new_class_embeds.append(few_shot_embed.tolist())

    embeddings = {"train_embeddings": train_embeddings, "train_labels": train_labels, 
                  "val_support_embeddings": val_support_embeddings, "val_support_labels": val_support_labels,
                  "val_query_embeddings": val_query_embeddings, "val_query_labels": val_query_labels,
                  "new_class_embeddings": new_class_embeds,
                  "class_embeddings": class_embeds, "accuracy" : accuracy, "config" : config}

    return embeddings

def save_pure_classification_embedding_result(train_loader, val_loader, model, config, accuracy, epoch, device):
    print("Extracting embedding results")

    model.eval()


    embeddings_dict = {}
    def hook(model, input, output):
        embeddings_dict["e"] = torch.squeeze(output.detach()).tolist()
    model.avgpool.register_forward_hook(hook)

    save_pure_classification_embeddings("train", train_loader, model, config, accuracy, epoch, embeddings_dict, device)
    save_pure_classification_embeddings("val", val_loader, model, config, accuracy, epoch, embeddings_dict, device)    

def make_embedding_data_folder(config):
    data_folder = os.path.join(os.path.realpath(__file__), '..', 'embeddingData', config.exp_name)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    else:
        logging.ERROR("Cannot save embeddings as folder already exists: " + data_folder)
        sys.exit()

def save_embedding_meta_data(config, accuracies):
    data = {'config': config, 'accuracies': accuracies}
    path = os.path.join('embeddingData', config.exp_name)
    fu.save_as_json(path, 'meta_data.json', data)

def save_pure_classification_embeddings(prefix, loader, model, config, accuracy, epoch, embeddings_dict, device):
    embeddings = []
    all_labels = []
    predictions = []
    for images, labels in loader:
        predictions.extend(model(images.to(device)).tolist())
        embeddings.extend(embeddings_dict["e"])
        all_labels.extend(labels.tolist())

    results = {prefix + "_embeddings": embeddings, prefix + "_labels": all_labels,
                  "predictions": predictions}
    
    print(f"Saving {prefix}: {len(embeddings)} {len(embeddings[0])}")
    fu.save_to_pickle(os.path.join('embeddingData', config.exp_name), f'classification_data_{prefix}_{epoch}.p', results)
