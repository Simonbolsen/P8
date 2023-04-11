import logging
import sys

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

def get_pure_classification_embedding_result(train_loader, val_loader, model, config, accuracy, device):
    print("saving few shot embedding results")
    train_embeddings = []
    val_embeddings = []

    model.eval()

    train_labels = []
    val_labels = []

    class_embeds = []

    embeddings_dict = {}
    def hook(model, input, output):
        embeddings_dict["e"] = output.detach()
    model.avgpool.register_forward_hook(hook)

    for images, labels in train_loader:
        model(images.to(device))
        train_embeddings.extend(embeddings_dict["e"])
        train_labels.extend(labels.tolist())
        embeddings_dict = {}

    for images, labels in val_loader:
        model(images.to(device))
        val_embeddings.extend(embeddings_dict["e"])
        val_labels.extend(labels.tolist())
        embeddings_dict = {}

    results = {"train_embeddings": train_embeddings, "train_labels": train_labels, 
                  "val_embeddings": val_embeddings, "val_labels": val_labels,
                  "class_embeddings": class_embeds, "accuracy" : accuracy, "config" : config}

    return results
