import sys
import ray
import embedding_model as emb_model
import torch.optim
from loader.loader import get_data_loader
import torch
from torch import optim
import embedding_model as emb_model
from ray import tune
from PTM.model_loader import load_pretrained, load_resnet_pure
from nn_util import get_emc_loss_function, get_pure_loss_function
import embedding_util as eu
import logging
from timeit import default_timer as timer
from datetime import timedelta

def setup_and_finetune(config, train_data, test_data, device, ray_tune = True):
    train_loader = get_data_loader(train_data, batch_size=config["batch_size"])
    validation_loader = get_data_loader(test_data, batch_size=config["batch_size"])

    img_size = train_loader.image_size
    img_channels = train_loader.channels    
    
    model, _ = load_pretrained(config["model"], config["num_of_classes"], config["d"], img_size, img_channels, feature_extract=False)
    model.to(device)
    model.device = device
    optimiser = optim.Adam(model.parameters(), lr=config["lr"])

    loss_func = get_emc_loss_function(config)
    max_epochs = config["num_of_epochs"]

    for epoch in range(max_epochs):
        print("training...")
        train_emc(model, train_loader, optimiser, loss_func, max_epochs, epoch, device)
        accuracy = eval_classification(model, validation_loader, device)
        if ray_tune:
            tune.report(accuracy=accuracy)

def finetune_pretrained_pure(config, training_data_ptr, val_data_ptr, device, args, ray_tune):
    train_loader = get_data_loader(ray.get(training_data_ptr), config["batch_size"])
    val_loader = get_data_loader(ray.get(val_data_ptr), config["batch_size"])

    loss_func = get_pure_loss_function(args, config)
    num_of_classes, image_channels, image_size = get_loader_info(train_loader)

    model, _ = load_resnet_pure(config["model_name"], num_of_classes, image_size, 
                                image_channels, device, feature_extract=config["feature_extract"], 
                                train_layers=config["train_layers"])

    model.to(device)
    emc_classification_setup(config, model, train_loader, val_loader, loss_func, device, ray_tune)

def train_pure_pretrained(model, train_loader, optimiser, loss_func, num_epochs, current_epoch, device):
    total_step = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, train_loader.channels, train_loader.image_size, train_loader.image_size)
        images = images.to(device)
        labels = labels.type(torch.LongTensor).to(device)

        res = model(images)

        loss = loss_func(res, labels)
        optimiser.zero_grad()
    
        loss.backward()
            
        optimiser.step()

        if (i+1) % 100 == 0 or i+1 == total_step:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}' 
                .format(current_epoch + 1, num_epochs, i + 1, total_step, loss.item()))

def train_emc(model, train_loader, optimiser, loss_func, 
          num_epochs, current_epoch, device): 
    embeds_map = { v.item() : i for i, v in enumerate(train_loader.unique_targets) }
    total_step = len(train_loader)
    
    start = timer()

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        res = model(images)
        
        out_embds = res[:-model.num_of_classes]
        class_embds = res[-model.num_of_classes:]
        
        assert len(class_embds) == model.num_of_classes

        loss, grad = loss_func(out_embds, class_embds, [embeds_map[v.item()] for v in labels], device)
        optimiser.zero_grad()

        if grad is None:
            # logging.debug("using torch autograd")
            loss.backward()
        else:
            # logging.debug("using manual grad")
            res.backward(gradient = grad)
            
        optimiser.step()

        if (i+1) % 100 == 0 or i+1 == total_step:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}' 
                .format(current_epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            
    end = timer()
    logging.debug(f"training estimated wall time: {timedelta(seconds=end - start)}")


def find_closest_embedding(query, class_embeddings):
    smallest_sqr_dist = sys.maxsize
    closest_target_index = 0
    for i, embedding in enumerate(class_embeddings):
        squared_dist = (embedding - query).pow(2).sum(0)
        if squared_dist < smallest_sqr_dist:
            smallest_sqr_dist = squared_dist
            closest_target_index = i

    return closest_target_index   

def eval_classification(model, val_loader, device):
    start = timer()
    index_to_target = { i : v.item() for i, v in enumerate(val_loader.unique_targets) }
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            test_output = model(images)
            
            class_embeddings = test_output[-model.num_of_classes:]
            image_embeds = test_output[:-model.num_of_classes]
            
            distances = torch.cdist(image_embeds, class_embeddings)
            closest_target_indexs = torch.min(distances, 1).indices
            
            for i, closest_target_index in enumerate(closest_target_indexs):
                predicted_target = index_to_target[closest_target_index.item()]
                
                if predicted_target == labels[i].item():
                    correct += 1
                total += 1
    
            # for i, output_embedding in enumerate(image_embeds):
            #     closest_target_index = find_closest_embedding(output_embedding, class_embeddings)
            #     # predicted_target = val_loader.unique_targets[closest_target_index]
            #     predicted_target = index_to_target[closest_target_index]
                
            #     if predicted_target == labels[i].item():
            #         correct += 1
            #     total += 1
        end = timer()
        logging.debug(f"evaluation estimated wall time: {timedelta(seconds=end - start)}")
        return correct / total

def eval_ohe_classification(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            for i, pred_label in enumerate(output):
                if torch.argmax(pred_label).item() == labels[i].item():
                    correct += 1
                total += 1
        return correct / total

def setup_classification_custom_model(config, training_data_ptr, val_data_ptr, device, args, ray_tune):
    train_loader = get_data_loader(ray.get(training_data_ptr), config["batch_size"])
    val_loader = get_data_loader(ray.get(val_data_ptr), config["batch_size"])

    loss_func = get_emc_loss_function(args, config)
    num_of_classes, image_channels, image_size = get_loader_info(train_loader)
    
    model = emb_model.Convnet(device, config["lr"], config["d"],
                              num_of_classes, config["channels"], config["kernel_size"],
                              config["stride"], image_channels, image_size, config["linear_layers"],
                              config["linear_size"]).to(device)
    emc_classification_setup(config, model, train_loader, val_loader, loss_func, device, ray_tune)  


def setup_emc_classification_pretrained(config, training_data_ptr, val_data_ptr, device, args, ray_tune):
    train_loader = get_data_loader(ray.get(training_data_ptr), config["batch_size"])
    val_loader = get_data_loader(ray.get(val_data_ptr), config["batch_size"])

    loss_func = get_emc_loss_function(args, config)
    num_of_classes, image_channels, image_size = get_loader_info(train_loader)

    model, _ = load_pretrained(config["model_name"], num_of_classes, 
                            config["d"], image_size, 
                            image_channels, device, feature_extract=config["feature_extract"], 
                            train_layers=config["train_layers"])
    model.to(device)
    emc_classification_setup(config, model, train_loader, val_loader, loss_func, device, ray_tune)

def setup_pure_classification_pretrained(config, training_data_ptr, val_data_ptr, device, args, ray_tune):
    train_loader = get_data_loader(ray.get(training_data_ptr), config["batch_size"])
    val_loader = get_data_loader(ray.get(val_data_ptr), config["batch_size"])

    loss_func = get_pure_loss_function(args, config)
    num_of_classes, image_channels, image_size = get_loader_info(train_loader)

    model, _ = load_resnet_pure(config["model_name"], num_of_classes, image_size, 
                            image_channels, device, feature_extract=config["feature_extract"], train_layers=config["train_layers"])
    model.to(device)
    pure_classification_setup(config, model, train_loader, val_loader, loss_func, device, ray_tune)

def get_loader_info(train_loader):
    num_of_classes = train_loader.unique_targets.size()[0]
    image_channels = train_loader.channels
    image_size = train_loader.image_size
    return num_of_classes,image_channels,image_size


def emc_classification_setup(config, model, train_loader, val_loader, loss_func, device, ray_tune = True):
    optimiser = optim.Adam(model.parameters(), lr=config["lr"])
    max_epochs = config["max_epochs"]

    print("start training classification...")
    for epoch in range(max_epochs):
        train_emc(model, train_loader, optimiser, loss_func, max_epochs, current_epoch=epoch, device=device)
        print("evaluating...")
        accuracy = classifiers["nearest_neighbour"](model, val_loader, device=device)
        
        if ray_tune:
            tune.report(accuracy=accuracy)
        else: 
            print(f"accuracy: {accuracy}")

def pure_classification_setup(config, model, train_loader, val_loader, loss_func, device, ray_tune = True):
    optimiser = optim.Adam(model.parameters(), lr=config["lr"])
    max_epochs = config["max_epochs"]
    accuracies = []

    if not ray_tune:
        eu.make_embedding_data_folder(config)
        print("Save embeddings: True")

    print("start training classification...")
    for epoch in range(max_epochs):
        train_pure_pretrained(model, train_loader, optimiser, loss_func, max_epochs, current_epoch=epoch, device=device)
        print("evaluating...")
        accuracy = classifiers["one_hot_encoding"](model, val_loader, device=device)
        accuracies.append(accuracy)
        
        if ray_tune:
            tune.report(accuracy=accuracy)
        else: 
            print(f"accuracy: {accuracy}")
            eu.save_pure_classification_embedding_result(train_loader, val_loader, model, config, accuracy, epoch, device)
    
    if not ray_tune:
        eu.save_embedding_meta_data(config, accuracies)



classifiers = {
    "one_hot_encoding": eval_ohe_classification,
    "nearest_neighbour": eval_classification
}