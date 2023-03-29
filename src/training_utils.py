import ray
import embedding_model as emb_model
import torch.optim
from loader.loader import get_data_loader
import torch
from torch import optim
import embedding_model as emb_model
from ray import tune
from PTM.model_loader import load_pretrained
from nn_util import get_loss_function

def setup_and_finetune(config, train_data, test_data, device, ray_tune = True):
    train_loader = get_data_loader(train_data, batch_size=config["batch_size"])
    validation_loader = get_data_loader(test_data, batch_size=config["batch_size"])

    img_size = train_loader.image_size
    img_channels = train_loader.channels    
    
    model, _ = load_pretrained(config["model"], config["num_of_classes"], config["d"], img_size, img_channels, feature_extract=False)
    model.to(device)
    model.device = device
    optimiser = optim.Adam(model.parameters(), lr=config["lr"])

    loss_func = get_loss_function(config)
    max_epochs = config["num_of_epochs"]

    for epoch in range(max_epochs):
        print("training...")
        train(model, train_loader, optimiser, loss_func, max_epochs, epoch, device)
        accuracy = eval_classification(model, validation_loader, device)
        if ray_tune:
            tune.report(accuracy=accuracy)

def train(model, train_loader, optimiser, loss_func, 
          num_epochs, current_epoch, device): 
    embeds_map = { v.item() : i for i, v in enumerate(train_loader.unique_targets) }
    total_step = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        res = model(images)
        
        out_embds = res[:-model.num_of_classes]
        class_embds = res[-model.num_of_classes:]
        
        assert len(class_embds) == model.num_of_classes
        
        loss = loss_func(out_embds, class_embds, [embeds_map[v.item()] for v in labels], device)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (i+1) % 100 == 0 or i+1 == total_step:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}' 
                .format(current_epoch + 1, num_epochs, i + 1, total_step, loss.item()))   

def eval_classification(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            test_output = model(images)
    
            # Smukt
            for i, output_embedding in enumerate(test_output[:-model.num_of_classes]):
                smallest_sqr_dist = 100000000
                smallest_k = 0
                for k in range(model.num_of_classes):
                    actual_class_embedding = test_output[k - model.num_of_classes]
                    squared_dist = (actual_class_embedding - output_embedding).pow(2).sum(0)

                    if squared_dist < smallest_sqr_dist:
                        smallest_sqr_dist = squared_dist
                        smallest_k = k

                if smallest_k == labels[i].item():
                    correct += 1
                total += 1
        return correct / total

def setup_classification_custom_model(config, training_data_ptr, val_data_ptr, device, args, ray_tune):
    train_loader = get_data_loader(ray.get(training_data_ptr), config["batch_size"])
    val_loader = get_data_loader(ray.get(val_data_ptr), config["batch_size"])

    loss_func = get_loss_function(args, config)
    num_of_classes, image_channels, image_size = get_loader_info(train_loader)
    
    model = emb_model.Convnet(device, config["lr"], config["d"],
                              num_of_classes, config["channels"], config["kernel_size"],
                              config["stride"], image_channels, image_size, config["linear_layers"],
                              config["linear_size"]).to(device)
    classification_setup(config, model, train_loader, val_loader, loss_func, device, ray_tune)  


def setup_classification_pretrained(config, training_data_ptr, val_data_ptr, device, args, ray_tune):
    train_loader = get_data_loader(ray.get(training_data_ptr), config["batch_size"])
    val_loader = get_data_loader(ray.get(val_data_ptr), config["batch_size"])

    loss_func = get_loss_function(args, config)
    num_of_classes, image_channels, image_size = get_loader_info(train_loader)

    model, _ = load_pretrained(config["model_name"], num_of_classes, 
                            config["d"], image_size, 
                            image_channels, device, train_layers=config["train_layers"])
    
    classification_setup(config, model, train_loader, val_loader, loss_func, device, ray_tune)

def get_loader_info(train_loader):
    num_of_classes = train_loader.unique_targets 
    image_channels = train_loader.channels
    image_size = train_loader.image_size
    return num_of_classes,image_channels,image_size


def classification_setup(config, model, train_loader, val_loader, loss_func, device, ray_tune = True):
    optimiser = optim.Adam(model.parameters(), lr=config["lr"])
    max_epochs = config["max_epochs"]

    print("start training classification...")
    for epoch in range(max_epochs):
        train(model, train_loader, optimiser, loss_func, max_epochs, current_epoch=epoch, device=device)
        print("evaluating...")
        accuracy = eval_classification(model, val_loader, device=device)
        
        if ray_tune:
            tune.report(accuracy=accuracy)
        else: 
            print(f"accuracy: {accuracy}")