import embedding_model as emb_model
import torch.optim
from loader.loader import get_data_loader
import torch
from torch import optim
import embedding_model as emb_model
from ray import tune



def train(model, train_loader, optimiser, loss_func, 
          num_epochs, current_epoch, device): 
    
    total_step = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        res = model(images)
        
        out_embds = res[:-model.num_of_classes]
        class_embds = res[model.num_of_classes:]
        
        loss = loss_func(out_embds, class_embds, device)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}' 
                .format(current_epoch + 1, num_epochs, i + 1, total_step, loss.item()))   

def eval_classification(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
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

def classification_setup(config, train_data, test_data, loss_func, device, ray_tune = True):
    train_loader = get_data_loader(train_data, config["batch_size"])
    test_loader = get_data_loader(test_data, config["batch_size"])

    img_size = train_loader.image_size
    channels = train_loader.channels
    
    model = emb_model.Convnet(device, config["lr"], 
                              config["d"], 
                              config["num_of_classes"], 
                              config["channels"],
                              config["k_size"], 
                              config["stride"],
                              channels, img_size, config["linear_n"], config["linear_size"]).to(device)
    
    optimiser = optim.Adam(model.parameters(), lr=model.lr)
    max_epochs = config["num_of_epochs"]

    for epoch in range(max_epochs):
        train(model, train_loader, optimiser, loss_func, max_epochs, current_epoch=epoch, device=device)
        accuracy = eval_classification(model, test_loader, device=device)
        
        if ray_tune:
            tune.report(accuracy=accuracy)