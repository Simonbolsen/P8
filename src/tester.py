import torch
from functools import partial
from torch import nn
from torch import optim
import os
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import nn_util
import embedding_model as emb_model
import plotting_util as plot
import math
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def get_data(dir="./data"):
    train_data = datasets.MNIST(
        root = dir,
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )

    test_data = datasets.MNIST(
        root = dir, 
        train = False, 
        transform = ToTensor()
    )

    return train_data, test_data

def load_data(train_data, test_data, batch_size=100):
    loaders = {
        "train": torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1),
        "test": torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    }

    return loaders

def main():
    if (device.type == 'cuda'):
        print('Using GPU')
    else:
        print('Using CPU')

    train_data, test_data = get_data()
    loaders = load_data(train_data, test_data)

    print("Training data size: ", len(train_data))
    print("Test data size: ", len(test_data))

    resources = {"cpu": 4, "gpu": 1}
    scheduler = AsyncHyperBandScheduler()

    smoke_test_space = {
            "lr": 0.0001,
            "d": 10,
            "num_of_classes": 10,
            "channels": 10,
            "batch_size": 100,
            "num_of_epochs": 1
        }

    tuner = tune.Tuner(
        tune.with_resources(partial(setup_and_train, train_data=train_data, test_data=test_data), resources=resources),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=scheduler,
            # num_samples=1
        ),
        run_config=air.RunConfig(
            name="test",
            stop={"training_iteration": 10}
        ),
        # param_space={
        #     "lr": tune.grid_search([0.0001, 0.0002, 0.0003]),
        #     "d": tune.grid_search([10, 15, 20]),
        #     "num_of_classes": 10,
        #     "channels": 64,
        #     "batch_size": tune.choice([25, 50, 100]),
        #     "num_of_epochs": 1
        # }
        param_space=smoke_test_space
    )

    results = tuner.fit()
    for i in range(len(results)):
        result = results[i]
        print(result.metrics)

    # param1_func = lambda p1 : 0.0002 * p1 + 0.0001
    # param2_func = lambda p2 : p2 + 3
    
    # param1_num = 5
    # param2_num = 30

    # labels = ["Learning Rate lr",  "Dimensions d"]

    # param1_axis = [param1_func(i) for i in range(param1_num)]
    # param2_axis = [param2_func(i) for i in range(param2_num)]

    # config_func = lambda p1, p2 : {"lr":p1, "d":15, "num_of_classes":10, "channels":64, "num_of_epochs":p2}

    # results = two_param_experiment(config_func, labels, 
    #                                             param1_axis, param2_axis, loaders)
    
    # for i in range(param1_num):
    #     for ii in range(param2_num):
    #         results[i][ii] = math.exp(results[i][ii])

    # plot.plotSurface([results], "Accuracy", param1_axis, labels[0], param2_axis, labels[1], surfaceLabels=["Accuracy"], num_of_surfaces=1)

def two_param_experiment(config_func, labels, param1_axis, param2_axis, loaders):
    results = []

    for p1 in param1_axis:
        results.append([])
        for p2 in param2_axis:
            print(f'{labels[0]}: {p1}, {labels[1]}: {p2}')
            setup_and_train(config_func, loaders, results, p1, p2)
    return results

# def setup_and_train(config_func, loaders, results, p1, p2):
def setup_and_train(config, train_data, test_data):
    # config = config_func(p1, p2)
    # train_data, test_data = get_data()
    loaders = load_data(train_data, test_data, config["batch_size"])
    model = emb_model.Convnet(device, lr = config["lr"], d = config["d"], num_of_classes=config["num_of_classes"], channels=config["channels"]).to(device)
    optimiser = optim.Adam(model.parameters(), lr=model.lr)
    loss_func = nn_util.simple_dist_loss
    target_class_map = { i:i for i in range(model.num_of_classes) }

    for epoch in range(config["num_of_epochs"]):
        train(model, loaders, optimiser, loss_func)
    # train(model, config["num_of_epochs"], loaders, optimiser, loss_func)

    accuracy = eval(model, loaders, target_class_map)
    tune.report(accuracy=accuracy)
    # results[-1].append(accuracy)
    # print(f'Test Accuracy of the model on the 10000 test images: {(accuracy * 100):.2f}%')   

def train(model, loaders, optimiser, loss_func): 
    # total_step = len(loaders['train'])
    # for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(loaders["train"]):
        images = images.to(device)
        labels = labels.to(device)

        res = model(images)
        loss, loss_div = loss_func(res, labels, model.num_of_classes, { i:i for i in range(model.num_of_classes) }, device)
        optimiser.zero_grad()
        res.backward(gradient = loss_div)
        optimiser.step()
            # if (i+1) % 100 == 0:
            #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}' 
            #            .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))               
            #     pass

def eval(model, loaders, target_class_map):
     # Test the model
    model.eval()    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            images = images.to(device)
            labels = labels.to(device)

            test_output = model(images)

            for i, output_embedding in enumerate(test_output[:-model.num_of_classes]):
                smallest_sqr_dist = 100000000
                smallest_k = 0
                for k in range(model.num_of_classes):
                    actual_class_embedding = test_output[k - model.num_of_classes]
                    squared_dist = (actual_class_embedding - output_embedding).pow(2).sum(0)
                    
                    if squared_dist < smallest_sqr_dist:
                        smallest_sqr_dist = squared_dist
                        smallest_k = k

                if smallest_k == target_class_map[labels[i].item()]:
                    correct += 1
                total += 1
    
    return correct / total

if __name__ == '__main__':
    main()