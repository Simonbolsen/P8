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
import Plotting.plotting_util as plot
import math
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from datahandling_util import get_data, load_data
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
import datetime

ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

if (device.type == 'cuda'):
    print('Using GPU')
else:
    print('Using CPU')

def main():
    train_data, test_data = get_data()

    print("Training data size: ", len(train_data))
    print("Test data size: ", len(test_data))

    resources = {"cpu": 4, "gpu": 0.25}
    scheduler = AsyncHyperBandScheduler(grace_period=3)
    reporter = tune.CLIReporter(
        metric_columns=["accuracy", "training_iteration"]
    )

    smoke_test_space = {
            "lr": hp.uniform("lr", 0.00001, 0.01),
            "d": hp.uniformint("d", 16, 256),
            "num_of_classes": 10,
            "channels": hp.uniformint("channels", 16, 256),
            "batch_size": 100,
            "num_of_epochs": hp.uniformint("num_of_epochs", 5, 30)
        }
    
    good_start = {"num_of_epochs": 10,
                  "lr": 0.0005,
                  "d" : 60,
                  "channels" : 64}

    training_function = partial(setup_and_train, 
                                train_data=train_data, 
                                test_data=test_data)
    
    hyper_opt_search = HyperOptSearch(smoke_test_space, 
                                      metric="accuracy", 
                                      mode="max", 
                                    #   n_initial_points=2, 
                                      points_to_evaluate=[good_start])

    tuner_config = tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=scheduler,
            search_alg=hyper_opt_search,
            num_samples=1000
    )

    run_config = air.RunConfig(
            name="mnist_initial_test",
            progress_reporter=reporter,
            # stop={"training_iteration": 10}
    )

    tuner = tune.Tuner(
        tune.with_resources(training_function, resources=resources),
        tune_config=tuner_config,
        run_config=run_config
    )

    results = tuner.fit()
    print(results.get_best_result().metrics)

    # lrs = []
    # dims = []
    # accuracies = []
    # epochs = []

    # for result in results:
    #     lrs.append(result.config["lr"])
    #     dims.append(result.config["d"])
    #     epochs.append(result.metrics["training_iteration"])
    #     accuracies.append(result.metrics["accuracy"])

    # print(lrs)
    # print(dims)
    # print(accuracies)
    # print(epochs)

    # visualize_hyperparameters(lrs, dims, "learning_rate", "dimensions", accuracies)

# Probably not needed
def visualize_hyperparameters(param1_axis, param2_axis, param1_name, param2_name, results, 
                            visualization_func = lambda x : 1 - math.sqrt(1 - x**2)):
    visual_results = [[visualization_func(i) for i in r] for r in results]
    # visual_results = [[visualization_func(r)] for r in results]
    plot.plotSurface([visual_results], 
                     "Accuracy", 
                     param1_axis, 
                     param1_name, 
                     param2_axis, 
                     param2_name, 
                     surfaceLabels=["Accuracy"], 
                     num_of_surfaces=1)

# Probably not needed
def two_param_experiment(config_func, labels, param1_axis, param2_axis, loaders):
    results = []

    for i, p1 in enumerate(param1_axis):
        results.append([])
        for ii, p2 in enumerate(param2_axis):
            print(f'\n{labels[0]}: {p1}, {labels[1]}: {p2}, run {i * len(param1_axis) + ii}/{len(param1_axis) * len(param2_axis)}')
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
    max_epochs = config["num_of_epochs"]

    for epoch in range(max_epochs):
        train(model, loaders, optimiser, loss_func, max_epochs, current_epoch=epoch, device=device)
        accuracy = eval(model, loaders, target_class_map, device=device)
        tune.report(accuracy=accuracy)
        

    # train(model, config["num_of_epochs"], loaders, optimiser, loss_func)

    # results[-1].append(accuracy)
    # print(f'Test Accuracy of the model on the 10000 test images: {(accuracy * 100):.2f}%')   

def train(model, loaders, optimiser, loss_func, num_epochs, current_epoch, device): 
    total_step = len(loaders['train'])

    for i, (images, labels) in enumerate(loaders["train"]):
        images = images.to(device)
        labels = labels.to(device)

        res = model(images)
        loss, loss_div = loss_func(res, labels, model.num_of_classes, { i:i for i in range(model.num_of_classes) }, device)
        optimiser.zero_grad()
        res.backward(gradient = loss_div)
        optimiser.step()    
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}' 
                .format(current_epoch + 1, num_epochs, i + 1, total_step, loss.item()))   

def eval(model, loaders, target_class_map, device):
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
    

def few_shot_eval(model, loaders):
     # Test the model
    model.eval()    
    with torch.no_grad():
        num_of_new_classes = len(loaders)
        new_class_embeddings = []
        correct = []
        total = []

        # create average embeddings
        for loader in loaders:
            for images, labels in loader: #TODO proper loader
                images = images.to(device)
                labels = labels.to(device)

                few_shot_output = model(images)
            
                new_class_embeddings.append(few_shot_output[:-model.num_of_classes])
                break

        new_class_embeddings = [sum(item) / len(item) for item in new_class_embeddings]    

        # do evaluation
        for i, loader in enumerate(loaders):
            for images, labels in loader[1:]:
                images = images.to(device)
                labels = labels.to(device)

                test_output = model(images)

                for output_embedding in test_output[:-model.num_of_classes]:
                    smallest_sqr_dist = 100000000
                    smallest_k = 0
                    for k in range(num_of_new_classes):
                        actual_class_embedding = new_class_embeddings[k]
                        squared_dist = (actual_class_embedding - output_embedding).pow(2).sum(0)
                        
                        if squared_dist < smallest_sqr_dist:
                            smallest_sqr_dist = squared_dist
                            smallest_k = k

                    if smallest_k == labels[0].item():
                        correct[i] += 1
                    total[i] += 1
        
        return correct, total

if __name__ == '__main__':
    main()