
import argparse
from PTM.model_loader import load_pretrained
from loader.loader import load_data, get_data, get_data_loader
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
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
import datetime


def gtzero_int(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError("Minimum value is 1")
    return x

def gtzero_intlist(x):
    for v in x:
        if v < 1:
            raise argparse.ArgumentTypeError("Minimum value is 1")
    return x

def gtzero_float(x):
    x = float(x)
    if x <= 0:
        raise argparse.ArgumentTypeError("Minimum value is >0")
    return x


ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

if (device.type == 'cuda'):
    print('Using GPU')
else:
    print('Using CPU')

datasets = {"mnist": 0, 
            "omniglot": 1, 
            "cifar10": 2,
            "cifar100": 3,
            "cifarfs": 4}

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', dest="dataset", type=str, default="mnist", choices=datasets.keys(),
                        help="Determines the dataset on which training occurs. Choose between: ".format(datasets.keys()))
argparser.add_argument('--datadir', dest="data_dir", type=str, default="./data", help="Path to the data relative to current path")

# Training arguments
argparser.add_argument('--epochs', dest="epochs", nargs="+", type=gtzero_int, default=[5,30], help="Epochs must be > 0. Can be multiple values")
argparser.add_argument('--classes', dest="num_of_classes", type=gtzero_int, help="Number of unique classes for the dataset")
argparser.add_argument('--batch', dest="batch_size", type=gtzero_int, default=100, help="Batch size must be > 0")

argparser.add_argument('--channels', dest="cnn_channels", nargs="+", type=gtzero_int, default=[16, 32, 64, 128, 256], help="Number of channels in each convolutional layer")
argparser.add_argument('--layers', dest="cnn_layers", type=gtzero_int, default=5, help="Number of convolutional layers")

# Optimiser arguments
argparser.add_argument('--lr', dest="lr", nargs="+", type=gtzero_float, default=[0.00001, 0.0001], help="One or more learning rates")
argparser.add_argument('--dims', dest="dims", nargs="+", type=gtzero_int, default=[10, 100], help="One or more embedding dimensions")

# Raytune arguments
argparser.add_argument('--gpu', dest="gpu", type=gtzero_float, default=0.25, help="GPU resources")
argparser.add_argument('--cpu', dest="cpu", type=gtzero_float, default=3, help="CPU resources")
argparser.add_argument('--grace', dest="grace", type=gtzero_int, default=4, help="Grace period before early stopping")
argparser.add_argument('-t', dest="tuning", action="store_true", help="Tuning flag")

def legal_args(args):
    if (args.tuning):
        return len(args.dims) > 1 and len(args.lr) > 1 and len(args.epochs) > 1 and (len(args.cnn_channels) == args.cnn_layers)
    return True

def run_tune(args):
    train_data, test_data = get_data(args)

    print("Training data size: ", len(train_data))
    print("Test data size: ", len(test_data))

    resources = {"cpu": args.cpu, "gpu": args.gpu}
    scheduler = AsyncHyperBandScheduler(grace_period=args.grace)
    reporter = tune.CLIReporter(
        metric_columns=["accuracy", "training_iteration"]
    )

    smoke_test_space = {
            "lr": hp.uniform("lr", args.lr[0], args.lr[1]),
            "d": hp.uniformint("d", args.dims[0], args.dims[1]),
            "num_of_classes": args.num_of_classes,
            "channels": hp.choice("channels", args.cnn_channels),
            "batch_size": args.batch_size,
            "num_of_epochs": hp.uniformint("num_of_epochs", args.epochs[0], args.epochs[1])
        }
    
    good_start = {"num_of_epochs": 10,
                  "lr": 0.0005,
                  "d" : 60,
                  "channels" : 64,
                  "num_of_classes": 964,
                  "batch_size": 100,
                  }

    training_function = partial(setup_and_train)
    
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
        tune.with_parameters(training_function, train_data=train_data, test_data=test_data),
        tune_config=tuner_config,
        run_config=run_config
    )
    
    if (args.tuning):
        results = tuner.fit()
        print(results.get_best_result().metrics)
    else:
        setup_and_train(good_start, train_data, test_data)


def setup_and_train(config, train_data=None, test_data=None):
    train_loader = get_data_loader(train_data, batch_size=config["batch_size"])
    validation_loader = get_data_loader(test_data, batch_size=config["batch_size"])
    model = emb_model.Convnet(device, lr = config["lr"], d = config["d"], num_of_classes=config["num_of_classes"], 
                              channels=config["channels"], image_size=train_loader.image_size, image_channels=train_loader.channels).to(device)
    optimiser = optim.Adam(model.parameters(), lr=model.lr)
    loss_func = nn_util.dist_and_proximity_loss(500)
    target_class_map = { i:i for i in range(config["num_of_classes"]) }
    max_epochs = config["num_of_epochs"]

    for epoch in range(max_epochs):
        train(model, train_loader, optimiser, loss_func, max_epochs, current_epoch=epoch, device=device)
        accuracy = eval(model, validation_loader, target_class_map, device=device)
        tune.report(accuracy=accuracy)

def setup_and_finetune(config, train_data=None, test_data=None):
    train_loader = get_data_loader(train_data, batch_size=config["batch_size"])
    validation_loader = get_data_loader(test_data, batch_size=config["batch_size"])
    model, input_size = load_pretrained("resnet18", config["num_of_classes"], config["d"], feature_extract=False)
    #model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.to(device)
    model.device = device
    optimiser = optim.Adam(model.parameters(), lr=config["lr"])
    loss_func = nn_util.dist_and_proximity_loss(500)
    target_class_map = { i:i for i in range(config["num_of_classes"]) }
    max_epochs = config["num_of_epochs"]

    for epoch in range(max_epochs):
        train(model, train_loader, optimiser, loss_func, max_epochs, current_epoch=epoch, device=device)
        accuracy = eval(model, validation_loader, target_class_map, device=device)
        tune.report(accuracy=accuracy)

def train(model, loader, optimiser, loss_func, num_epochs, current_epoch, device): 
    total_step = len(loader)

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        res = model(images)
        # loss, loss_div = loss_func(res, labels, model.num_of_classes, { i:i for i in range(model.num_of_classes) }, device)
        loss = loss_func(res, labels, model.num_of_classes, { i:i for i in range(model.num_of_classes) }, device)
        optimiser.zero_grad()
        # res.backward(gradient = loss_div)
        loss.backward()
        optimiser.step()    
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}' 
                .format(current_epoch + 1, num_epochs, i + 1, total_step, loss.item()))   

def eval(model, loader, target_class_map, device):
     # Test the model
    model.eval()    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
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
    args = argparser.parse_args()
    if (not legal_args(args)):
        raise argparse.ArgumentError("Illegal config")

    print(args.dataset)
    run_tune(args)
    print("Determines the dataset on which training occurs. Choose between: {}".format(", ".join(datasets)))

