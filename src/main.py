
import argparse
from loader.loader import load_data
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
from datahandling_util import get_data
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
import datetime
from tester import train

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

datasets = {"mnist": 0, "omniglot": 1}

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', dest="dataset", type=str, default="mnist", choices=datasets.keys(),
                        help="Determines the dataset on which training occurs. Choose between: ".format(datasets.keys()))


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
    loader = load_data(args)
    train_data = loader["train"]
    test_data = loader["test"]

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
                  "channels" : 64}

    training_function = partial(setup_and_train, 
                                loader=loader)
    
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


def setup_and_train(config, loader):
    model = emb_model.Convnet(device, lr = config["lr"], d = config["d"], num_of_classes=config["num_of_classes"], channels=config["channels"]).to(device)
    optimiser = optim.Adam(model.parameters(), lr=model.lr)
    loss_func = nn_util.simple_dist_loss
    target_class_map = { i:i for i in range(model.num_of_classes) }
    max_epochs = config["num_of_epochs"]

    for epoch in range(max_epochs):
        train(model, loader, optimiser, loss_func, max_epochs, current_epoch=epoch, device=device)
        accuracy = eval(model, loader, target_class_map, device=device)
        tune.report(accuracy=accuracy)


if __name__ == '__main__':
    args = argparser.parse_args()
    if (not legal_args(args)):
        raise argparse.ArgumentError("Illegal config")

    print(args.dataset)
    run_tune(args)
    print("Determines the dataset on which training occurs. Choose between: {}".format(", ".join(datasets)))

