from training_utils import classification_setup
import argparse
from loader.loader import load_data, get_data, get_fs_data, get_data_loader
from PTM.model_loader import load_pretrained
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
import ray
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
import datetime
from nn_util import simple_dist_loss
from few_shot_utils import train_few_shot
from training_utils import train, eval_classification

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

datasets = {"mnist": 0, 
            "omniglot": 1, 
            "cifar10": 2,
            "cifar100": 3,
            "cifarfs": 4}

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', dest="dataset", type=str, default="mnist", choices=datasets.keys(),
                        help="Determines the dataset on which training occurs. Choose between: ".format(datasets.keys()))
argparser.add_argument('--datadir', dest="data_dir", type=str, default="./data", help="Path to the data relative to current path")
argparser.add_argument('-fs', dest="few_shot", action="store_true", help="Few-shot flag")

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

def determine_device(ngpu):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if (device.type == 'cuda'):
        print('Using GPU')
    else:
        print('Using CPU')

def run_tune_fewshot(args):
    device = determine_device(ngpu=1)
    train_data, val_data, _ = get_fs_data(args)

    print("Training data size: ", len(train_data))
    print("Validation data size: ", len(val_data))

    # resources = {"cpu": args.cpu, "gpu": args.gpu}
    scheduler = AsyncHyperBandScheduler(grace_period=args.grace)
    reporter = tune.CLIReporter(
        metric_columns=["accuracy", "training_iteration"]
    )
    
    loss_func = simple_dist_loss

    smoke_test_space = {
            "lr": hp.uniform("lr", args.lr[0], args.lr[1]),
            "d": hp.uniformint("d", args.dims[0], args.dims[1]),
            "num_of_classes": args.num_of_classes,
            "channels": hp.choice("channels", args.cnn_channels),
            "batch_size": args.batch_size,
            "num_of_epochs": hp.uniformint("num_of_epochs", args.epochs[0], args.epochs[1])
        }
    
    good_start = {"num_of_epochs": 1,
                  "lr": 0.0005,
                  "d" : 60,
                  "channels" : 64,
                  "num_of_classes": 64,
                  "batch_size": 100,
                  "k_size": 4,
                  "stride": 1,
                  "linear_n": 1,
                  "linear_size": 64,
                  "shots": 5
                  }

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
        tune.with_parameters(classification_setup, train_data=train_data, test_data=val_data),
        tune_config=tuner_config,
        run_config=run_config
    )
    
    if (args.tuning):
        results = tuner.fit()
        print(results.get_best_result().metrics)
    else:
        # classification_setup(good_start, train_data, test_data, loss_func, device, ray_tune=False)
        #train_few_shot(good_start, train_data, val_data, loss_func, device, ray_tune=False)
        # train_few_shot(good_start, train_data, val_data, None, loss_func, device, ray_tune=False)
        setup_and_finetune(good_start, train_data, val_data, device)

def run_tune(args):
    device = determine_device(ngpu=1)
    train_data, test_data,  = get_data(args)

    print("Training data size: ", len(train_data))
    print("Test data size: ", len(test_data))

    resources = {"cpu": args.cpu, "gpu": args.gpu}
    scheduler = AsyncHyperBandScheduler(grace_period=args.grace)
    reporter = tune.CLIReporter(
        metric_columns=["accuracy", "training_iteration"]
    )
    
    loss_func = simple_dist_loss

    smoke_test_space = {
            "lr": hp.uniform("lr", args.lr[0], args.lr[1]),
            "d": hp.uniformint("d", args.dims[0], args.dims[1]),
            "num_of_classes": args.num_of_classes,
            "channels": hp.choice("channels", args.cnn_channels),
            "batch_size": args.batch_size,
            "num_of_epochs": hp.uniformint("num_of_epochs", args.epochs[0], args.epochs[1])
        }
    
    good_start = {"num_of_epochs": 1,
                  "lr": 0.0005,
                  "d" : 60,
                  "channels" : 64,
                  "num_of_classes": 64,
                  "batch_size": 100,
                  "k_size": 4,
                  "stride": 1,
                  "linear_n": 1,
                  "linear_size": 64,
                  }

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
        tune.with_parameters(classification_setup, train_data=train_data, test_data=test_data),
        tune_config=tuner_config,
        run_config=run_config
    )
    
    if (args.tuning):
        results = tuner.fit()
        print(results.get_best_result().metrics)
    else:
        #classification_setup(good_start, train_data, test_data, loss_func, device, ray_tune=False)
        #train_few_shot(good_start, train_data, test_data, test_data, loss_func, device, ray_tune=False)
        # classification_setup(good_start, train_data, test_data, loss_func, device, ray_tune=False)
        # train_few_shot(good_start, train_data, test_data, test_data, loss_func, device, ray_tune=False)
        setup_and_finetune(good_start, train_data, test_data, device)

def setup_and_finetune(config, train_data, test_data, device):
    train_loader = get_data_loader(train_data, batch_size=config["batch_size"])
    validation_loader = get_data_loader(test_data, batch_size=config["batch_size"])

    img_size = train_loader.image_size
    img_channels = train_loader.channels    
    
    model, _ = load_pretrained("resnet18", config["num_of_classes"], config["d"], img_size, img_channels, feature_extract=False)
    model.to(device)
    model.device = device
    optimiser = optim.Adam(model.parameters(), lr=config["lr"])

    loss_func = nn_util.simple_dist_loss
    max_epochs = config["num_of_epochs"]

    for epoch in range(max_epochs):
        print("training...")
        train(model, train_loader, optimiser, loss_func, max_epochs, epoch, device)
        # train(model, train_loader, optimiser, loss_func, max_epochs, current_epoch=epoch, device=device)
        # accuracy = eval(model, validation_loader, target_class_map, device=device)
        accuracy = eval_classification(model, validation_loader, device)
        print(accuracy)
        # tune.report(accuracy=accuracy)

if __name__ == '__main__':
    args = argparser.parse_args()
    if (not legal_args(args)):
        raise argparse.ArgumentError("Illegal config")

    if args.tuning:
        ray.init(num_cpus=args.cpu, num_gpus=args.gpu)

    print(args.dataset)

    if (args.few_shot):
        run_tune_fewshot(args)
    else:
        run_tune(args)

