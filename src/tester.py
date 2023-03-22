import torch
from functools import partial
from torch import nn
from torch import optim
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import nn_util
import embedding_model as emb_model
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from datahandling_util import get_data, load_data, split_data, k_shot_loaders
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
import datetime
import json
import sys
from torch.utils.data import Dataset


ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
run_raytune = True


if device.type == 'cuda':
    print('Using GPU')
else:
    print('Using CPU')


def main():
    few_shot_targets = [7, 8, 9]
    num_of_classes = 10 - len(few_shot_targets)
    
    train_data, test_data = get_data()
    train_data, test_data, support_data = split_data(train_data, test_data, few_shot_targets)

    print("Training data size: ", len(train_data))
    print("Test data size: ", len(test_data))
    print("Support data size: ", len(support_data))

    resources = {"cpu": 1, "gpu": 0.5}
    scheduler = AsyncHyperBandScheduler(grace_period=3)
    reporter = tune.CLIReporter(
        metric_columns=["accuracy", "training_iteration"]
    )

    smoke_test_space = {
        "lr": hp.loguniform("lr", np.log(1e-9), np.log(1e-4)),
        "d": hp.uniformint("d", 10, 250),
        "num_of_classes": num_of_classes,
        "channels": hp.uniformint("channels", 10, 100),
        "batch_size": hp.choice("batch_size", [2, 4, 8, 16, 32, 64, 128, 256]),
        "num_of_epochs": hp.uniformint("num_of_epochs", 3, 30),
        "k_size": 4,
        "stride": 1,
        "linear_n" : hp.uniformint("linear_n", 1, 10),
        "linear_size" : hp.uniformint("linear_size", 2 ** 4, 2 ** 7),
        "shots": 5
    }
    
    good_start = {"num_of_epochs": 10,
                  "lr": 0.0005,
                  "d": 60,
                  "num_of_classes" : num_of_classes,
                  "channels": 64,
                  "batch_size" : 128,
                  "k_size": 4,
                  "stride": 1,
                  "linear_n" : 1,
                  "linear_size" : 128,
                  "shots": 5, 
                  }

    loss_func = nn_util.simple_dist_loss

    training_function = partial(setup_and_train,
                                train_data=train_data,
                                test_data=test_data,
                                support_data=support_data,
                                loss_func=loss_func)

    hyper_opt_search = HyperOptSearch(smoke_test_space,
                                      metric="accuracy",
                                      mode="max",
                                      points_to_evaluate=[good_start])

    tuner_config = tune.TuneConfig(
        metric="accuracy",
        mode="max",
        scheduler=scheduler,
        search_alg=hyper_opt_search,
        num_samples=250
    )

    run_config = air.RunConfig(
        name="mnist_initial_few_shot_test2",
        progress_reporter=reporter,
    )

    tuner = tune.Tuner(
        tune.with_resources(training_function, resources=resources),
        tune_config=tuner_config,
        run_config=run_config
    )

    #setup_and_train(good_start, train_data, test_data, support_data, loss_func)
    
    results = tuner.fit()
    print(results.get_best_result().metrics)
    #setup_and_train(good_start, train_data, test_data)

if __name__ == '__main__':
    main()
