import torch
from functools import partial
from torch import nn
from torch import optim
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import nn_util
import embedding_model as emb_model
import plotting_util as plot
import math
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from datahandling_util import get_data, load_data, split_data, k_shot_loaders
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

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

    resources = {"cpu": 2, "gpu": 1}
    scheduler = AsyncHyperBandScheduler(grace_period=2, reduction_factor=2)
    reporter = tune.CLIReporter(
        metric_columns=["accuracy", "training_iteration"]
    )

    smoke_test_space = {
        "lr": hp.uniform("lr", 1e-5, 1e-1),
        "d": hp.uniformint("d", 10, 100),
        "num_of_classes": num_of_classes,
        "channels": hp.uniformint("channels", 10, 100),
        "batch_size": 100,
        "num_of_epochs": hp.uniformint("num_of_epochs", 5, 10),
        "shots": hp.choice("shots", [5, 10, 20]),
    }

    training_function = partial(setup_and_train,
                                train_data=train_data,
                                test_data=test_data,
                                support_data=support_data)

    hyper_opt_search = HyperOptSearch(smoke_test_space,
                                      metric="accuracy",
                                      mode="max",
                                      n_initial_points=2,
                                      points_to_evaluate=[{"num_of_epochs": 1,
                                                           "lr": 0.0005,
                                                           "d": 60,
                                                           "channels": 64,
                                                           "shots": 5, }
                                                          ])

    tuner_config = tune.TuneConfig(
        metric="accuracy",
        mode="max",
        scheduler=scheduler,
        search_alg=hyper_opt_search,
        num_samples=10
    )

    run_config = air.RunConfig(
        name="test",
        progress_reporter=reporter,
        # stop={"training_iteration": 10}
    )

    tuner = tune.Tuner(
        tune.with_resources(training_function, resources=resources),
        tune_config=tuner_config,
        run_config=run_config,
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
                              visualization_func=lambda x: 1 - math.sqrt(1 - x ** 2)):
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
            print(
                f'\n{labels[0]}: {p1}, {labels[1]}: {p2}, run {i * len(param1_axis) + ii}/{len(param1_axis) * len(param2_axis)}')
            setup_and_train(config_func, loaders, results, p1, p2)
    return results


# def setup_and_train(config_func, loaders, results, p1, p2):
def setup_and_train(config, train_data, test_data, support_data):
    # config = config_func(p1, p2)
    # train_data, test_data = get_data()
    loaders = load_data(train_data, test_data, config["batch_size"])
    support_loaders, query_loader = k_shot_loaders(support_data, config["shots"])

    model = emb_model.Convnet(device, lr=config["lr"], d=config["d"], num_of_classes=config["num_of_classes"],
                              channels=config["channels"]).to(device)
    optimiser = optim.Adam(model.parameters(), lr=model.lr)
    loss_func = nn_util.simple_dist_loss
    max_epochs = config["num_of_epochs"]

    for epoch in range(max_epochs):
        train(model, loaders, optimiser, loss_func, max_epochs, current_epoch=epoch, device=device)
        # accuracy = eval(model, loaders, device=device)
        correct, total = few_shot_eval(model, support_loaders, query_loader)
        accuracy = sum(correct) / sum(total)
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
        loss, loss_div = loss_func(res, labels, model.num_of_classes, device)
        optimiser.zero_grad()
        res.backward(gradient=loss_div)
        optimiser.step()
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}'
                  .format(current_epoch + 1, num_epochs, i + 1, total_step, loss.item()))


def eval(model, loaders, device):
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

                if smallest_k == labels[i].item():
                    correct += 1
                total += 1
        return correct / total


def few_shot_eval(model, support_loaders, query_loader):
    # Test the model
    model.eval()
    with torch.no_grad():
        # Get targets we have not seen before
        few_shot_targets = []
        for loader in support_loaders:
            for _, labels in loader:
                few_shot_targets.append(labels[0].item())
                break

        num_of_new_classes = len(few_shot_targets)
        new_class_embeddings = []
        correct = [0] * num_of_new_classes
        total = [0] * num_of_new_classes

        # get all images from support loaders
        for loader in support_loaders:
            for images, labels in loader:
                images = images.view(-1, 1, 28, 28).float().to(device)
                labels = labels.to(device)

                few_shot_output = model(images)

                new_class_embeddings.append(few_shot_output[:-model.num_of_classes])

        # calculate all average embeddings
        new_class_embeddings = [sum(item) / len(item) for item in new_class_embeddings]

        # do evaluation
        for images, labels in query_loader:
            images = images.view(-1, 1, 28, 28).float().to(device)
            test_output = model(images)

            for i, output_embedding in enumerate(test_output[:-model.num_of_classes]):
                smallest_sqr_dist = 100000000
                closest_target = 0
                closest_target_index = 0
                for k, target in enumerate(few_shot_targets):
                    actual_class_embedding = new_class_embeddings[k]
                    squared_dist = (actual_class_embedding - output_embedding).pow(2).sum(0)

                    if squared_dist < smallest_sqr_dist:
                        smallest_sqr_dist = squared_dist
                        closest_target = target
                        closest_target_index = k

                if closest_target == labels[i].item():
                    correct[closest_target_index] += 1
                total[closest_target_index] += 1

        return correct, total


if __name__ == '__main__':
    main()
