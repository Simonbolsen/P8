import os
from ray import tune
import Plotting.results_analysis_util as analysis
import json

# THIS SCRIPT ASSUMES THE FOLLOWING:
# assumes max epochs = training iteration
# and feature extraction not used
# the model used is a PRETRAINED network
# if cross entropy is used the model is pure

experiments = [
    "cl_embed_push_res_large_fashion",
    "cl_embed_cosine_res_large_fashion",
    "cl_embed_simple_res_large_fashion",
    "cl_embed_push_res_med_fashion",
    "cl_embed_cosine_res_med_fashion",
    "cl_embed_simple_res_med_fashion",
    "cl_embed_push_res_small_fashion",
    "cl_embed_cosine_res_small_fashion",
    "cl_embed_simple_res_small_fashion",
    "cl_pure_res_large_fashion_mnist",
    "cl_pure_res_med_fashion_mnist",
    "cl_pure_res_small_fashion_mnist"
]

config_to_param = {
    "batch_size": "--batch",
    "d": "--dims",
    "dataset": "--dataset",
    "exp_name": "--exp-name",
    "feature_extract": "-fe",
    "loss_func": "--loss-func",
    "lr": "--lr",
    # "max_epochs": "--epochs", # training iteration
    "model_name": "--model",
    "prox_mult": "--prox-mult"
}

dataset_to_transform = {
    "fashion": "cheap_mnist_resnet",
    "cifar10": "cifar10_resnet",
    "cifar100": "cifar100_resnet"
}

exp_jsons = []

for exp in experiments:
    path = os.path.join("~/ray_results/", exp)
    results = tune.ExperimentAnalysis(path, 
                                  default_metric="accuracy", 
                                  default_mode="max")
    
    best_results = analysis.best_iterations_per_trial(results)

    keys_by_acc = {result["data"]["accuracy"]: key for key, result in best_results.items()}
    best = best_results[keys_by_acc[max(keys_by_acc.keys())]]
    
    exp_json = {
        "label": "BEST : " + best['config']['exp_name'],
        "command": "${command:python.interpreterPath}",
        "type": "shell",
        "group": {
            "kind": "build",
            "isDefault": True
        },
        "args": [
            "./src/main.py",
            "-test",
            "-se",
            "-pt",
            "--datadir", "'./data'",
            # "--epochs", str(best['data']['training_iteration'])
            "--epochs", "30"
        ],
        "presentation": {
            "reveal": "always",
            "panel": "new",
            "focus": True
        }
    }

    for param, value in best['config'].items():
        if param == 'max_epochs' or param == 'train_layers' or param == 'feature_extract':
            continue

        arg_name = config_to_param[param]

        # Ensure it doesnt break
        if param == 'lr':
            exp_json["args"].append(arg_name)
            exp_json["args"].append(str(value))
            exp_json["args"].append(str(1))
        elif param == 'prox_mult' or param == 'd':
            exp_json["args"].append(arg_name)
            exp_json["args"].append(str(value))
            exp_json["args"].append(str(value + 1))
        # just in case so old folder doesnt get overwritten but shouldnt be
        # needed cause we dont run tuning
        elif param == 'exp_name':
            exp_json["args"].append(arg_name)
            exp_json["args"].append(str(value) + "_BEST")
        elif param == 'loss_func' and value == 'cross_entropy':
            exp_json["args"].append('-pure') 
            exp_json["args"].append(arg_name)
            exp_json["args"].append(value)
        else:
            exp_json["args"].append(arg_name)
            exp_json["args"].append(str(value))

    transform = dataset_to_transform[best['config']['dataset']]

    exp_json["args"].append("--train_transforms")
    exp_json["args"].append(transform)

    exp_json["args"].append("--test_transforms")
    exp_json["args"].append(transform)

    exp_jsons.append(exp_json)

with open('./best_runs.txt', 'x') as f:
    f.write(json.dumps(exp_jsons, indent=4))


