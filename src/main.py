from training_utils import classification_setup
import argparse
from loader.loader import load_data, get_data, get_fs_data, get_data_loader
from PTM.model_loader import load_pretrained
import torch
from torchvision import datasets
import ray
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
from nn_util import simple_dist_loss, dist_and_proximity_loss, comparison_dist_loss
from few_shot_utils import setup_few_shot_pretrained
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

# Pretrained
argparser.add_argument('--pretrained', dest="pretrained", action='store_true', help="If training should run a pretrained model")
argparser.add_argument('--model', dest='model', type=str, help='Model name to run for pretrained')

# Few-shot
argparser.add_argument('--shots', dest="shots", type=gtzero_int, help="Shots in few-shot learning")

# Optimiser arguments
argparser.add_argument('--lr', dest="lr", nargs="+", type=gtzero_float, default=[0.00001, 0.0001], help="One or more learning rates")
argparser.add_argument('--dims', dest="dims", nargs="+", type=gtzero_int, default=[10, 100], help="One or more embedding dimensions")

# Raytune arguments
argparser.add_argument('--gpu', dest="gpu", type=gtzero_float, default=0.25, help="GPU resources")
argparser.add_argument('--cpu', dest="cpu", type=gtzero_float, default=3, help="CPU resources")
argparser.add_argument('--grace', dest="grace", type=gtzero_int, default=4, help="Grace period before early stopping")
argparser.add_argument('-t', dest="tuning", action="store_true", help="Tuning flag")
argparser.add_argument('--samples', dest='samples', type=gtzero_int, help='Samples to run for experiment')

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
   
def get_base_config(args):
    base_config = {
        "lr": hp.uniform("lr", args.lr[0], args.lr[1]),
        "max_epochs": args.epochs,
        "batch_size": args.batch_size, # TODO: make choice?
        "d" : args.dims
    }
    
    return base_config
    
def get_scheduler(args):
    return AsyncHyperBandScheduler(grace_period=args.grace)

def get_run_config(args, metric_columens = ["accuracy", "training_iteration"]):
    reporter = tune.CLIReporter(
        metric_columns=metric_columens
    )
    
    return air.RunConfig(
            name=args.exp_name,
            progress_reporter=reporter,
    )


def get_tune_config(args, search_alg, metric="accuracy", mode="max"):
    scheduler = get_scheduler(args)    
    
    tune.TuneConfig(
            metric=metric,
            mode=mode,
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=args.samples
    )
    
def get_few_shot_config(args):
    return {
        "shots" : args.shots
    }

def get_hyper_opt(space, metric="accuracy", mode="max", good_starts=None):
    return HyperOptSearch(space, metric=metric, mode=mode, 
                                #   n_initial_points=2, 
                                points_to_evaluate=good_starts)

# def run_tune_fewshot(args):
#     device = determine_device(ngpu=1)
#     train_data, val_data, _ = get_data(args)

#     print("Training data size: ", len(train_data))
#     print("Validation data size: ", len(val_data))

#     base_config = get_base_config(args)

#     smoke_test_space = {
#             "lr": hp.uniform("lr", args.lr[0], args.lr[1]),
#             "d": hp.uniformint("d", args.dims[0], args.dims[1]),
#             "num_of_classes": args.num_of_classes,
#             "channels": hp.choice("channels", args.cnn_channels),
#             "batch_size": args.batch_size,
#             "num_of_epochs": hp.uniformint("num_of_epochs", args.epochs[0], args.epochs[1])
#         }
    
#     good_start = {"num_of_epochs": 10,
#                   "lr": 0.0005,
#                   "d" : 60,
#                   "channels" : 64,
#                   "num_of_classes": 64,
#                   "batch_size": 100,
#                   "k_size": 4,
#                   "stride": 1,
#                   "linear_n": 1,
#                   "linear_size": 64,
#                   "shots": 5
#                   }

#     hyper_opt_search = HyperOptSearch(smoke_test_space, 
#                                       metric="accuracy", 
#                                       mode="max", 
#                                       points_to_evaluate=[good_start])

#     tuner_config = get_tune_config(args, hyper_opt_search)

#     run_config = get_run_config(args)

#     tuner = tune.Tuner(
#         tune.with_parameters(classification_setup, train_data=train_data, test_data=None),
#         tune_config=tuner_config,
#         run_config=run_config
#     )
    
#     if (args.tuning):
#         results = tuner.fit()
#         print(results.get_best_result().metrics)
#     else:
#         # classification_setup(good_start, train_data, test_data, loss_func, device, ray_tune=False)
#         # train_few_shot(good_start, train_data, val_data, None, loss_func, device, ray_tune=False)
#         setup_and_finetune(good_start, train_data, val_data, device)

def pretrained_fewshot(args):
    device = determine_device(ngpu=1)
    train_data, val_data,  = get_fs_data(args)
    print("Training data size: ", len(train_data))
    print("Test data size: ", len(val_data))

    base_config = get_base_config(args)
    few_shot_config = get_few_shot_config(args)
    model = args.model
    
    space = base_config | few_shot_config
    
    search_alg = get_hyper_opt(space)
    
    tuner_config = get_tune_config(args, search_alg)
    run_config = get_run_config(args)
    
    tuner = tune.Tuner(
        tune.with_parameters(setup_few_shot_pretrained, model_name=model, train_data=train_data, few_shot_data=val_data, device=device),
        tune_config=tuner_config,
        run_config=run_config
    )

    if args.tuning:
        results = tuner.fit()
        print(results.get_best_result().metrics)
    else:
        print("fewshot pretrained setup non ray function not implemented")
        exit(1)

# def run_tune(args):
#     device = determine_device(ngpu=1)
#     train_data, test_data,  = get_data(args)

#     print("Training data size: ", len(train_data))
#     print("Test data size: ", len(test_data))

#     resources = {"cpu": args.cpu, "gpu": args.gpu}
#     scheduler = AsyncHyperBandScheduler(grace_period=args.grace)
#     reporter = tune.CLIReporter(
#         metric_columns=["accuracy", "training_iteration"]
#     )
    
#     loss_func = simple_dist_loss

#     smoke_test_space = {
#             "lr": hp.uniform("lr", args.lr[0], args.lr[1]),
#             "d": hp.uniformint("d", args.dims[0], args.dims[1]),
#             "num_of_classes": args.num_of_classes,
#             "channels": hp.choice("channels", args.cnn_channels),
#             "batch_size": args.batch_size,
#             "num_of_epochs": hp.uniformint("num_of_epochs", args.epochs[0], args.epochs[1])
#         }
    
#     good_start = {"num_of_epochs": 10,
#                   "lr": 0.0005,
#                   "d" : 60,
#                   "channels" : 64,
#                   "num_of_classes": 10,
#                   "batch_size": 100,
#                   "k_size": 4,
#                   "stride": 1,
#                   "linear_n": 1,
#                   "linear_size": 64,
#                   "shots": 5
#                   }

#     hyper_opt_search = HyperOptSearch(smoke_test_space, 
#                                       metric="accuracy", 
#                                       mode="max", 
#                                     #   n_initial_points=2, 
#                                       points_to_evaluate=[good_start])

#     tuner_config = tune.TuneConfig(
#             metric="accuracy",
#             mode="max",
#             scheduler=scheduler,
#             search_alg=hyper_opt_search,
#             num_samples=1000
#     )

#     run_config = air.RunConfig(
#             name="mnist_initial_test",
#             progress_reporter=reporter,
#             # stop={"training_iteration": 10}
#     )

#     tuner = tune.Tuner(
#         tune.with_parameters(classification_setup, train_data=train_data, test_data=test_data),
#         tune_config=tuner_config,
#         run_config=run_config
#     )
    
#     if (args.tuning):
#         results = tuner.fit()
#         print(results.get_best_result().metrics)
#     else:
#         # classification_setup(good_start, train_data, test_data, loss_func, device, ray_tune=False)
#         # train_few_shot(good_start, train_data, test_data, test_data, loss_func, device, ray_tune=False)
#         setup_and_finetune(good_start, train_data, test_data, device)

if __name__ == '__main__':
    args = argparser.parse_args()
    if (not legal_args(args)):
        raise argparse.ArgumentError("Illegal config")

    if args.tuning:
        ray.init(num_cpus=args.cpu, num_gpus=args.gpu)

    if args.pretrained:
        pretrained_fewshot(args)
    print(args.dataset)

    if (args.few_shot):
        run_tune_fewshot(args)
    else:
        run_tune(args)

