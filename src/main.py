from functools import partial
import os
from training_utils import classification_setup
import argparse
from loader.loader import load_data, get_data, get_fs_data, get_data_loader, transforms_dict
from PTM.model_loader import load_pretrained
import torch
from torchvision import datasets, transforms
import ray
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
from nn_util import simple_dist_loss, dist_and_proximity_loss, comparison_dist_loss, loss_functions
from few_shot_utils import setup_few_shot_pretrained, setup_few_shot_custom_model
from training_utils import train, eval_classification
from bcolors import bcolors, printlc
import logging

def gezero_int(x):
    x = int()
    if x < 0:
        raise argparse.ArgumentTypeError("Minimum value is 0")
    return x

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
            "cifarfs": 4,
            "fc100": 5}

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', dest="dataset", type=str, default="mnist", choices=datasets.keys(),
                        help="Determines the dataset on which training occurs. Choose between: ".format(datasets.keys()))

argparser.add_argument('--train_transforms', dest="train_transforms", type=str, default="toTensor", choices=transforms_dict.keys(),
                        help="Determines the transforms applied to the training data. Choose between: ".format(transforms_dict.keys()))
argparser.add_argument('--test_transforms', dest="test_transforms", type=str, default="toTensor", choices=transforms_dict.keys(),
                        help="Determines the transforms applied to the test data. Choose between: ".format(transforms_dict.keys()))

argparser.add_argument('--datadir', dest="data_dir", type=str, default="./data", help="Path to the data relative to working dir path")
argparser.add_argument('-fs', dest="few_shot", action="store_true", help="Few-shot flag")

# Training arguments
argparser.add_argument('--epochs', dest="epochs", type=gtzero_int, default=1, help="Epochs must be > 0")
# argparser.add_argument('--classes', dest="num_of_classes", type=gtzero_int, help="Number of unique classes for the dataset")
# TODO: batch size list
argparser.add_argument('--batch', dest="batch_size", type=gtzero_int, default=100, help="Batch size must be > 0")

argparser.add_argument('--channels', dest="cnn_channels", nargs="+", type=gtzero_int, default=[16, 32, 64, 128, 256], help="Number of channels in each convolutional layer")
argparser.add_argument('--layers', dest="cnn_layers", type=gtzero_int, default=5, help="Number of convolutional layers")

argparser.add_argument('--loss-func', dest='loss_func', default='simple-dist', choices=loss_functions.keys())

# Pretrained
argparser.add_argument('-pt', dest="pretrained", action='store_true', help="If training should run a pretrained model")
argparser.add_argument('--model', dest='model', type=str, help='Model name to run for pretrained')

# Few-shot
argparser.add_argument('--shots', dest="shots", type=gtzero_int, default=5, help="Shots in few-shot learning")

# Optimiser arguments
argparser.add_argument('--lr', dest="lr", nargs="+", type=gtzero_float, default=[0.00001, 0.0001], help="One or more learning rates")
argparser.add_argument('--dims', dest="dims", nargs="+", type=gtzero_int, default=[10, 100], help="One or more embedding dimensions")

# Raytune arguments
argparser.add_argument('--gpu', dest="gpu", type=gtzero_float, default=0.25, help="GPU resources")
argparser.add_argument('--cpu', dest="cpu", type=gtzero_float, default=3, help="CPU resources")
argparser.add_argument('--grace', dest="grace", type=gtzero_int, default=4, help="Grace period before early stopping")
argparser.add_argument('-t', dest="tuning", action="store_true", help="Tuning flag")
argparser.add_argument('--samples', dest='samples', type=gtzero_int, default=1, help='Samples to run for experiment')
argparser.add_argument('--exp-name', dest='exp_name', type=str, help='Name for raytune experiement')
argparser.add_argument('--verbosity', dest='verbosity', type=gezero_int, default=2, help='Verbosity level for raytune reporter.')
argparser.add_argument('--log', dest='log_level', type=str, help='Set log level for logger. See https://docs.python.org/3/howto/logging.html for levels.')


def legal_args(args):
    if (args.tuning):
        return len(args.dims) > 1 and len(args.lr) > 1 and (len(args.cnn_channels) == args.cnn_layers)
    return True

def determine_device(ngpu):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if (device.type == 'cuda'):
        print(f'{bcolors.OKGREEN}OK: Using GPU{bcolors.ENDC}')
    else:
        print(f'{bcolors.WARNING}Warning: Using CPU{bcolors.ENDC}')
    
    return device
   

# autoAugment = transforms.AutoAugment(AutoAugmentPolicy = transforms.AutoAugmentPolicy.IMAGENET, interpolation: InterpolationMode = InterpolationMode.NEAREST, fill: Optional[List[float]] = None)
autoAugment = transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
def torch_augment_image(img: torch.Tensor) -> torch.Tensor:
    return autoAugment.forward(img)


def get_base_config(args):
    base_config = {
        "lr": hp.uniform("lr", args.lr[0], args.lr[1]),
        "max_epochs": args.epochs,
        "batch_size": args.batch_size, # TODO: make choice?
        "d" : hp.uniformint("d", args.dims[0], args.dims[1]),
        "loss_func" : args.loss_func,
        "augment_image": torch_augment_image
    }
    
    return base_config
    
def get_non_tune_base_config(args):
    base_config = {
        "lr": args.lr[0],
        "max_epochs": args.epochs,
        "batch_size": args.batch_size, # TODO: make choice?
        "d" : args.dims[0],
        "loss_func" : args.loss_func,
        "augment_image": torch_augment_image
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
            verbose=args.verbosity
    )


def get_tune_config(args, search_alg, metric="accuracy", mode="max"):
    scheduler = get_scheduler(args)    
     
    return tune.TuneConfig(
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

   
def get_custom_net_config(args):
    return {
        
    }

def custom_net_fewshot(args):
    printlc("Running custom new few shot", bcolors.OKCYAN)
    device = determine_device(ngpu=1)
    train_data, val_data, _  = get_fs_data(args)
    train_data_ptr = ray.put(train_data)
    val_data_ptr = ray.put(val_data)

    print("Training data size: ", len(train_data))
    print("Test data size: ", len(val_data))
 
    base_config = get_base_config(args)
    custom_net_config = get_custom_net_config(args)

    space = base_config | custom_net_config

    setup_func = partial(setup_few_shot_custom_model, train_data_ptr=train_data_ptr, 
                         few_shot_data_ptr=val_data_ptr, device=device, args=args, ray_tune=args.tuning) 

    tuner = create_tuner(args, space, setup_func)
    
    if args.tuning:
        start_ray_experiment(tuner)
    else:
        print(f"{bcolors.FAIL}fewshot custom network setup non ray function not implemented{bcolors.ENDC}")
        os.exit(1)


def pretrained_fewshot(args):
    print("Running pretrained few shot")
    device = determine_device(ngpu=1)
    train_data, val_data, _  = get_fs_data(args)
    train_data_ptr = ray.put(train_data)
    val_data_ptr = ray.put(val_data)

    print("Training data size: ", len(train_data))
    print("Test data size: ", len(val_data))
    model = args.model

    base_config = get_base_config(args)
    few_shot_config = get_few_shot_config(args)
    
    space = base_config | few_shot_config
    
    setup_func = partial(setup_few_shot_pretrained, model_name=model, train_data=train_data_ptr,
                         few_shot_data=val_data_ptr, args=args, device=device, ray_tune=args.tuning)
    
    tuner = create_tuner(args, space, setup_func)

    if args.tuning:
        start_ray_experiment(tuner)
    else:
        setup_few_shot_pretrained(get_non_tune_base_config(args) | few_shot_config, model_name=model, train_data=train_data_ptr,
                         few_shot_data=val_data_ptr, args=args, device=device, ray_tune=args.tuning)

def start_ray_experiment(tuner):
    print(f"{bcolors.OKBLUE}starting experiment with ray tune{bcolors.ENDC}")
    results = tuner.fit()
    print(results.get_best_result().metrics)

def create_tuner(args, space, setup_func):
    resources = {"cpu": args.cpu, "gpu": args.gpu}
    search_alg = get_hyper_opt(space)
    
    tuner_config = get_tune_config(args, search_alg)
    run_config = get_run_config(args)
    
    tuner = tune.Tuner(
        tune.with_resources(setup_func, resources=resources),
        tune_config=tuner_config,
        run_config=run_config,
    )
    
    return tuner
    
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


def run_main(args):
    if (not legal_args(args)):
        raise argparse.ArgumentError("Illegal config")
    
    if args.loglevel:
        numeric_level = getattr(logging, args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            logging.error('incorrect logging level... exiting...')
            os.exit(1)
        logging.basicConfig(level=numeric_level)
        print('Setting log level to: ', args.loglevel)

    print(args.dataset)

    if args.few_shot:
        if args.pretrained:
            pretrained_fewshot(args)
        else:
            custom_net_fewshot(args)
        # run_tune_fewshot(args)
    else:
        pass
        # run_tune(args)

if __name__ == '__main__':
    args = argparser.parse_args()
    run_main(args)
   