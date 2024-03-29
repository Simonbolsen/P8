from functools import partial
import os

import numpy as np
from Plotting.auto_plotter import make_experiment_plots
from file_util import get_files_dict
from training_utils import emc_classification_setup, setup_classification_custom_model, setup_emc_classification_pretrained, setup_pure_classification_pretrained
import argparse
from loader.loader import load_data, get_data, get_fs_data, get_data_loader, transforms_dict
from PTM.model_loader import load_pretrained
import torch
from torchvision import datasets, transforms
import ray
from ray import air, tune
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
from nn_util import cone_loss_hyperparam, pnp_hyperparam, simple_dist_loss, dist_and_proximity_loss, comparison_dist_loss, emc_loss_functions, pure_loss_functions
from few_shot_utils import setup_few_shot_pretrained, setup_few_shot_custom_model
from training_utils import train_emc, eval_classification
from bcolors import bcolors, printlc
import logging
from logging_utils import setup_logger
from tabulate import tabulate
from discord.status_setter import send_discord_message

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

def gtezero_float(x):
    x = float(x)
    if x < 0:
        raise argparse.ArgumentTypeError("Minimum value is =0")
    return x

def args_pretty_print(args):
    tab = tabulate(vars(args).items(), headers=["arg", "value"], missingval=f"{bcolors.WARNING}None{bcolors.ENDC}")
    return tab

datasets = {"mnist": 0, 
            "omniglot": 1, 
            "cifar10": 2,
            "cifar100": 3,
            "cifarfs": 4,
            "fc100": 5,
            "cub200": 6,
            "fashion": 7,
            "fashion_test": 8,
            "kuzushuji49": 9,
            "kmnist": 10}

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', dest="dataset", type=str, default="mnist", choices=datasets.keys(),
                        help="Determines the dataset on which training occurs. Choose between: ".format(datasets.keys()))
# Transformations to apply to the training and test data respectively
argparser.add_argument('--train_transforms', dest="train_transforms", type=str, default="toTensor", choices=transforms_dict.keys(),
                        help="Determines the transforms applied to the training data. Choose between: ".format(transforms_dict.keys()))
argparser.add_argument('--test_transforms', dest="test_transforms", type=str, default="toTensor", choices=transforms_dict.keys(),
                        help="Determines the transforms applied to the test data. Choose between: ".format(transforms_dict.keys()))

argparser.add_argument('--datadir', dest="data_dir", type=str, default="./data", help="Path to the data relative to working dir path")
argparser.add_argument('-fs', dest="few_shot", action="store_true", help="Few-shot flag")

# Training arguments
argparser.add_argument('--epochs', dest="epochs", type=gtzero_int, default=1, help="Epochs must be > 0")
argparser.add_argument('-test', dest='test', action='store_true', help="Flag for using the test setup")
argparser.add_argument('-se,', dest='save_embeds', action='store_true', help="Flag for saving embedding results")

# argparser.add_argument('--classes', dest="num_of_classes", type=gtzero_int, help="Number of unique classes for the dataset")
argparser.add_argument('--batch', dest="batch_size", nargs="+", type=gtzero_int, default=[100], help="Batch sizes to choose from. Must be > 0")

# Custom network settings
argparser.add_argument('--channels', dest="cnn_channels", nargs="+", type=gtzero_int, default=[16, 32, 64, 128], help="Number of channels in each convolutional layer")
# TODO: cnnlayers not used at the momenet(?)
argparser.add_argument('--cnnlayers', dest="cnn_layers", type=gtzero_int, default=4, help="Number of convolutional layers")
argparser.add_argument('--linlayers', dest="linear_layers", type=gtzero_int, default=5, help="Number of linear layers")
argparser.add_argument('--linsize', dest="linear_size", type=gtzero_int, default=100, help="Number of output features in linear layers")
argparser.add_argument('--stride', dest='stride', type=gtzero_int, default=1, help="Stride for convolutional layers")
argparser.add_argument('--kernsize', dest='kernel_size', type=gtzero_int, default=4, help="Size of kernal in convolutional layers")

argparser.add_argument('--loss-func', dest='loss_func', default='simple-dist', choices=list(emc_loss_functions.keys())+list(pure_loss_functions.keys()))
argparser.add_argument('--prox-mult', dest='prox_mult', nargs="+", default=[10,100], type=gtzero_int, 
                       help="Proximity multiplier for push loss functions. Only used with the push loss function")
argparser.add_argument('--p', dest='p', nargs="+", type=gtezero_float, default=[0, 2], help="p used in cone loss function")
argparser.add_argument('--q', dest='q', nargs="+", type=float, default=[-10, 10], help="q used in cone loss function")

# Pretrained
argparser.add_argument('-pt', dest="pretrained", action='store_true', 
                       help="If training should run a pretrained model")
argparser.add_argument('--model', dest='model', type=str, help='Model name to run for pretrained')
argparser.add_argument('-pure', dest="pure", action='store_true', help="Flag for using pure models as opposed to emc")
argparser.add_argument('--train-layers', dest='train_layers', nargs='+', type=int, default=[-1, 0], help='Number of layers of the pre-trained to train')
argparser.add_argument('-fe', dest='feature_extract', action='store_true', help="Flag for feature extraction")

# Few-shot
argparser.add_argument('--shots', dest="shots", type=gtzero_int, default=5, help="Shots in few-shot learning")

# Optimiser arguments
argparser.add_argument('--lr', dest="lr", nargs="+", type=gtzero_float, default=[0.00001, 0.0001], help="One or more learning rates")
argparser.add_argument('--dims', dest="dims", nargs="+", type=gtzero_int, default=[10, 100], help="One or more embedding dimensions")

# Raytune arguments
argparser.add_argument('--gpu', dest="gpu", type=gtzero_float, default=0.25, help="GPU resources")
argparser.add_argument('--cpu', dest="cpu", type=gtzero_float, default=3, help="CPU resources")
argparser.add_argument('--grace', dest="grace", type=gtzero_int, default=5, help="Grace period before early stopping")
argparser.add_argument('-t', dest="tuning", action="store_true", help="Tuning flag")
argparser.add_argument('--samples', dest='samples', type=gtzero_int, default=1, help='Samples to run for experiment')
argparser.add_argument('--exp-name', dest='exp_name', type=str, help='Name for raytune experiement')
argparser.add_argument('--verbosity', dest='verbosity', type=gezero_int, default=2, help='Verbosity level for raytune reporter.')
argparser.add_argument('--log', dest='log_level', type=str, help='Set log level for logger. See https://docs.python.org/3/howto/logging.html for levels.')

# Visualization Arguments
argparser.add_argument('--plot', dest="make_plots", action="store_true", help="Whether plots should be generated to output folder")

def legal_args(args):
    if (args.tuning):
        return len(args.dims) > 1 and len(args.lr) > 1 and \
              (len(args.cnn_channels) == args.cnn_layers) and \
              (len(args.batch_size) > 0) and \
              (len(args.prox_mult) >= 1)
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

def get_base_base_config(args):
    base_config = {
        "lr": hp.loguniform("lr", np.log(args.lr[0]), np.log(args.lr[1])),
        "max_epochs": args.epochs,
        "batch_size": hp.choice("batch_size", args.batch_size),
        "loss_func" : args.loss_func,
        "exp_name": args.exp_name,
        "dataset": args.dataset,
        "save_embeds": args.save_embeds
    }

    return base_config

def get_base_config(args):
    base_config = get_base_base_config(args)
    base_config["d"] = hp.uniformint("d", args.dims[0], args.dims[1])
    
    loss_func = emc_loss_functions[args.loss_func] if args.loss_func in emc_loss_functions else pure_loss_functions[args.loss_func]

    if not args.tuning:
        return base_config

    if loss_func is dist_and_proximity_loss:
        printlc(f"==> using class-push loss function... with prox_mult: {args.prox_mult}", bcolors.OKCYAN)
        base_config |= {
            "prox_mult" : hp.uniformint("prox_mult", args.prox_mult[0], args.prox_mult[1])
        }
    elif loss_func is cone_loss_hyperparam:
        printlc(f"==> using cone loss function with p = {args.p} and q = {args.q}", bcolors.OKBLUE)
        base_config |= {
            "q": hp.uniform("q", args.q[0], args.q[1]),
            "p": hp.uniform("p", args.p[0], args.p[1])
        }
    elif loss_func is pnp_hyperparam:
        # printlc(f"==> using pnp loss function with q = {args.q}", bcolors.OKBLUE)
        printlc("==> WARNING: USING PNP LOSS HARD CODED WITH q = 0.95", bcolors.WARNING)
        base_config |= {
            # "q": hp.uniform("q", args.q[0], args.q[1]),
            "q": 0.95
        }
    
    return base_config
    
def get_pure_base_config(args):
    base_config = get_base_base_config(args)
    
    base_config["train_layers"] = hp.uniformint("train_layers", args.train_layers[0], args.train_layers[1])
    loss_func = pure_loss_functions[args.loss_func]

    return base_config

def get_non_tune_base_config(args):
    base_config = get_base_config(args)
    base_config["lr"] = args.lr[0]
    base_config["d"] = args.dims[0]
    base_config["batch_size"] = args.batch_size[0]
    base_config["prox_mult"] = args.prox_mult[0]
    base_config["p"] = args.p[0]
    # base_config["q"] = args.q[0]
    base_config["q"] = 0.95
    printlc(f"==> WARNING: USING HARDCODED q = {base_config['q']}", bcolors.WARNING)
    base_config["train_layers"] = args.train_layers[0]

    return base_config


def get_scheduler(args):
    return AsyncHyperBandScheduler(grace_period=args.grace)

def get_run_config(args, metric_columens = ["accuracy", "training_iteration"]):
    reporter = tune.CLIReporter(
        metric_columns=metric_columens
    )
    
    stopper = TrialPlateauStopper(metric="accuracy", grace_period=args.grace, mode="max", num_results=6, std=0.005)
    
    return air.RunConfig(
            name=args.exp_name,
            progress_reporter=reporter,
            verbose=args.verbosity,
            stop=stopper
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
       "channels": args.cnn_channels,
       "linear_layers": args.linear_layers,
       "linear_size": args.linear_size,
       "stride": args.stride,
       "kernel_size": args.kernel_size
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
        printlc("fewshot custom network setup non ray function not implemented", bcolors.FAIL)
        os.exit(1)

def get_pretrained_config(args):
    pretrained_config = {
        "model_name" : args.model
    }
 
    if args.feature_extract:
        pretrained_config |= {
            "train_layers": hp.uniformint("train_layers", args.train_layers[0], args.train_layers[1]),
            "feature_extract": True
        }
    else:
        pretrained_config |= {
            "train_layers": -1,
            "feature_extract": False
        }

    return pretrained_config

def pretrained_fewshot(args):
    print("Running pretrained few shot")
    device = determine_device(ngpu=1)
    train_data, val_data, _  = get_fs_data(args)
    train_data_ptr = ray.put(train_data)
    val_data_ptr = ray.put(val_data)

    print("Training data size: ", len(train_data))
    print("Test data size: ", len(val_data))

    base_config = get_base_config(args)
    few_shot_config = get_few_shot_config(args)
    pretrained_config = get_pretrained_config(args)
    
    space = base_config | few_shot_config | pretrained_config
    
    setup_func = partial(setup_few_shot_pretrained, train_data=train_data_ptr,
                         few_shot_data=val_data_ptr, args=args, device=device, ray_tune=args.tuning)
    
    tuner = create_tuner(args, space, setup_func)

    if args.tuning:
        start_ray_experiment(tuner)
    else:
        print("==> starting non tune pretrained fewshot")
        non_tune_config = get_non_tune_base_config(args) | few_shot_config | pretrained_config
        setup_few_shot_pretrained(non_tune_config, train_data=train_data_ptr,
                         few_shot_data=val_data_ptr, args=args, device=device, ray_tune=args.tuning)

def custom_net_classification(args):
    device = determine_device(1)
    train_data, val_data, _ = get_data(args) # TODO: THIS MAYBE NEEDS TO BE FIXED???
    train_data_ptr = ray.put(train_data)
    val_data_ptr = ray.put(val_data)
    
    print("Training data size: ", len(train_data))
    print("Test data size: ", len(val_data))
    
    base_config = get_base_config(args)
    custom_net_config = get_custom_net_config(args)
    
    space = base_config | custom_net_config
    
    setup_func = partial(setup_classification_custom_model, 
                         training_data_ptr=train_data_ptr,
                         val_data_ptr=val_data_ptr, device=device, args=args, ray_tune=args.tuning)
    
    tuner = create_tuner(args, space, setup_func)
    
    if args.tuning:
        start_ray_experiment(tuner)
    else:
        printlc("running classification with custom network", bcolors.OKCYAN)
        non_tune_config = get_non_tune_base_config(args) | custom_net_config
        print("config: ", non_tune_config)
        setup_classification_custom_model(config=non_tune_config, training_data_ptr=train_data_ptr,
                                          val_data_ptr=val_data_ptr, device=device, args=args, ray_tune=args.tuning)

def pretrained_emc_classification(args):
    device = determine_device(1)
    train_data, val_data, _ = get_data(args) # TODO: THIS MAYBE NEEDS TO BE FIXED???
    train_data_ptr = ray.put(train_data)
    val_data_ptr = ray.put(val_data)

    print("Training data size: ", len(train_data))
    print("Validation data size: ", len(val_data))
    
    base_config = get_base_config(args)
    pretrained_config = get_pretrained_config(args)
    
    space = base_config | pretrained_config
        
    setup_func = partial(setup_emc_classification_pretrained, training_data_ptr=train_data_ptr, 
                         val_data_ptr=val_data_ptr, device=device, args=args, ray_tune=args.tuning)

    tuner = create_tuner(args, space, setup_func)

    if args.tuning:
        start_ray_experiment(tuner)
    else:
        non_tune_config =  pretrained_config | get_non_tune_base_config(args)
        print("==> pretrained embed classification config: ", non_tune_config)
        setup_emc_classification_pretrained(non_tune_config, training_data_ptr=train_data_ptr,
                                        val_data_ptr=val_data_ptr, device=device, args=args, ray_tune=args.tuning)

def pretrained_pure_classification(args):
    device = determine_device(1)
    # device = "cpu" # TODO: REMOVE THIS
    # printlc("USING CPU HARDCODED REMOVE THIS WHEN EXPERIMENT DONE!!!!!!!", bcolors.FAIL)
    train_data, val_data, _ = get_data(args) # TODO: THIS MAYBE NEEDS TO BE FIXED???
    train_data_ptr = ray.put(train_data)
    val_data_ptr = ray.put(val_data)

    print("Training data size: ", len(train_data))
    print("Validation data size: ", len(val_data))
    
    base_config = get_pure_base_config(args)
    pretrained_config = get_pretrained_config(args)
    
    space = base_config | pretrained_config
        
    setup_func = partial(setup_pure_classification_pretrained, training_data_ptr=train_data_ptr, 
                         val_data_ptr=val_data_ptr, device=device, args=args, ray_tune=args.tuning)

    tuner = create_tuner(args, space, setup_func)

    if args.tuning:
        start_ray_experiment(tuner)
    else:
        non_ray_config = pretrained_config | get_non_tune_base_config(args)
        print("==> pretrained pure classification: ", non_ray_config)
        setup_pure_classification_pretrained(non_ray_config, training_data_ptr=train_data_ptr,
                                        val_data_ptr=val_data_ptr, device=device, args=args, ray_tune=args.tuning)

def pure_test_classification(args):
    device = determine_device(1)
    train_data, test_data = get_data(args) 
    train_data_ptr = ray.put(train_data)
    test_data_ptr = ray.put(test_data)

    print("Training data size: ", len(train_data))
    print("Validation data size: ", len(test_data))
    
    pretrained_config = get_pretrained_config(args)

    non_ray_config = pretrained_config | get_non_tune_base_config(args)
    print("==> testing pretrained pure classification: ", non_ray_config)
    setup_pure_classification_pretrained(non_ray_config, training_data_ptr=train_data_ptr,
                                    val_data_ptr=test_data_ptr, device=device, args=args, ray_tune=args.tuning)


def emc_test_classification(args):
    device = determine_device(1)
    train_data, test_data = get_data(args) 
    train_data_ptr = ray.put(train_data)
    test_data_ptr = ray.put(test_data)

    print("Training data size: ", len(train_data))
    print("Validation data size: ", len(test_data))
    
    pretrained_config = get_pretrained_config(args)

    non_tune_config =  pretrained_config | get_non_tune_base_config(args)
    print("==> testing pretrained embed classification config: ", non_tune_config)
    setup_emc_classification_pretrained(non_tune_config, training_data_ptr=train_data_ptr,
                                    val_data_ptr=test_data_ptr, device=device, args=args, ray_tune=args.tuning)

def start_ray_experiment(tuner):
    printlc("starting experiment with ray tune", bcolors.OKBLUE)
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
    

def run_main(args):
    if (not legal_args(args)):
        raise argparse.ArgumentError("Illegal config")

    print(args_pretty_print(args))
    print("\n")
    
    if args.log_level:
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            logging.error('incorrect logging level... exiting...')
            os.quit(1)
        setup_logger(args.log_level.upper()) 
        print('==> Setting log level to: ', args.log_level)

    if args.few_shot:
        if args.pure:
            pretrained_fewshot(args)
        else:
            custom_net_fewshot(args)
    else:
        if args.test:
            if args.pure:
                pure_test_classification(args)
            else:
                emc_test_classification(args)
        else:
            if args.pretrained and args.pure:
                pretrained_pure_classification(args)
            elif args.pretrained:
                pretrained_emc_classification(args)
            else:
                custom_net_classification(args)

    if args.make_plots:
        make_experiment_plots(args.exp_name, os.path.join(os.path.expanduser("~/ray_plots"), args.exp_name))

if __name__ == '__main__':
    args = argparser.parse_args()
    send_discord_message(token_path="discord_token.secret", channel_id=1095627677848834128, message="Started\n" + str(args_pretty_print(args)))
    run_main(args)

    if args.make_plots:
        with get_files_dict(os.path.join(os.path.expanduser("~/ray_plots"), args.exp_name)) as plots:
            send_discord_message(token_path="discord_token.secret", channel_id=1095627677848834128, message="Done @here" + args.exp_name, files=plots)
    else:
        send_discord_message(token_path="discord_token.secret", channel_id=1095627677848834128, message="Done @here" + args.exp_name)

   