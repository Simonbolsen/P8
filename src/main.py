
import argparse

def gtzero_int(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError("Minimum value is 1")
    return x


datasets = {"mnist": 0, "omniglot": 1}

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', dest="dataset", type=str, default="mnist", choices=datasets.keys(),
                        help="Determines the dataset on which training occurs. Choose between: ".format(datasets.keys()))
argparser.add_argument('--epochs', dest="epochs", type=gtzero_int, default=3, help="Epochs must be > 0")
argparser.add_argument('--batch', dest="batch_size", type=gtzero_int, default=100, help="Batch size must be > 0")
argparser.add_argument('--dims', dest="embedding_dims", type=gtzero_int, default=2, help="Embedding dimensions must be > 0")
argparser.add_argument('--channels', dest="cnn_channels", type=gtzero_int, default=64, help="Number of channels in each convolutional layer")

if __name__ == '__main__':
    args = argparser.parse_args()

    print(args.dataset)
    print("Determines the dataset on which training occurs. Choose between: {}".format(", ".join(datasets)))

