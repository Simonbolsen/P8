from torchvision import datasets

class Kuzushuji(datasets.MNIST):

    mirrors = ["http://codh.rois.ac.jp/kmnist/dataset/k49/"]

    resources = [
        ("k49-train-imgs.npz", None),
        ("k49-train-labels.npz", None),
        ("k49-test-imgs.npz", None),
        ("k49-test-labels.npz", None),
    ]
    

