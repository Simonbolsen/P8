import os
import sys
import matplotlib.scale as scales
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from article_vizualization import load_meta_data, is_pure, cosine, load_data, get_path
import file_util as fu
import Plotting.plotting_util as plot

def get_avg_dist_between_class_centers(class_centers):
    sum_of_distances = 0

    for i in range(len(class_centers)):
        for j in range(len(class_centers)):
            if j > i:
                sum_of_distances += np.linalg.norm(class_centers[i] - class_centers[j])

    return sum_of_distances / len(class_centers)


def get_avg_dist_data_from_folder(folder, max_epoch):
    avg_dists_class_centers = []
    epochs = range(0, max_epoch)
    for epoch in epochs:
        class_centers = load_center_data(epoch, folder)

        avg_dists_class_centers.append(get_avg_dist_between_class_centers(class_centers))

    return avg_dists_class_centers

def load_center_data(epoch, data_folder):
    path = get_path(data_folder)
    center_path = f"class_centers_{epoch}.p"
    if os.path.exists(path + "/" + center_path):
        class_centers = fu.read_pickle_file(path + "/" + center_path)
    
    return class_centers["val"]


if __name__ == '__main__':

    data_folder = os.path.join("plots", "plotData")
    plot_folder = "plots"
    # folders = [
    #     "cl_embed_push_res_small_mnist_BEST",
    #     "cl_embed_push_res_small_fashion_BEST",
       
    #     "cl_embed_push_res_small_kmnist_BEST",
    #     "cl_embed_push_res_small_cifar_10_BEST",
    #     "cl_embed_push_res_small_cifar_100_BEST"
    # ]
    folders = [
        "cl_embed_pnp_res_small_mnist_BEST",
        "cl_embed_pnp_res_small_fashion_BEST",
        "cl_embed_PNP_res_small_kmnist_BEST",
        "cl_embed_pnp_res_small_cifar_100_BEST"
    ]
    rel_path = os.path.join(os.path.realpath(__file__), '..', '..', '..')

    
    dist_data = []
    # names = ["MNIST", "F-MNIST", "K-MNIST", "CIFAR10", "CIFAR100"]
    names = ["MNIST", "F-MNIST", "K-MNIST", "CIFAR100"]

    for f in folders:
        meta_data = load_meta_data(f)
        dist_data.append(get_avg_dist_data_from_folder(f, meta_data["config"]["max_epochs"]))

    # saved_data = fu.read_json_file(os.path.join(rel_path, data_folder, folder+".json"))
    # medians = saved_data["medians"][1]["median_to_class_centers_eucledian"]

    plot.plot_line_2d(range(0, 30), dist_data, names, lambda x:x, x_label="Epoch ep", y_label="Mean Class Centre Distance", 
                      y_scale=scales.LogScale(None))#, straigh_lines=[dists[0] for dists in dist_data])

    print(" ")





