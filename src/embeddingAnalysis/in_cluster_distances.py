import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from article_vizualization import load_meta_data, is_pure, cosine, load_data
import file_util as fu
import Plotting.plotting_util as plot

def get_avg_dist_between_class_centers(class_centers):
    sum_of_distances = 0

    for i in range(len(class_centers)):
        for j in range(len(class_centers)):
            if j > i:
                sum_of_distances += np.linalg.norm(class_centers[i] - class_centers[j])

    return sum_of_distances / len(class_centers)




if __name__ == '__main__':

    data_folder = os.path.join("plots", "plotData")
    plot_folder = "plots"
    folder = "cl_embed_cosine_res_small_fashion_BEST"
    rel_path = os.path.join(os.path.realpath(__file__), '..', '..', '..')

    meta_data = load_meta_data(folder)
    avg_dists_class_centers = []
    epochs = range(0, meta_data["config"]["max_epochs"])
    for epoch in epochs:
        _, _, _, _, class_centers, _, _ = load_data(epoch, folder)

        avg_dists_class_centers.append(get_avg_dist_between_class_centers(class_centers))

    saved_data = fu.read_json_file(os.path.join(rel_path, data_folder, folder+".json"))
    medians = saved_data["medians"][1]["median_to_class_centers_eucledian"]

    plot.plot_line_2d(epochs, [avg_dists_class_centers], ["daw"], lambda x:x)

    print(" ")





