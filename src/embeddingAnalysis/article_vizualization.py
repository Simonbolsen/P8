import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import os.path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import Plotting.plotting_util as plot
import numpy as np
import matplotlib.cm as cm
import math
import analysis_util as au
import file_util as fu
import embedding_visualization as ev
import analysis_util as au
from collections import Counter


def get_class_centers(embeddings, labels):
    num_of_classes = len(set(labels))
    class_center_train = [np.zeros(len(embeddings[0])) for _ in range (num_of_classes)]
    class_size = [0 for _ in range(10)]
    for i, e in enumerate(embeddings):
        label = labels[i]
        npe = np.array(e)
        class_center_train[label] = class_center_train[label] + npe
        class_size[label] += 1

    return [e / class_size[i] for i, e in enumerate(class_center_train)]

def save_class_centers(train_embeddings, train_labels, val_class_embeddings, val_labels, path, center_path):
    class_centers = {"train": get_class_centers(train_embeddings, train_labels), "val": get_class_centers(val_class_embeddings, val_labels)}
    
    print(f"Saving {center_path}...")
    fu.save_to_pickle(path, center_path, class_centers)
    print("Class Centers Saved")
    return class_centers

def load_pure_data(epoch, data_folder) :
    path = os.path.dirname(__file__) + "/../../embeddingData/" + data_folder
    data = fu.read_pickle_file(path + f"/classification_data_train_{epoch}.p")
    train_embeddings = data["train_embeddings"]
    train_labels = data["train_labels"]
    #train_predictions = data["predictions"]

    data = fu.read_pickle_file(path + f"/classification_data_val_{epoch}.p")
    val_embeddings = data["val_embeddings"]
    val_labels = data["val_labels"]
    #val_predictions = data["predictions"]

    center_path = f"class_centers_{epoch}.p"
    if os.path.exists(path + "/" + center_path):
        class_centers = fu.read_pickle_file(path + "/" + center_path)
    else:
        class_centers = save_class_centers(train_embeddings, train_labels, val_embeddings, val_labels, path, center_path)

    data = fu.read_json_file(path + "/meta_data.json")
    #data["train_predictions"] = train_predictions
    #data["val_predictions"] = val_predictions

    return train_embeddings, train_labels, val_embeddings, val_labels, data, class_centers["val"], class_centers["train"]

def plot_acc_by_dist_funcs_and_epoch(data_folder, save = False): 
    ed_acc = []
    cod_acc = []
    cacd_acc = []
    epochs = range(0, 30)
    for epoch in epochs:
        print(f"\n{epoch + 1} / 30.", end="")
        euclidean_misclassified = 0
        cosine_origo_misclassified = 0
        
        train_embeddings, train_labels, val_embeddings, val_labels, data, val_class_embeddings, train_class_embeddings = load_pure_data(epoch, data_folder)
        #val_predictions = data["val_predictions"]

        center = np.zeros(len(train_class_embeddings[0]))
        for i in train_class_embeddings:
            center += i
        center = center / len(train_class_embeddings)

        eucledian_dist = lambda c, v : ((np.array(v) - np.array(c))**2).sum()
        def cosine_origo_dist(c,v): 
            cnp = np.array(c)
            vnp = np.array(v)
            return -vnp.dot(cnp) / (np.linalg.norm(vnp) * np.linalg.norm(cnp))

        print(end=".")
        for i, v in enumerate(train_embeddings):
            label = train_labels[i]

            clossest_ed_class = au.argmin([eucledian_dist(c,v) for c in train_class_embeddings])
            clossest_cod_class = au.argmin([cosine_origo_dist(c,v) for c in train_class_embeddings])
            
            euclidean_misclassified += 0 if clossest_ed_class == label else 1
            cosine_origo_misclassified += 0 if clossest_cod_class == label else 1
            
        print(end=".")
        ed_acc.append(plot.round_scale(1.0 - euclidean_misclassified/len(train_embeddings)))
        cod_acc.append(plot.round_scale(1.0 - cosine_origo_misclassified/len(val_embeddings)))
        
    plot.plot_line_2d([i for i in epochs], [ed_acc, cacd_acc, [plot.round_scale(data["accuracies"][i]) for i in epochs]], 
                      ["Euclidean", "Cosine avg Center", "Pure"], x_label = "Epoch", y_label = "Classification Accuracy",
                      save_path= data_folder + "/acc_plot.png" if save else "") #Closest class center classification
    


def pca_pure_classification(data_folder, save = False):
    i = 29
    train_embeddings, train_labels, val_embeddings, val_labels, data, val_class_embeddings, train_class_embeddings = load_pure_data(i, data_folder)

    points = 100

    scores = []
    xs = []
    done = False
    i = 5
    while not done and i < points:
        pca = PCA(n_components = i)
        if i % int(points / 10) == 0:
            print(i)
        i += int(len(train_embeddings[0]) / points)
        _ = pca.fit_transform(train_embeddings)
        try:
            scores.append(pca.score(val_embeddings))
            xs.append(i)
        except:
            done = True
        
    return xs, scores
    #plot.plot_line_2d(xs, 
    #                  [scores], 
    #                  ["PCA Scores"], lambda x:x, "PCA Components", "PCA Score", data_folder + "/pca_plot" if save else "" )
    
def get_dist_(data_folder):
    all_medians = []
    all_cetc = []

    epochs = range(1, 30)
    for epoch in epochs:
        print(epoch)
        train_embeddings, train_labels, val_embeddings, val_labels, data, val_class_centers, train_class_centers = load_pure_data(epoch, data_folder)        
        val_dists = au.get_dists(val_embeddings, val_labels, train_class_centers, lambda x:x)

        center = np.zeros(len(train_class_centers[0]))
        for class_center in train_class_centers:
            center = center + np.array(class_center)
        
        center = center / len(train_class_centers)

        cetc = 0
        for c in train_class_centers:
            cetc += np.linalg.norm(np.array(c) - center)

        val_dists.sort()
        median = val_dists[int(len(val_dists)/2)]
        all_cetc.append(cetc)
        all_medians.append(median)

    return all_medians, all_cetc

    #plot.plot_line_2d([i for i in epochs], all_medians, [f"Class {i}" for i in range(10)], lambda x:x, 
    #                  "Epochs", "Mediuan Distance to Class Center", save_path=data_folder + "/median_plot.png" if save else "")
    #plot.plot_line_2d([i for i in epochs], all_cetc, [f"Class {i}" for i in range(10)], lambda x:x, 
    #                  "Epochs", "Class Center Distance to Avg Class Center", data_folder + "/cetc_plot.png" if save else "")