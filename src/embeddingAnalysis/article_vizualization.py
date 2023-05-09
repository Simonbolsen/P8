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

def get_path(data_folder):
    return os.path.dirname(__file__) + "/../../embeddingData/" + data_folder

def load_meta_data(data_folder):
    return fu.read_json_file(get_path(data_folder) + "/meta_data.json")

def load_data(epoch, data_folder) :
    path = get_path(data_folder)
    data = fu.read_pickle_file(path + f"/classification_data_train_{epoch}.p")
    train_embeddings = data["train_embeddings"]
    train_labels = data["train_labels"]
    #train_predictions = data["predictions"]
    class_embeddings = data["train_class_embeddings"] if "train_class_embeddings" in data else None

    data = fu.read_pickle_file(path + f"/classification_data_val_{epoch}.p")
    val_embeddings = data["val_embeddings"]
    val_labels = data["val_labels"]
    #val_predictions = data["predictions"]

    center_path = f"class_centers_{epoch}.p"
    if os.path.exists(path + "/" + center_path):
        class_centers = fu.read_pickle_file(path + "/" + center_path)
    else:
        class_centers = save_class_centers(train_embeddings, train_labels, val_embeddings, val_labels, path, center_path)

    return train_embeddings, train_labels, val_embeddings, val_labels, class_centers["val"], class_centers["train"], class_embeddings

def is_pure(meta_data):
    return meta_data["config"]["loss_func"] == "cross_entropy" #pure" in meta_data["config"] and meta_data["config"]["pure"]

def cosine_origo_dist(c,v): 
    cnp = np.array(c)
    vnp = np.array(v)
    return -vnp.dot(cnp) / (np.linalg.norm(vnp) * np.linalg.norm(cnp))

def acc_by_dist_funcs(data_folder, save = False): 
    ed_acc = []
    cod_acc = []
    ce_cod_acc = []

    meta_data = load_meta_data(data_folder)
    epochs = range(0, meta_data["config"]["max_epochs"])
    for epoch in epochs:
        print(f"\n{epoch + 1} / 30.", end="")
        euclidean_misclassified = 0
        cosine_origo_misclassified = 0
        class_embedding_cosine_misclassified = 0
        
        train_embeddings, train_labels, val_embeddings, val_labels, _, train_class_centers, class_embeddings = load_data(epoch, data_folder)
        #val_predictions = data["val_predictions"]

        center = np.zeros(len(train_class_centers[0]))
        for i in train_class_centers:
            center += i
        center = center / len(train_class_centers)

        eucledian_dist = lambda c, v : ((np.array(v) - np.array(c))**2).sum()
        

        print(end=".")
        for i, v in enumerate(train_embeddings):
            label = train_labels[i]

            clossest_ed_class = au.argmin([eucledian_dist(c,v) for c in train_class_centers])
            clossest_cod_class = au.argmin([cosine_origo_dist(c,v) for c in train_class_centers])

            if not is_pure(meta_data):
                clossest_ce_cod_class = au.argmin([cosine_origo_dist(c,v) for c in class_embeddings])
                class_embedding_cosine_misclassified += 0 if clossest_ce_cod_class == label else 1

            euclidean_misclassified += 0 if clossest_ed_class == label else 1
            cosine_origo_misclassified += 0 if clossest_cod_class == label else 1
            
        print(end=".")
        ed_acc.append(1.0 - euclidean_misclassified/len(train_embeddings))
        cod_acc.append(1.0 - cosine_origo_misclassified/len(val_embeddings))
        ce_cod_acc.append(1.0 - class_embedding_cosine_misclassified/len(val_embeddings))
        
    if is_pure(meta_data):
        return [i for i in epochs], {"eucledian": ed_acc, "cosine": cod_acc, "pure": [meta_data["accuracies"][i] for i in epochs]}
    else:
        return [i for i in epochs], {"eucledian": ed_acc, "cosine": cod_acc, "eucledian_ce": [meta_data["accuracies"][i] for i in epochs], "cosine_ce": ce_cod_acc}
    #plot.plot_line_2d([i for i in epochs], [ed_acc, cod_acc, [plot.round_scale(data["accuracies"][i]) for i in epochs]], 
    #                 ["Euclidean", "Cosine avg Center", "Pure"], x_label = "Epoch", y_label = "Classification Accuracy",
    #                  save_path= data_folder + "/acc_plot.png" if save else "") #Closest class center classification
    


def pca_scores(data_folder, save = False):
    meta_data = load_meta_data(data_folder)
    i = meta_data["config"]["max_epochs"] - 1
    train_embeddings, _, val_embeddings, _, _, _, _ = load_data(i, data_folder)

    scores = []
    xs = []
    done = False
    i = 5

    print("\n")
    while not done:
        pca = PCA(n_components = i)
        try:
            _ = pca.fit_transform(train_embeddings)
            scores.append(pca.score(val_embeddings))
            xs.append(i)
        except:
            done = True
        
        i += max(int(len(train_embeddings[0]) / 100), 1)
        print(f"PCA: {i}")
        
    return xs, scores
    #plot.plot_line_2d(xs, 
    #                  [scores], 
    #                  ["PCA Scores"], lambda x:x, "PCA Components", "PCA Score", data_folder + "/pca_plot" if save else "" )

def get_eucledian_and_cosine_dists(embeddings, labels, class_embeddings):
    eucledian_dists = []
    cosine_dists = []
    for i, embedding in enumerate(embeddings):
        class_embedding = class_embeddings[labels[i]]
        eucledian_dists.append(np.linalg.norm(np.array(class_embedding) - np.array(embedding)))
        cosine_dists.append(cosine_origo_dist(class_embedding, embedding))
    return eucledian_dists, cosine_dists

def median(l):
    return l[int(len(l) / 2)]

def median_dists(data_folder):
    median_to_class_centers_eucledian = []
    median_to_class_centers_cosine = []
    median_to_class_embeddings_eucledian = []
    median_to_class_embeddings_cosine = []

    meta_data = load_meta_data(data_folder)
    epochs = range(0, meta_data["config"]["max_epochs"])
    for epoch in epochs:
        print(epoch)
        train_embeddings, train_labels, val_embeddings, val_labels, val_class_centers, train_class_centers, class_embedings = load_data(epoch, data_folder)        
        val_eucledian_dists, val_cosine_dist = get_eucledian_and_cosine_dists(val_embeddings, val_labels, train_class_centers)

        val_eucledian_dists.sort()
        val_cosine_dist.sort()
        median_to_class_centers_eucledian.append(median(val_eucledian_dists))
        median_to_class_centers_cosine.append(median(val_eucledian_dists))
        
        if is_pure(meta_data):
            val_eucledian_dists, val_cosine_dist = get_eucledian_and_cosine_dists(val_embeddings, val_labels, class_embedings)

            val_eucledian_dists.sort()
            val_cosine_dist.sort()
            median_to_class_embeddings_eucledian.append(median(val_eucledian_dists))
            median_to_class_embeddings_cosine.append(median(val_eucledian_dists))

    if is_pure(meta_data):
        return [i for i in epochs], {"median_to_class_centers_eucledian": median_to_class_centers_eucledian, 
                                     "median_to_class_centers_cosine": median_to_class_centers_cosine}
    else:
        return [i for i in epochs], {"median_to_class_centers_eucledian": median_to_class_centers_eucledian, 
                                     "median_to_class_centers_cosine": median_to_class_centers_cosine,
                                     "median_to_class_embeddings_eucledian": median_to_class_embeddings_eucledian,
                                     "median_to_class_embeddings_cosine": median_to_class_embeddings_cosine}

    #plot.plot_line_2d([i for i in epochs], all_medians, [f"Class {i}" for i in range(10)], lambda x:x, 
    #                  "Epochs", "Mediuan Distance to Class Center", save_path=data_folder + "/median_plot.png" if save else "")
    #plot.plot_line_2d([i for i in epochs], all_cetc, [f"Class {i}" for i in range(10)], lambda x:x, 
    #                  "Epochs", "Class Center Distance to Avg Class Center", data_folder + "/cetc_plot.png" if save else "")

def get_avg_dist_to_avg_center(embeddings):
    center = np.zeros(len(embeddings[0]))
    for class_center in embeddings:
        center = center + np.array(class_center)
    
    center = center / len(embeddings)

    avg_dist = 0
    avg_cosine = 0
    for c in embeddings:
        avg_dist += np.linalg.norm(np.array(c) - center)
        avg_cosine += cosine_origo_dist(c, center)
    return avg_dist / len(embeddings), avg_cosine / len(embeddings)

def center_dists(data_folder):
    class_centers_to_avg_center_eucledian = []
    class_centers_to_avg_center_cosine = []
    class_embeddings_to_avg_center_eucledian = []
    class_embeddings_to_avg_center_cosine = []

    meta_data = load_meta_data(data_folder)
    epochs = range(0, meta_data["config"]["max_epochs"])
    for epoch in epochs:
        print(epoch)
        _, _, _, _, _, train_class_centers, class_embeddings = load_data(epoch, data_folder)        
        
        eucledian, cosine = get_avg_dist_to_avg_center(train_class_centers)
        class_centers_to_avg_center_eucledian.append(eucledian)
        class_centers_to_avg_center_cosine.append(cosine)
        if not is_pure(meta_data):
            eucledian, cosine = get_avg_dist_to_avg_center(class_embeddings)
            class_embeddings_to_avg_center_eucledian.append(eucledian)
            class_embeddings_to_avg_center_cosine.append(cosine)
        

    if is_pure(meta_data):
        return [i for i in epochs], {"class_centers_to_avg_center_eucledian": class_centers_to_avg_center_eucledian, 
                                     "class_centers_to_avg_center_cosine": class_centers_to_avg_center_cosine}
    else:
        return [i for i in epochs], {"class_centers_to_avg_center_eucledian": class_centers_to_avg_center_eucledian, 
                                     "class_centers_to_avg_center_cosine": class_centers_to_avg_center_cosine,
                                     "class_embeddings_to_avg_center_eucledian": class_embeddings_to_avg_center_eucledian,
                                     "class_embeddings_to_avg_center_cosine": class_embeddings_to_avg_center_cosine}



if __name__ == "__main__":

    folder = "cifar10_medium_embeddings"

    plot_data = {"name": folder, "meta_data": load_meta_data(folder), "acc": acc_by_dist_funcs(folder), 
                 "pca": pca_scores(folder), "medians": median_dists(folder), "center_dists": center_dists(folder)}
    fu.save_to_json("plotData", folder + ".json", plot_data)


    print("Done")