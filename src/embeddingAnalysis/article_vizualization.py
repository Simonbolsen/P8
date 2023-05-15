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
    class_size = [0 for _ in np.unique(np.array(labels))]
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

def cosine(c,v): 
    cnp = np.array(c)
    vnp = np.array(v)
    return vnp.dot(cnp) / (np.linalg.norm(vnp) * np.linalg.norm(cnp))

def acc_by_dist_funcs(data_folder, save = False): 
    ed_acc = []
    cod_acc = []
    ce_cod_acc = []

    meta_data = load_meta_data(data_folder)
    epochs = range(0, meta_data["config"]["max_epochs"])
    for epoch in epochs:
        print(f"\n{epoch + 1} / {meta_data['config']['max_epochs']}.", end="")
        euclidean_misclassified = 0
        cosine_origo_misclassified = 0
        class_embedding_cosine_misclassified = 0
        
        train_embeddings, train_labels, val_embeddings, val_labels, _, train_class_centers, class_embeddings = load_data(epoch, data_folder)
        #val_predictions = data["val_predictions"]

        print(end=".")
        center = np.zeros(len(train_class_centers[0]))
        for i in train_class_centers:
            center += i
        center = center / len(train_class_centers)

        train_embeddings = np.array(train_embeddings)
        train_class_centers = np.array(train_class_centers)

        count = 0
        step = int(len(train_embeddings) / 1000)
        for i in range(0, len(train_embeddings), step):
            count += np.count_nonzero(np.argmin(np.sum((train_embeddings[i:i+step, np.newaxis, :] - train_class_centers[np.newaxis, :, :]) ** 2, axis=-1), axis=1)  == np.array(train_labels[i:i+step]))

        ed_acc.append(count / len(train_embeddings)) 

        normalized_train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1)[:, np.newaxis]
        normalized_train_class_centers = train_class_centers / np.linalg.norm(train_class_centers, axis=1)[:, np.newaxis]

        cod_acc.append(np.count_nonzero(np.argmax(np.dot(normalized_train_embeddings, np.transpose(normalized_train_class_centers)), axis=1) == np.array(train_labels)) / len(train_embeddings))

        if not is_pure(meta_data):
            normalized_class_embeddings = np.array(class_embeddings)
            normalized_class_embeddings = normalized_class_embeddings / np.linalg.norm(normalized_class_embeddings, axis=1)[:, np.newaxis]
            ce_cod_acc.append(np.count_nonzero(np.argmax(np.dot(normalized_train_embeddings, np.transpose(normalized_class_embeddings)), axis=1) == np.array(train_labels)) / len(train_embeddings))
        
    if is_pure(meta_data):
        return [i for i in epochs], {"eucledian": ed_acc, "cosine": cod_acc, "pure": [meta_data["accuracies"][i] for i in epochs]}
    else:
        return [i for i in epochs], {"eucledian": ed_acc, "cosine": cod_acc, "eucledian_ce": [meta_data["accuracies"][i] for i in epochs], "cosine_ce": ce_cod_acc}
      
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
   
def get_eucledian_and_cosine_dists(embeddings, labels, class_embeddings):
    eucledian_dists = []
    cosine_dists = []
    for i, embedding in enumerate(embeddings):
        class_embedding = class_embeddings[labels[i]]
        eucledian_dists.append(np.linalg.norm(np.array(class_embedding) - np.array(embedding)))
        cosine_dists.append(cosine(class_embedding, embedding))
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
        print(f"Median Dists {epoch}")
        train_embeddings, train_labels, val_embeddings, val_labels, val_class_centers, train_class_centers, class_embedings = load_data(epoch, data_folder)        
        val_eucledian_dists, val_cosine_dist = get_eucledian_and_cosine_dists(val_embeddings, val_labels, train_class_centers)

        val_eucledian_dists.sort()
        val_cosine_dist.sort()
        median_to_class_centers_eucledian.append(median(val_eucledian_dists))
        median_to_class_centers_cosine.append(median(val_cosine_dist))
        
        if not is_pure(meta_data):
            val_eucledian_dists, val_cosine_dist = get_eucledian_and_cosine_dists(val_embeddings, val_labels, class_embedings)

            val_eucledian_dists.sort()
            val_cosine_dist.sort()
            median_to_class_embeddings_eucledian.append(median(val_eucledian_dists))
            median_to_class_embeddings_cosine.append(median(val_cosine_dist))

    if is_pure(meta_data):
        return [i for i in epochs], {"median_to_class_centers_eucledian": median_to_class_centers_eucledian, 
                                     "median_to_class_centers_cosine": median_to_class_centers_cosine}
    else:
        return [i for i in epochs], {"median_to_class_centers_eucledian": median_to_class_centers_eucledian, 
                                     "median_to_class_centers_cosine": median_to_class_centers_cosine,
                                     "median_to_class_embeddings_eucledian": median_to_class_embeddings_eucledian,
                                     "median_to_class_embeddings_cosine": median_to_class_embeddings_cosine}

def get_avg_dist_to_avg_center(embeddings):
    center = np.zeros(len(embeddings[0]))
    for class_center in embeddings:
        center = center + np.array(class_center)
    
    center = center / len(embeddings)

    avg_dist = 0
    avg_cosine = 0
    for c in embeddings:
        avg_dist += np.linalg.norm(np.array(c) - center)
        avg_cosine += cosine(c, center)
    return avg_dist / len(embeddings), avg_cosine / len(embeddings)

def center_dists(data_folder):
    class_centers_to_avg_center_eucledian = []
    class_centers_to_avg_center_cosine = []
    class_embeddings_to_avg_center_eucledian = []
    class_embeddings_to_avg_center_cosine = []

    meta_data = load_meta_data(data_folder)
    epochs = range(0, meta_data["config"]["max_epochs"])
    for epoch in epochs:
        print(f"Center Dists: {epoch}")
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

def gather_data(input_folders, data_folder):
    for folder in input_folders:
        plot_data = {"name": folder, "meta_data": load_meta_data(folder), "acc": acc_by_dist_funcs(folder), 
                    "pca": pca_scores(folder), "medians": median_dists(folder), "center_dists": center_dists(folder)}
        fu.save_to_json(data_folder, folder + ".json", plot_data)

def plot_data(data_folder, save_path = ""):

    all_files = fu.read_all_json_files(data_folder)
    
    classifiers = ['eucledian', 'cosine', 'eucledian_ce', 'cosine_ce', 'pure']
    labels = ['Eucledian', 'Cosine', 'Eucledian Class Embeddings', 'Cosine Class Embeddings', 'Pure']
    networks = ["resnet18", "resnet50", "resnet101"]
    loss_functions = ["cross_entropy", "simple-dist", "class-push", "cosine-loss"]

    file_by_dataset = {}

    for file in all_files:
        dataset = file["meta_data"]["config"]["dataset"]
        if dataset in file_by_dataset:
            file_by_dataset[dataset].append(file)
        else:
            file_by_dataset[dataset] = [file]
            path = save_path + "/" + dataset
            if not os.path.exists(path):
                print(f"plot folder for {dataset} created")
                os.mkdir(path)

    for dataset, files in file_by_dataset.items():
        acc_by_l_n_c = [[[0 for _ in classifiers] for _ in networks] for _ in loss_functions]
        xs_acc, ys_acc, ls_acc = [], [], []
        xs_pca, ys_pca, ls_pca = [], [], []
        xs_med_eu, ys_med_eu, ls_med_eu = [], [], []
        xs_med_co, ys_med_co, ls_med_co = [], [], []
        xs_cen_eu, ys_cen_eu, ls_cen_eu = [], [], []
        xs_cen_co, ys_cen_co, ls_cen_co = [], [], []


        for f, file in enumerate(files):

            x, y = file["acc"]
            config = file["meta_data"]["config"]
            for key, item in y.items():
                l, n, c = loss_functions.index(config["loss_func"]), networks.index(config["model_name"]), classifiers.index(key)
                maximum = max(item)
                if maximum > acc_by_l_n_c[l][n][c]:
                    acc_by_l_n_c[l][n][c] = maximum
                xs_acc.append(x)
                ys_acc.append(item)
                ls_acc.append(f'{file["name"]} {key}')

            x, y = file["pca"]
            for i, num in enumerate(y):
                if num < 0:
                    y = y[:i]
                    x = x[:i]
                    break

            xs_pca.append(x)
            ys_pca.append(y)
            ls_pca.append(f'{file["name"]}')
                
            x, y = file["medians"]
            for key, item in y.items():
                if "cosine" in key:
                    xs_med_co.append(x)
                    ys_med_co.append(item)
                    ls_med_co.append(f'{file["name"]} {key}')
                else:
                    xs_med_eu.append(x)
                    ys_med_eu.append(item)
                    ls_med_eu.append(f'{file["name"]} {key}')

            x, y = file["center_dists"]
            for key, item in y.items():
                if "cosine" in key:
                    xs_cen_co.append(x)
                    ys_cen_co.append(item)
                    ls_cen_co.append(f'{file["name"]} {key}')
                else:
                    xs_cen_eu.append(x)
                    ys_cen_eu.append(item)
                    ls_cen_eu.append(f'{file["name"]} {key}')

        dataset_path = save_path+"/"+dataset
        plot.plot_nested_bars(acc_by_l_n_c, loss_functions, labels, "Loss Functions", "Accuracy a")
        plot.plot_line_series_2d(xs_acc, ys_acc, ls_acc, "Epoch ep", "Accuracy a", save_path=dataset_path+"/acc")
        plot.plot_line_series_2d(xs_pca, ys_pca, ls_pca, "PCA Componments pc", "Score s", save_path=dataset_path+"/pca")
        plot.plot_line_series_2d(xs_med_eu, ys_med_eu, ls_med_eu, "Epoch ep", "Median Eucledian Distance ed", save_path=dataset_path+"/med_euc")
        plot.plot_line_series_2d(xs_med_co, ys_med_co, ls_med_co, "Epoch ep", "Median Cosine Similarity cs", save_path=dataset_path+"/med_cos")
        plot.plot_line_series_2d(xs_cen_eu, ys_cen_eu, ls_cen_eu, "Epoch ep", "Center Eucledian Distance ed", save_path=dataset_path+"/cen_euc")
        plot.plot_line_series_2d(xs_cen_co, ys_cen_co, ls_cen_co, "Epoch ep", "Center Cosine Similarity cs", save_path=dataset_path+"/cen_cos")

    print("Plotted")

if __name__ == "__main__":
    
    #"cl_embed_push_res_small_fashion_BEST",
    #"cl_embed_simple_res_med_fashion_BEST",
    #"cl_embed_cosine_res_med_fashion_BEST",
    #"cl_embed_push_res_med_fashion_BEST",
    #"cl_embed_simple_res_large_fashion_BEST",
    #"cl_embed_cosine_res_large_fashion_BEST",
    #"cl_embed_push_res_large_fashion_BEST",
    #"cl_embed_push_res_large_cifar_100_BEST",

    input_folders = [
        "cl_embed_simple_res_large_cifar_100_BEST",
        "cl_pure_res_small_cifar_10_BEST",
        "cl_pure_res_med_cifar_10_BEST",
        "cl_pure_res_large_cifar_10_BEST",
        "cl_embed_simple_res_small_cifar_10_BEST",
        "cl_embed_simple_res_large_cifar_10_BEST",
        "cl_embed_cosine_res_large_cifar_10_BEST",
        "cl_embed_push_res_large_cifar_10_BEST",
        "cl_embed_cosine_res_small_cifar_10_BEST",
        "cl_embed_push_res_small_cifar_10_BEST",
        "cl_embed_simple_res_med_cifar_10_BEST",
        "cl_embed_cosine_res_med_cifar_10_BEST",
        "cl_embed_push_res_med_cifar_10_BEST",
        "cl_pure_res_small_fashion_mnist_BEST",
        "cl_pure_res_med_fashion_mnist_BEST",
        "cl_pure_res_large_fashion_mnist_BEST",
        "cl_embed_simple_res_small_fashion_BEST",
        "cl_embed_cosine_res_small_fashion_BEST"]
    data_folder = "plots/plotData"
    plot_folder = "plots"
    #gather_data(input_folders, data_folder)
    plot_data(data_folder, plot_folder) #Without a save path the plots are shown and not saved

    print("Done")