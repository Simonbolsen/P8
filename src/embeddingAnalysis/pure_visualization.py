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

def load_mtm_data(epoch, data_folder):
    print("TODO")

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
        i += int(1)
        _ = pca.fit_transform(train_embeddings)
        try:
            xs.append(i)
            scores.append(pca.score(train_embeddings))
        except:
            done = True
        
    plot.plot_line_2d(xs, 
                      [scores], 
                      ["PCA Scores"], lambda x:x, "PCA Components", "PCA Score", data_folder + "/pca_plot" if save else "" )
    
def plot_pure_distances_to_class_centers():
    all_dists = []

    for epoch in range(30):
        print(epoch)
        train_embeddings, train_labels, val_embeddings, val_labels = load_pure_data(epoch)
        class_embeddings = get_class_centers(train_embeddings, train_labels, 10)

        center = np.zeros(len(class_embeddings[0]))
        for i in class_embeddings:
            center += i
        center = center / len(class_embeddings)
        
        dists = []

        for i in class_embeddings:
            dists.append(np.linalg.norm(center - i))

        all_dists.append(dists)

    plot.plotSurface([all_dists], "Distance To Class Center", [i for i in range(30)], "Epochs", [i for i in range(10)], "Classes", 1, [""])

def plot_pure():
    best_epoch = 29
    train_embeddings, train_labels, val_embeddings, val_labels = load_pure_data(best_epoch)

    pca = PCA(n_components = 3)
    train_projections = pca.fit_transform(train_embeddings)
    val_projections  = pca.transform(val_embeddings)

    COLOR = plot.get_colors(10)
    series = [{"marker": ".", "color": COLOR[i % 10], "label": f"{i%10}", "points":[]} for i in range(10)]

    class_embeddings = get_class_centers(val_embeddings, val_labels, 10)

    misclassified = 0
    for i, v in enumerate(val_embeddings):
        label = val_labels[i]

        clossest_class = au.argmin([((np.array(v) - np.array(c))**2).sum() for c in class_embeddings])
        index = 0 if clossest_class == label else 1
        misclassified += index
        
        series[label]["points"].append(train_projections[i])

    print(f"{misclassified}/{len(val_embeddings)}, acc: {100.0 * (1.0 - misclassified/len(val_embeddings)):.2f}%")
    plot.plotCustomPoints(series, legend=True, axes=[0,1,2])
        
def plot_pure_distance_distribution():
    val_buckets = []
    train_buckets = []
    all_val_dists = []
    all_train_dists = []
    maximum = -float("inf")
    minimum = float("inf")
    r = range(1, 30, 7)
    for epoch in r:
        print(epoch)
        train_embeddings, train_labels, val_embeddings, val_labels, data, val_class_embeddings, train_class_embeddings = load_pure_data(epoch)

        val_dists = list(filter(lambda x : x < 60, au.get_dists(val_embeddings, val_labels, train_class_embeddings, lambda x : x)))
        train_dists = list(filter(lambda x : x < 60, au.get_dists(train_embeddings, train_labels, train_class_embeddings, lambda x : x)))

        all_val_dists.append(val_dists)
        all_train_dists.append(train_dists)
        num_buckets = 100

        maximum = max(max(val_dists), max(train_dists), maximum)
        minimum = min(min(val_dists), min(train_dists), minimum)

    for i in range(len(all_val_dists)):
        val_buckets.append(au.get_buckets(all_val_dists[i], maximum, minimum, num_buckets))
        train_buckets.append(au.get_buckets(all_train_dists[i], maximum, minimum, num_buckets))

    #plot.plot_line_2d([(i * (maximum - minimum) / (num_buckets) + minimum) for i in range(num_buckets)], 
    #                  val_buckets, 
    #                  [f"valing epoch {i}" for i in r], lambda x:x)

    #plot.plotSurface([val_buckets, train_buckets], "Dist Frequency", [i for i in range(30)], "Epoch", 
    #                 [(i * (maximum - minimum) / (num_buckets) + minimum) for i in range(num_buckets)], "Dist", 1, ["val", "Train"])

def plot_acc_by_epoch(): 
    val_acc = []
    train_acc = []
    acc = []
    for epoch in range(30):
        print(f"\n{epoch + 1} / 30.", end="")
        val_misclassified = 0
        train_misclassified = 0
        
        train_embeddings, train_labels, val_embeddings, val_labels, data, val_class_embeddings, train_class_embeddings = load_pure_data(epoch)
        print(end=".")
        for i, v in enumerate(val_embeddings):
            dist_func = lambda c, v : ((np.array(v) - np.array(c))**2).sum()

            clossest_val_class = au.argmin([dist_func(c,v) for c in val_class_embeddings])
            clossest_train_class = au.argmin([dist_func(c,v) for c in train_class_embeddings])
            val_misclassified += 0 if clossest_val_class == val_labels[i] else 1
            train_misclassified += 0 if clossest_train_class == val_labels[i] else 1
        print(end=".")
        val_acc.append(100.0 * (1.0 - val_misclassified/len(val_embeddings)))
        train_acc.append(100.0 * (1.0 - train_misclassified/len(val_embeddings))) #Yes, both must divide by val
        acc.append(100.0 * data["accuracy"])
    
    plot.plot_line_2d([i for i in range(30)], [val_acc, train_acc, acc], ["val C4", "Train C4", "Pure"], lambda x:x) #Closest class center classification

def plot_acc_by_dist_funcs_and_epoch(data_folder, save = False): 
    ed_acc = []
    cod_acc = []
    cacd_acc = []
    codun_acc = []
    cacdun_acc = []
    epochs = range(0, 30)
    ed_intersection = []
    cacd_intersection = []
    for epoch in epochs:
        print(f"\n{epoch + 1} / 30.", end="")
        euclidean_misclassified = 0
        cosine_origo_misclassified = 0
        cosine_avg_center_misclassified = 0
        cosine_origo_un_misclassified = 0
        cosine_avg_center_un_misclassified = 0
        ed_intersection_num = 0
        cacd_intersection_num = 0
        
        train_embeddings, train_labels, val_embeddings, val_labels, data, val_class_embeddings, train_class_embeddings = load_pure_data(epoch, data_folder)
        #val_predictions = data["val_predictions"]

        center = np.zeros(len(train_class_embeddings[0]))
        for i in train_class_embeddings:
            center += i
        center = center / len(train_class_embeddings)

        print(end=".")
        for i, v in enumerate(train_embeddings):
            label = train_labels[i]

            eucledian_dist = lambda c, v : ((np.array(v) - np.array(c))**2).sum()
            def cosine_origo_dist(c,v): 
                cnp = np.array(c)
                vnp = np.array(v)
                return -vnp.dot(cnp) / (np.linalg.norm(vnp) * np.linalg.norm(cnp))

            def cosine_avg_center_dist(c, b):
                cnp = np.array(c) - center
                vnp = np.array(v) - center
                return -vnp.dot(cnp) / (np.linalg.norm(vnp) * np.linalg.norm(cnp))

            def cosine_origo_un_dist(c,v): 
                cnp = np.array(c)
                vnp = np.array(v)
                return -vnp.dot(cnp)

            def cosine_avg_center_un_dist(c, b):
                cnp = np.array(c) - center
                vnp = np.array(v) - center
                return -vnp.dot(cnp)

            clossest_ed_class = au.argmin([eucledian_dist(c,v) for c in train_class_embeddings])
            #clossest_cod_class = au.argmin([cosine_origo_dist(c,v) for c in train_class_embeddings])
            clossest_cacd_class = au.argmin([cosine_avg_center_dist(c,v) for c in train_class_embeddings])
            #clossest_codun_class = au.argmin([cosine_origo_un_dist(c,v) for c in train_class_embeddings])
            #clossest_cacdun_class = au.argmin([cosine_avg_center_un_dist(c,v) for c in train_class_embeddings])

            euclidean_misclassified += 0 if clossest_ed_class == label else 1
            #cosine_origo_misclassified += 0 if clossest_cod_class == label else 1
            cosine_avg_center_misclassified += 0 if clossest_cacd_class == label else 1
            #cosine_origo_un_misclassified += 0 if clossest_codun_class == label else 1
            #cosine_avg_center_un_misclassified += 0 if clossest_cacdun_class == label else 1
            #ed_intersection_num += 1 if au.argmax(val_predictions[i]) == clossest_ed_class else 0
            #cacd_intersection_num += 1 if au.argmax(val_predictions[i]) == clossest_cacd_class else 0

        print(end=".")
        ed_acc.append(plot.round_scale(1.0 - euclidean_misclassified/len(train_embeddings)))
        #cod_acc.append(plot.round_scale(1.0 - cosine_origo_misclassified/len(val_embeddings)))
        cacd_acc.append(plot.round_scale(1.0 - cosine_avg_center_misclassified/len(train_embeddings)))
        #codun_acc.append(plot.round_scale(1.0 - cosine_origo_un_misclassified/len(val_embeddings)))
        #cacdun_acc.append(plot.round_scale(1.0 - cosine_avg_center_un_misclassified/len(val_embeddings)))
        #ed_intersection.append(ed_intersection_num / len(val_embeddings))
        #cacd_intersection.append(cacd_intersection_num / len(val_embeddings))
    
    plot.plot_line_2d([i for i in epochs], [ed_acc, cacd_acc, [plot.round_scale(data["accuracies"][i]) for i in epochs]], 
                      ["Euclidean", "Cosine avg Center", "Pure"], x_label = "Epoch", y_label = "Classification Accuracy",
                      save_path= data_folder + "/acc_plot.png" if save else "") #Closest class center classification
    #plot.plot_line_2d([i for i in epochs], [ed_acc, cod_acc, cacd_acc, codun_acc, cacdun_acc, [plot.round_scale(data["accuracies"][i]) for i in epochs]], 
    #                  ["Euclidean", "Cosine Origo", "Cosine avg Center","Cosine Origo UN", "Cosine avg Center UN", "Pure"]) #Closest class center classification
    #plot.plot_line_2d([i for i in epochs], [ed_intersection, cacd_intersection], 
    #                  ["Euclidean and Pure Intersection", "Cosine avg Center and Pure Intersection"], 
    #                  lambda x:x, "Epoch", "Proportion of Prediction Intersection", data_folder + "/intersection_plot.png" if save else "")

def plot_pure_distance_distribution():
    num_buckets = 30
    minimum = 0
    maximum = 60
    val_dists = []
    val_buckets = [[] for _ in range(10)] #label x epoch x dists
    r = range(1, 30, 14)
    for epoch in r:
        print(epoch)
        train_embeddings, train_labels, val_embeddings, val_labels, data, val_class_embeddings, train_class_embeddings = load_pure_data(epoch)
        
        val_dists.append(au.get_dists_by_label(val_embeddings, val_labels, train_class_embeddings))
        #train_dists = get_dists_by_label(train_embeddings, train_labels, train_class_embeddings)

    for epoch, dists_by_label in enumerate(val_dists):
        print(epoch)
        for i, label in enumerate(dists_by_label):
            label = list(filter(lambda x: x <= maximum, label))
            val_buckets[i].append(au.get_buckets(label, maximum, minimum, num_buckets))

    plot.plotSurface(val_buckets, "Dist Frequency", [i for i in r], "Epochs", 
                    [(n * (maximum - minimum) / (num_buckets) + minimum) for n in range(num_buckets)], 
                    "Dist Buckets", 3, [f"Class {i}" for i in range(len(val_buckets))])

def get_median_and_iqr(sorted_list):
    half = int(len(sorted_list)/2)
    median = sorted_list[half]
    i = half - 1
    j = half + 1
    while j - i - 2 < half:
        if i < 0:
            j +=1
        elif j >= len(sorted_list):
            i -= 1
        elif median - sorted_list[i] < sorted_list[j] - median:
            i -= 1
        else:
            j += 1
    return median, min(median - sorted_list[i], sorted_list[j] - median)
        
def plot_pure_dist_median(data_folder, save = False):
    all_medians = [[] for _ in range(10)]
    all_cetc = [[] for _ in range(10)]
    all_epochs = [[] for _ in range(10)]

    epochs = range(1, 20)
    for epoch in epochs:
        print(epoch)
        train_embeddings, train_labels, val_embeddings, val_labels, data, val_class_embeddings, train_class_embeddings = load_pure_data(epoch, data_folder)        
        val_dists = au.get_dists_by_label(val_embeddings, val_labels, train_class_embeddings)
        #val_dists = au.get_dists(train_embeddings, train_labels, train_class_embeddings, lambda x : x)
        #val_dists.sort()

        center = np.zeros(len(train_class_embeddings[0]))
        for class_center in train_class_embeddings:
            center = center + np.array(class_center)
        
        center = center / len(train_class_embeddings)

        for i, label_i_dists in enumerate(val_dists):
            label_i_dists.sort()
            median = label_i_dists[int(len(label_i_dists)/2)]
            all_medians[i].append(median)
            all_cetc[i].append(np.linalg.norm(np.array(train_class_embeddings[i]) - center))
            all_epochs[i].append(epoch)

    #plot.plotPoints(all_epochs, all_cetc, all_medians, ["Epoch", "Class center to Avg Center dist", "Median"], num_of_series=10, 
    #                series_labels=[f"Class {i}" for i in range(10)], function=lambda x:x, marker="-")
    #plot.plot_points_series_2d(all_cetc, all_medians, "Dist to avg center", "Median dists", [f"Class {i}" for i in range(10)])

    plot.plot_line_2d([i for i in epochs], all_medians, [f"Class {i}" for i in range(10)], lambda x:x, 
                      "Epochs", "Mediuan Distance to Class Center", save_path=data_folder + "/median_plot.png" if save else "")
    plot.plot_line_2d([i for i in epochs], all_cetc, [f"Class {i}" for i in range(10)], lambda x:x, 
                      "Epochs", "Class Center Distance to Avg Class Center", data_folder + "/cetc_plot.png" if save else "")
    #plot.plot_line_2d([i for i in epochs], all_medians[:1], ["Median Dist All classes"], lambda x:x)

def print_stats():
    train_embeddings, train_labels, val_embeddings, val_labels, data, val_class_embeddings, train_class_embeddings = load_pure_data(29)    

    au.print_distance_matrix(train_class_embeddings)
    au.print_distances_from_center(train_class_embeddings)

def kmeans():
    cluster_range = range(5, 30, 2)
    epoch = 29
    scores = []
    for n in cluster_range:
        train_embeddings, train_labels, val_embeddings, val_labels, data, val_class_embeddings, train_class_embeddings = load_pure_data(epoch)   

        kmeans = KMeans(n_clusters= n, random_state=0, n_init="auto").fit(np.array(train_embeddings))
        scores.append(kmeans.score(np.array(val_embeddings)))

    plot.plot_line_2d([i for i in cluster_range], [scores], ["Score"], lambda x:x)
    #au.print_distance_matrix(train_class_embeddings, kmeans.cluster_centers_)

def confusion_dist_matrix(data_folder):
    path = os.path.dirname(__file__) + "/../../embeddingData/" + data_folder
    data = fu.read_pickle_file(path + f"/classification_data_train_{10}.p")
    train_class_embeddings = data["train_class_embeddings"]
    train_embeddings = data["train_embeddings"]
    print("")
    for i in range(len(train_class_embeddings)):
        print(f"  {i}", end="  ")
    print("")
    for i, ci in enumerate(train_class_embeddings):
        cia = np.array(ci)
        for ii, cii in enumerate(train_class_embeddings):
            count = 0
            ciia = np.array(cii)
            for e in train_embeddings:
                if ((np.array(e) - cia)**2).sum(0) <= ((np.array(e) - ciia)**2).sum(0) :
                    count += 1
            print(f" {(count / len(train_embeddings)):.2f}", end="")

        print("")

def print_centers():
    epoch = 5
    path = os.path.dirname(__file__) + "/../../embeddingData/" + data_folder
    data = fu.read_pickle_file(path + f"/classification_data_train_{epoch}.p")
    train_embeddings = data["train_embeddings"]
    class_embeddings = data["train_class_embeddings"]

    center_path = f"class_centers_{epoch}.p"
    class_centers = fu.read_pickle_file(path + "/" + center_path)["train"]

    avg_center = np.zeros(len(class_embeddings[0]))
    embeddings_center = np.zeros(len(class_embeddings[0]))

    for i in range(10):
        avg_center = avg_center + class_centers[i]
        embeddings_center = embeddings_center + class_embeddings[i]

    avg_center = np.array(avg_center / 10)
    embeddings_center = np.array(embeddings_center / 10)

    ac_norm = np.linalg.norm(avg_center)
    ec_norm = np.linalg.norm(embeddings_center)
    print(f"{np.dot(ac_norm, ec_norm) / ac_norm / ec_norm}" )
    print(f"{ac_norm}")
    print(f"{ec_norm}")
    print("")

    for i in range(10):
        print("")
        for ii in range(10):
            ce = np.array(class_embeddings[i] - embeddings_center)
            cc = np.array(class_centers[ii] - avg_center)
            cc_norm = np.linalg.norm(cc)
            ce_norm = np.linalg.norm(ce)
            print(f" {np.dot(ce, cc) / ce_norm / cc_norm:.2f}", end = "")

def find_error():
    a = []
    #for epoch in range(0, 1):
    epoch = 5
    path = os.path.dirname(__file__) + "/../../embeddingData/" + data_folder
    data = fu.read_pickle_file(path + f"/classification_data_train_{epoch}.p")
    class_embeddings = data["train_class_embeddings"]
    center_path = f"class_centers_{epoch}.p"
    class_centers = fu.read_pickle_file(path + "/" + center_path)["train"]
    
    pca = PCA(n_components = 3)
    pca_train_embeddings = pca.fit_transform(data["train_embeddings"])
    train_labels = data["train_labels"]

    data = fu.read_pickle_file(path + f"/classification_data_val_{epoch}.p")
    pca_val_embeddings = pca.transform(data["val_embeddings"])
    val_labels = data["val_labels"]
    pca_class_centers = pca.transform(class_centers)
    pca_class_embeddings = pca.transform(class_embeddings)

    avg_class_center = sum(pca_class_centers)/len(pca_class_centers)
    pca_class_centers = [(pcc - avg_class_center)/np.linalg.norm(pcc - avg_class_center) + avg_class_center for pcc in pca_class_centers]

    COLOR = plot.get_colors(10)
    series = [{"marker": ".", "color": COLOR[i], "label": f"{i}cc", 
               "points":[e for ei, e in enumerate(pca_train_embeddings) if train_labels[ei] == i]} for i in range(10)] + [{"marker": "*", "color": COLOR[i], "label": f"{i}ce", 
                                                               "points":pca_class_embeddings[i:i+1]} for i in range(10)]

    plot.plotCustomPoints(series, legend=False, axes=[0,1,2])

    print("break")

if __name__ == '__main__':
    #plot_acc_by_epoch()
    #plot_pure_distance_distribution()
    #plot_pure_dist_median_iqr()
    #print_stats()
    
    #plot_acc_by_epoch()
    #plot_acc_by_dist_funcs_and_epoch()

    #kmeans()


    #USEFUL:
    data_folder = "cl_embed_cone_res_med_cifar_10"#"cifar10_medium_pure_embeddings"
    save = True

    



    #train_embeddings, train_labels, val_embeddings, val_labels, data, val_class_embeddings, train_class_embeddings = load_pure_data(29, data_folder)
    #pca_pure_classification(data_folder, save)
    #plot_acc_by_dist_funcs_and_epoch(data_folder, save)
    #plot_pure_dist_median(data_folder, save)
    find_error()
    print("Done")