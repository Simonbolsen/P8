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

def pca_pure_classification():
    i = 29
    data = fu.read_pickle_file(os.path.dirname(__file__) + f"/../embeddingData/classification_data_train_{i}.json")
    #print(f"{i}: {data['accuracy']}")
    train_embeddings = data["train_embeddings"]

    points = 100

    f = lambda x : int(((x + 1)/points)**2 * (1725-points) + x + 1)

    scores = []
    for i in range(points):
        pca = PCA(n_components = f(i))
        _ = pca.fit_transform(train_embeddings)
        scores.append(pca.score(train_embeddings))
        if i % int(points / 10) == 0:
            print(i)

    plot.plot_line_2d([f(i) for i in range(points)], 
                      [scores], 
                      ["PCA Scores"], lambda x:x)
    
def get_class_centers(embeddings, labels, num_of_classes):
    class_center_train = [np.zeros(len(embeddings[0])) for _ in range (num_of_classes)]
    class_size = [0 for _ in range(10)]
    for i, e in enumerate(embeddings):
        label = labels[i]
        npe = np.array(e)
        class_center_train[label] = class_center_train[label] + npe
        class_size[label] += 1

    return [e / class_size[i] for i, e in enumerate(class_center_train)]

def plot_pure_distances_to_class_centers():
    all_dists = []

    for epoch in range(30):
        print(epoch)
        train_embeddings, train_labels, test_embeddings, test_labels = load_pure_data(epoch)
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

def load_pure_data(epoch) :
    data = fu.read_pickle_file(os.path.dirname(__file__) + f"/../embeddingData/classification_data_train_{epoch}.json")
    train_embeddings = data["train_embeddings"]
    train_labels = data["train_labels"]

    data = fu.read_pickle_file(os.path.dirname(__file__) + f"/../embeddingData/classification_data_val_{epoch}.json")
    test_embeddings = data["val_embeddings"]
    test_labels = data["val_labels"]

    class_centers = fu.read_pickle_file(os.path.dirname(__file__) + f"/../embeddingData/class_centers.p")[epoch]
    return train_embeddings, train_labels, test_embeddings, test_labels, data, class_centers["test"], class_centers["train"]

def plot_pure():
    best_epoch = 29
    train_embeddings, train_labels, test_embeddings, test_labels = load_pure_data(best_epoch)

    pca = PCA(n_components = 3)
    train_projections = pca.fit_transform(train_embeddings)
    test_projections  = pca.transform(test_embeddings)

    COLOR = plot.get_colors(10)
    series = [{"marker": ".", "color": COLOR[i % 10], "label": f"{i%10}", "points":[]} for i in range(10)]

    class_embeddings = get_class_centers(test_embeddings, test_labels, 10)

    misclassified = 0
    for i, v in enumerate(test_embeddings):
        label = test_labels[i]

        clossest_class = au.argmin([((np.array(v) - np.array(c))**2).sum() for c in class_embeddings])
        index = 0 if clossest_class == label else 1
        misclassified += index
        
        series[label]["points"].append(train_projections[i])

    print(f"{misclassified}/{len(test_embeddings)}, acc: {100.0 * (1.0 - misclassified/len(test_embeddings)):.2f}%")
    plot.plotCustomPoints(series, legend=True, axes=[0,1,2])
        
def plot_pure_distance_distribution():
    test_buckets = []
    train_buckets = []
    all_test_dists = []
    all_train_dists = []
    maximum = -float("inf")
    minimum = float("inf")
    r = range(1, 30, 7)
    for epoch in r:
        print(epoch)
        train_embeddings, train_labels, test_embeddings, test_labels, data, test_class_embeddings, train_class_embeddings = load_pure_data(epoch)

        test_dists = list(filter(lambda x : x < 60, au.get_dists(test_embeddings, test_labels, train_class_embeddings, lambda x : x)))
        train_dists = list(filter(lambda x : x < 60, au.get_dists(train_embeddings, train_labels, train_class_embeddings, lambda x : x)))

        all_test_dists.append(test_dists)
        all_train_dists.append(train_dists)
        num_buckets = 100

        maximum = max(max(test_dists), max(train_dists), maximum)
        minimum = min(min(test_dists), min(train_dists), minimum)

    for i in range(len(all_test_dists)):
        test_buckets.append(au.get_buckets(all_test_dists[i], maximum, minimum, num_buckets))
        train_buckets.append(au.get_buckets(all_train_dists[i], maximum, minimum, num_buckets))

    #plot.plot_line_2d([(i * (maximum - minimum) / (num_buckets) + minimum) for i in range(num_buckets)], 
    #                  test_buckets, 
    #                  [f"Testing epoch {i}" for i in r], lambda x:x)

    #plot.plotSurface([test_buckets, train_buckets], "Dist Frequency", [i for i in range(30)], "Epoch", 
    #                 [(i * (maximum - minimum) / (num_buckets) + minimum) for i in range(num_buckets)], "Dist", 1, ["Test", "Train"])

def plot_acc_by_epoch(): 
    test_acc = []
    train_acc = []
    acc = []
    for epoch in range(30):
        print(f"\n{epoch + 1} / 30.", end="")
        test_misclassified = 0
        train_misclassified = 0
        
        train_embeddings, train_labels, test_embeddings, test_labels, data, test_class_embeddings, train_class_embeddings = load_pure_data(epoch)
        print(end=".")
        for i, v in enumerate(test_embeddings):
            dist_func = lambda c, v : ((np.array(v) - np.array(c))**2).sum()

            clossest_test_class = au.argmin([dist_func(c,v) for c in test_class_embeddings])
            clossest_train_class = au.argmin([dist_func(c,v) for c in train_class_embeddings])
            test_misclassified += 0 if clossest_test_class == test_labels[i] else 1
            train_misclassified += 0 if clossest_train_class == test_labels[i] else 1
        print(end=".")
        test_acc.append(100.0 * (1.0 - test_misclassified/len(test_embeddings)))
        train_acc.append(100.0 * (1.0 - train_misclassified/len(test_embeddings))) #Yes, both must divide by test
        acc.append(100.0 * data["accuracy"])
    
    plot.plot_line_2d([i for i in range(30)], [test_acc, train_acc, acc], ["Test C4", "Train C4", "Pure"], lambda x:x) #Closest class center classification

def plot_acc_by_dist_funcs_and_epoch(): 
    ed_acc = []
    cod_acc = []
    cacd_acc = []
    acc = []
    epochs = range(0, 30)
    for epoch in epochs:
        print(f"\n{epoch + 1} / 30.", end="")
        euclidean_misclassified = 0
        cosine_origo_misclassified = 0
        cosine_avg_center_misclassified = 0
        
        train_embeddings, train_labels, test_embeddings, test_labels, data, test_class_embeddings, train_class_embeddings = load_pure_data(epoch)

        center = np.zeros(len(train_class_embeddings[0]))
        for i in train_class_embeddings:
            center += i
        center = center / len(train_class_embeddings)

        print(end=".")
        for i, v in enumerate(test_embeddings):
            label = test_labels[i]

            eucledian_dist = lambda c, v : ((np.array(v) - np.array(c))**2).sum()
            def cosine_origo_dist(c,v): 
                cnp = np.array(c)
                vnp = np.array(v)
                return 1 - vnp.dot(cnp) / (np.linalg.norm(vnp) * np.linalg.norm(cnp))

            def cosine_avg_center_dist(c, b):
                cnp = np.array(c) - center
                vnp = np.array(v) - center
                return 1 - vnp.dot(cnp) / (np.linalg.norm(vnp) * np.linalg.norm(cnp))

            clossest_ed_class = au.argmin([eucledian_dist(c,v) for c in train_class_embeddings])
            clossest_cod_class = au.argmin([cosine_origo_dist(c,v) for c in train_class_embeddings])
            clossest_cacd_class = au.argmin([cosine_avg_center_dist(c,v) for c in train_class_embeddings])

            euclidean_misclassified += 0 if clossest_ed_class == label else 1
            cosine_origo_misclassified += 0 if clossest_cod_class == label else 1
            cosine_avg_center_misclassified += 0 if clossest_cacd_class == label else 1

        print(end=".")
        ed_acc.append(100.0 * (1.0 - euclidean_misclassified/len(test_embeddings)))
        cod_acc.append(100.0 * (1.0 - cosine_origo_misclassified/len(test_embeddings))) 
        cacd_acc.append(100.0 * (1.0 - cosine_avg_center_misclassified/len(test_embeddings))) 
        acc.append(100.0 * data["accuracy"])
    
    plot.plot_line_2d([i for i in epochs], [ed_acc, cod_acc, cacd_acc, acc], ["Euclidean", "Cosine Origo", "Cosine avg Center", "Pure"], lambda x:x) #Closest class center classification

def plot_pure_distance_distribution():
    num_buckets = 30
    minimum = 0
    maximum = 60
    test_dists = []
    test_buckets = [[] for _ in range(10)] #label x epoch x dists
    r = range(1, 30, 14)
    for epoch in r:
        print(epoch)
        train_embeddings, train_labels, test_embeddings, test_labels, data, test_class_embeddings, train_class_embeddings = load_pure_data(epoch)
        
        test_dists.append(au.get_dists_by_label(test_embeddings, test_labels, train_class_embeddings))
        #train_dists = get_dists_by_label(train_embeddings, train_labels, train_class_embeddings)

    for epoch, dists_by_label in enumerate(test_dists):
        print(epoch)
        for i, label in enumerate(dists_by_label):
            label = list(filter(lambda x: x <= maximum, label))
            test_buckets[i].append(au.get_buckets(label, maximum, minimum, num_buckets))

    plot.plotSurface(test_buckets, "Dist Frequency", [i for i in r], "Epochs", 
                    [(n * (maximum - minimum) / (num_buckets) + minimum) for n in range(num_buckets)], 
                    "Dist Buckets", 3, [f"Class {i}" for i in range(len(test_buckets))])

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
        

def plot_pure_dist_median_iqr():
    all_medians = [[] for _ in range(10)]
    all_cetc = [[] for _ in range(10)]
    all_epochs = [[] for _ in range(10)]

    epochs = range(1, 30)
    for epoch in epochs:
        print(epoch)
        train_embeddings, train_labels, test_embeddings, test_labels, data, test_class_embeddings, train_class_embeddings = load_pure_data(epoch)        
        test_dists = au.get_dists_by_label(test_embeddings, test_labels, train_class_embeddings)
        #test_dists = au.get_dists(train_embeddings, train_labels, train_class_embeddings, lambda x : x)
        #test_dists.sort()

        center = np.zeros(len(train_class_embeddings[0]))
        for class_center in train_class_embeddings:
            center = center + np.array(class_center)
        
        center = center / len(train_class_embeddings)

        for i, label_i_dists in enumerate(test_dists):
            label_i_dists.sort()
            median = label_i_dists[int(len(label_i_dists)/2)]
            all_medians[i].append(median)
            all_cetc[i].append(np.linalg.norm(np.array(train_class_embeddings[i]) - center))
            all_epochs[i].append(epoch)

    #plot.plotPoints(all_epochs, all_cetc, all_medians, ["Epoch", "Class center to Avg Center dist", "Median"], num_of_series=10, 
    #                series_labels=[f"Class {i}" for i in range(10)], function=lambda x:x, marker="-")
    plot.plot_points_series_2d(all_cetc, all_medians, "Dist to avg center", "Median dists", [f"Class {i}" for i in range(10)])

    #plot.plot_line_2d([i for i in epochs], all_medians, [f"Class {i}" for i in range(10)], lambda x:x)
    #plot.plot_line_2d([i for i in epochs], all_medians[:1], ["Median Dist All classes"], lambda x:x)

def print_stats():
    train_embeddings, train_labels, test_embeddings, test_labels, data, test_class_embeddings, train_class_embeddings = load_pure_data(29)    

    au.print_distance_matrix(train_class_embeddings)
    au.print_distances_from_center(train_class_embeddings)

def kmeans():
    cluster_range = range(5, 30, 2)
    epoch = 29
    scores = []
    for n in cluster_range:
        train_embeddings, train_labels, test_embeddings, test_labels, data, test_class_embeddings, train_class_embeddings = load_pure_data(epoch)   

        kmeans = KMeans(n_clusters= n, random_state=0, n_init="auto").fit(np.array(train_embeddings))
        scores.append(kmeans.score(np.array(test_embeddings)))

    plot.plot_line_2d([i for i in cluster_range], [scores], ["Score"], lambda x:x)
    #au.print_distance_matrix(train_class_embeddings, kmeans.cluster_centers_)

if __name__ == '__main__':
    #plot_acc_by_epoch()
    #plot_pure_distance_distribution()
    #plot_pure_dist_median_iqr()
    #print_stats()
    
    #plot_acc_by_epoch()
    #plot_acc_by_dist_funcs_and_epoch()

    kmeans()
    
    print("Done")