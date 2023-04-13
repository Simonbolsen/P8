import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import os.path
from sklearn.decomposition import PCA
import Plotting.plotting_util as plot
import numpy as np
import matplotlib.cm as cm
import math
import pickle

def read_json_file(path_to_json_file):
    with open(path_to_json_file, 'r') as json_file:
        data = json_file.read()
    return json.loads(json.loads(data))

def read_pickle_file(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
        return data

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]

def hyper_sphere_volume(radius, dimensions):
    if dimensions == 0:
        return 1
    if dimensions == 1:
        return 2 * radius
    return radius * radius * 2 * np.pi / dimensions * hyper_sphere_volume(radius, dimensions - 2)

train_embeddings = []
test_embeddings  = []
class_embeddings = []
test_labels = []
train_labels = []


def plot_pcas():
    scores = []
    accs = []
    for i in range(1, 60):
        pca = PCA(n_components = i)
        _ = pca.fit_transform(train_embeddings)
        scores.append(pca.score(train_embeddings))
        test_projections  = pca.transform(test_embeddings)
        class_projections = pca.transform(class_embeddings)
        if i % 5 == 0:
            print(i)
        misclassified = 0
        for i, v in enumerate(test_projections):
            label = data["test_labels"][i]
            clossest_class = argmin([((np.array(v) - np.array(c))**2).sum() for c in class_projections])
            index = 0 if clossest_class == label else 1
            misclassified += index
        acc = 1 - misclassified / len(test_embeddings)
        accs.append(1-math.sqrt(1 - acc**2))

    plot.plot_2d(accs)
    plot.plot_2d(scores)



def print_distance_matrix(embeddings):

    print("")
    for i in range(len(embeddings)):
        print(f"  {i}", end="  ")
    print("")
    for index, i in enumerate(embeddings):
        print(f"{index}", end=" ")
        for ii in embeddings:
            print(f"{np.linalg.norm(np.array(i) - np.array(ii)):.2f}",end=" ")
        print("")

def print_distances_from_center(embeddings):
    

    center = np.zeros(len(embeddings[0]))
    for i in embeddings:
        center += i
    center = center / len(embeddings)
    print()
    for i in embeddings:
        print(np.linalg.norm(center - i))

def plot_series():
    pca = PCA(n_components = 3)
    train_projections = pca.fit_transform(train_embeddings)
    test_projections  = pca.transform(test_embeddings)
    class_projections = pca.transform(class_embeddings)

    COLOR = plot.get_colors(10)
    series = [{"marker": "." if i < 10 else "*", "color": COLOR[i % 10], "label": f"{i%10} {'C' if i < 10 else 'Misc'}lassified", "points":[]} for i in range(20)]

    misclassified = 0
    for i, v in enumerate(test_embeddings):
        label = data["test_labels"][i]

        clossest_class = argmin([((np.array(v) - np.array(c))**2).sum() for c in class_embeddings])
        index = 0 if clossest_class == label else 1
        misclassified += index
        
        series[label + 10 * index]["points"].append(test_projections[i])

    print(f"{misclassified}/{len(test_embeddings)}, acc: {100.0 * (1.0 - misclassified/len(test_embeddings)):.2f}%")
    plot.plotCustomPoints(series, legend=True, axes=[0,1,2])
    #plot.plotCustomPoints(series, legend=False, axes=[0,1,3])
    #plot.plotCustomPoints(series, legend=False, axes=[1,2,3])
    #plot.plotCustomPoints(series, legend=False, axes=[0,2,3])

def get_dists(embeddings, labels, class_embeddings, func = lambda x: np.log(x)):
    dists = []
    for i, embedding in enumerate(embeddings):
        class_embedding = class_embeddings[labels[i]]
        dists.append(func(np.linalg.norm(np.array(class_embedding) - np.array(embedding))))
    return dists

def get_buckets(dists, maximum, minimum, num_buckets):
    buckets = [0 for _ in range(num_buckets)]
    for d in dists:
        index = int((d - minimum) * (num_buckets - 1) / (maximum - minimum))
        buckets[index] += 1/len(dists)
    return buckets

def get_sim_dists(dims, radius, samples):
    dists = []
    while len(dists) < samples:
        points = np.random.rand(dims * samples, dims) * 2 - np.full([dims * samples, dims], 1)
        for p in points:
            dist = np.linalg.norm(p)
            if dist < 1:
                dists.append(np.log(dist * radius))
        print(f"{len(dists)}/{samples}")
    return dists

def plot_distance_distribution():
    test_dists = get_dists(test_embeddings, test_labels, class_embeddings)
    train_dists = get_dists(train_embeddings, train_labels, class_embeddings)
    #sim_dists = get_sim_dists(4, 0.25, 10000)
    num_buckets = 100

    maximum = max(max(test_dists), max(train_dists))
    minimum = min(min(test_dists), min(train_dists))

    test_buckets = get_buckets(test_dists, maximum, minimum, num_buckets)
    train_buckets = get_buckets(train_dists, maximum, minimum, num_buckets)
    #sim_buckets = get_buckets(sim_dists, maximum, minimum, num_buckets)

    plot.plot_line_2d([(i * (maximum - minimum) / (num_buckets) + minimum) for i in range(num_buckets)], 
                      [test_buckets, train_buckets], 
                      ["Testing Points by log Distances", 
                       "Training Points by log Distances"], lambda x:x)

def project(v, u):
    return np.dot(v, u) / (np.linalg.norm(u))

def plot_center_class_projection():
    classes = class_embeddings

    center = np.zeros(len(classes[0]))
    for c in classes:
        center = center + c
    center = center / len(classes)

    xs = [[], [], []]
    ys = [[], [], []]

    for i, embedding in enumerate(test_embeddings):
        class_embedding = np.array(classes[test_labels[i]])
        e = np.array(embedding) - class_embedding
        c = center - class_embedding
        p = project(e, c)

        x = p
        y = np.linalg.norm(e - p * c / np.linalg.norm(c))

        clossest_class = argmin([((np.array(embedding) - np.array(ce))**2).sum() for ce in classes])
        index = 0 if clossest_class == test_labels[i] else 1
        
        xs[index].append(x)
        ys[index].append(y)

    for i, class_embedding in enumerate(classes):
        for ii, embedding in enumerate(classes):
            if i != ii:
                e = np.array(embedding) - class_embedding
                c = center - class_embedding
                p = project(e, c)

                x = p
                y = np.linalg.norm(e - p * c / np.linalg.norm(c))
                xs[2].append(x)
                ys[2].append(y)   

    #plot.plotHeatMap(projections, anti_projections, 50, 50, "Amount")
    plot.plot_points_series_2d(xs, ys)

def pca_pure_classification():
    i = 29
    data = read_pickle_file(os.path.dirname(__file__) + f"/../embeddingData/classification_data_train_{i}.json")
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

def print_stats_pure():
    all_dists = []

    for epoch in range(30):
        print(epoch)
        train_embeddings, train_labels, test_embeddings, test_labels = get_pure_data(epoch)
        class_embeddings = get_class_centers(test_embeddings, test_labels, 10)

        center = np.zeros(len(class_embeddings[0]))
        for i in class_embeddings:
            center += i
        center = center / len(class_embeddings)
        
        dists = []

        for i in class_embeddings:
            dists.append(np.linalg.norm(center - i))

        all_dists.append(dists)

    plot.plotSurface([all_dists], "Distance To Class Center", [i for i in range(30)], "Epochs", [i for i in range(10)], "Classes", 1, [""])

def get_pure_data(epoch) :
    data = read_pickle_file(os.path.dirname(__file__) + f"/../embeddingData/classification_data_train_{epoch}.json")
    train_embeddings = data["train_embeddings"]
    train_labels = data["train_labels"]

    data = read_pickle_file(os.path.dirname(__file__) + f"/../embeddingData/classification_data_val_{epoch}.json")
    test_embeddings = data["val_embeddings"]
    test_labels = data["val_labels"]
    return train_embeddings, train_labels, test_embeddings, test_labels

def plot_pure():
    best_epoch = 29
    train_embeddings, train_labels, test_embeddings, test_labels = get_pure_data(best_epoch)

    pca = PCA(n_components = 3)
    train_projections = pca.fit_transform(train_embeddings)
    test_projections  = pca.transform(test_embeddings)

    COLOR = plot.get_colors(10)
    series = [{"marker": ".", "color": COLOR[i % 10], "label": f"{i%10}", "points":[]} for i in range(10)]

    class_embeddings = get_class_centers(test_embeddings, test_labels, 10)

    misclassified = 0
    for i, v in enumerate(test_embeddings):
        label = test_labels[i]

        clossest_class = argmin([((np.array(v) - np.array(c))**2).sum() for c in class_embeddings])
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
    for epoch in range(1, 30, 14):
        print(epoch)
        train_embeddings, train_labels, test_embeddings, test_labels = get_pure_data(epoch)
        class_embeddings = get_class_centers(test_embeddings, test_labels, 10)

        test_dists = list(filter(lambda x : x < 60, get_dists(test_embeddings, test_labels, class_embeddings, lambda x : x)))
        train_dists = list(filter(lambda x : x < 60, get_dists(train_embeddings, train_labels, class_embeddings, lambda x : x)))

        all_test_dists.append(test_dists)
        all_train_dists.append(train_dists)
        num_buckets = 100

        maximum = max(max(test_dists), max(train_dists), maximum)
        minimum = min(min(test_dists), min(train_dists), minimum)

    for i in range(len(all_test_dists)):
        test_buckets.append(get_buckets(all_test_dists[i], maximum, minimum, num_buckets))
        train_buckets.append(get_buckets(all_train_dists[i], maximum, minimum, num_buckets))

    plot.plot_line_2d([(i * (maximum - minimum) / (num_buckets) + minimum) for i in range(num_buckets)], 
                      [test_buckets[0], test_buckets[1], test_buckets[2]], 
                      ["Testing epoch 1", "Testing epoch 15", "Training epoch 29"], lambda x:x)

    #plot.plotSurface([test_buckets, train_buckets], "Dist Frequency", [i for i in range(30)], "Epoch", 
    #                 [(i * (maximum - minimum) / (num_buckets) + minimum) for i in range(num_buckets)], "Dist", 1, ["Test", "Train"])

def plot_acc_by_epoch(): 
    acc = []
    for epoch in range(30):
        misclassified = 0
        
        train_embeddings, train_labels, test_embeddings, test_labels = get_pure_data(epoch)
        class_embeddings = get_class_centers(test_embeddings, test_labels, 10)

        for i, v in enumerate(test_embeddings):
            label = test_labels[i]

            clossest_class = argmin([((np.array(v) - np.array(c))**2).sum() for c in class_embeddings])
            misclassified += 0 if clossest_class == label else 1
        
        acc.append(100.0 * (1.0 - misclassified/len(test_embeddings)))
    
    plot.plot_simple_line_2d(acc, lambda x:x)

def get_dists_by_label(embeddings, labels, class_embeddings):
    dists = [[] for _ in range(10)]
    for i, embedding in enumerate(embeddings):
        class_embedding = class_embeddings[labels[i]]
        dists[labels[i]].append(np.log(np.linalg.norm(np.array(class_embedding) - np.array(embedding))))
    return dists

def get_zipped(l, h):
    output = []
    for i, e in enumerate(l):
        output.append(e + h[i])
    return output

def deep_filter(l, lvls, func, start_value):
    if lvls == 0:
        return l
    else:
        value = start_value
        for v in l:
            value = func(value, deep_filter(v, lvls - 1, func, start_value))
        return value

def plot_distance_distribution():
    epoch = 0
    train_embeddings, train_labels, test_embeddings, test_labels = get_pure_data(epoch)
    class_embeddings = get_class_centers(test_embeddings, test_labels, 10)

    test_dists = get_dists_by_label(test_embeddings, test_embeddings, class_embeddings)
    train_dists = get_dists_by_label(train_embeddings, train_labels, class_embeddings)
    all_dists = get_zipped(test_dists, train_dists)

    num_buckets = 30

    maximum = deep_filter(all_dists, 2, max, float("-inf"))
    minimum = deep_filter(all_dists, 2, min, float("inf"))

if __name__ == '__main__':
    #data = read_json_file(os.path.dirname(__file__) + "/../../embeddingData/json_data.json")

    #train_embeddings = data["train_embeddings"]
    #test_embeddings  = data["test_embeddings"]
    #class_embeddings = data["class_embeddings"]
    #test_labels = data["test_labels"]
    #train_labels = data["train_labels"]

    print_stats_pure()
    #plot_acc_by_epoch()
    #plot_pure_distance_distribution()

    #print_distance_matrix(class_embeddings)
    #print_distances_from_center()
    #plot_series()
    #plot_pcas()
    #plot_distance_distribution()
    #plot_center_class_projection()
    print("Done")

