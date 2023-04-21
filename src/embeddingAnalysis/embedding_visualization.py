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
import analysis_util as au

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
data = None


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
            clossest_class = au.argmin([((np.array(v) - np.array(c))**2).sum() for c in class_projections])
            index = 0 if clossest_class == label else 1
            misclassified += index
        acc = 1 - misclassified / len(test_embeddings)
        accs.append(1-math.sqrt(1 - acc**2))

    plot.plot_2d(accs)
    plot.plot_2d(scores)

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

        clossest_class = au.argmin([((np.array(v) - np.array(c))**2).sum() for c in class_embeddings])
        index = 0 if clossest_class == label else 1
        misclassified += index
        
        series[label + 10 * index]["points"].append(test_projections[i])

    print(f"{misclassified}/{len(test_embeddings)}, acc: {100.0 * (1.0 - misclassified/len(test_embeddings)):.2f}%")
    plot.plotCustomPoints(series, legend=True, axes=[0,1,2])
    #plot.plotCustomPoints(series, legend=False, axes=[0,1,3])
    #plot.plotCustomPoints(series, legend=False, axes=[1,2,3])
    #plot.plotCustomPoints(series, legend=False, axes=[0,2,3])

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
    test_dists = au.get_dists(test_embeddings, test_labels, class_embeddings)
    train_dists = au.get_dists(train_embeddings, train_labels, class_embeddings)
    #sim_dists = get_sim_dists(4, 0.25, 10000)
    num_buckets = 100

    maximum = max(max(test_dists), max(train_dists))
    minimum = min(min(test_dists), min(train_dists))

    test_buckets = au.get_buckets(test_dists, maximum, minimum, num_buckets)
    train_buckets = au.get_buckets(train_dists, maximum, minimum, num_buckets)
    #sim_buckets = get_buckets(sim_dists, maximum, minimum, num_buckets)

    plot.plot_line_2d([(i * (maximum - minimum) / (num_buckets) + minimum) for i in range(num_buckets)], 
                      [test_buckets, train_buckets], 
                      ["Testing Points by log Distances", 
                       "Training Points by log Distances"], lambda x:x)

def project(v, u):
    return np.dot(v, u) / (np.linalg.norm(u))

    all_epochs = []

    for epoch in range(30):       
        print(f"{epoch + 1} / 30") 
        train_embeddings, train_labels, test_embeddings, test_labels = get_pure_data(epoch)
        test_class_embeddings = get_class_centers(test_embeddings, test_labels, 10)
        train_class_embeddings = get_class_centers(train_embeddings, train_labels, 10)

        all_epochs.append({"test": test_class_embeddings, "train": train_class_embeddings})

    save_to_pickle("embeddingData", "class_centers", all_epochs)

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

        clossest_class = au.argmin([((np.array(embedding) - np.array(ce))**2).sum() for ce in classes])
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

if __name__ == '__main__':
    #data = read_json_file(os.path.dirname(__file__) + "/../../embeddingData/json_data.json")

    #train_embeddings = data["train_embeddings"]
    #test_embeddings  = data["test_embeddings"]
    #class_embeddings = data["class_embeddings"]
    #test_labels = data["test_labels"]
    #train_labels = data["train_labels"]


    print("Done")

