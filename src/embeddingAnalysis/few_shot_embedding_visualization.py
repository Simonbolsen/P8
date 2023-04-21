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
import embedding_visualization as ev
import analysis_util as au

data = ev.read_json_file(os.path.dirname(__file__) + "/../../embeddingData/json_data.json") #few_shot_test_data
def min_pair(iterable):
    m = float("inf")
    k = "ERROR"
    for i, v in iterable.items():
        if v < m:
            m = v
            k = i
    return k, m

def avg_embeddings(embeddings):
    avgs = []
    for terms in embeddings:
        sum = np.zeros(len(terms[0]))
        for e in terms:
            sum = sum + np.array(e)
        avgs.append(sum / len(terms))
    return avgs

def plot_pca():
    train_embeddings = data["train_embeddings"]
    vs_embeddings = data["val_support_embeddings"]
    vq_embeddings  = data["val_query_embeddings"]
    class_embeddings = data["class_embeddings"]

    train_labels = data["train_labels"]
    vs_labels = data["val_support_labels"]
    vq_labels = data["val_query_labels"]

    pca = PCA(n_components=3)
    
    vq_projections = pca.fit_transform(vq_embeddings)
    train_projections = pca.transform(train_embeddings)
    vs_projections = pca.transform(vs_embeddings)

    COLOR = plot.get_colors(10)
    series = [{"marker": ".", "color": COLOR[i], "label": f"{i}", "points":[]} for i in range(10)]
    #for i, embedding in enumerate(train_projections):
    #    series[train_labels[i]]["points"].append(embedding)

    for i, embedding in enumerate(vq_projections):
        series[vq_labels[i]]["points"].append(embedding)

    plot.plotCustomPoints(series, legend=True)

    print("Done")

def plot_pca_comparison():
    class_indices = [1, 3, 5, 7, 8, 9]
    new_class_indices = [0, 6]

    train_embeddings = data["train_embeddings"]
    class_embeddings = data["class_embeddings"]
    new_class_embeddings = avg_embeddings(data["new_class_embeddings"])
    vq_embeddings  = data["val_query_embeddings"]

    train_labels = data["train_labels"]
    vq_labels = data["val_query_labels"]

    scores = []
    accs = []
    few_shot_accs = []
    for i in range(1, 60):
        pca = PCA(n_components = i)
        train_projections = pca.fit_transform(train_embeddings)
        vq_projections = pca.transform(vq_embeddings)

        new_class_projections = pca.transform(new_class_embeddings)
        class_projections = pca.transform(class_embeddings)
        
        scores.append(pca.score(train_embeddings))
        if i % 5 == 0:
            print(i)
        misclassified = 0

        for i, v in enumerate(train_projections):
            label = train_labels[i]
            clossest_class, _ = min_pair({class_indices[i]:((np.array(v) - np.array(c))**2).sum() for i, c in enumerate(class_projections)})
            index = 0 if clossest_class == label else 1
            misclassified += index
        acc = 1 - misclassified / len(train_projections)
        accs.append(1-math.sqrt(1 - acc**2))

        if acc > 1 or acc < 0:
            print(f"WTF! train {acc}")

        misclassified = 0
        for i, v in enumerate(vq_projections):
            label = vq_labels[i]
            clossest_class, _ = min_pair({new_class_indices[i]:((np.array(v) - np.array(c))**2).sum() for i, c in enumerate(new_class_projections)})
            index = 0 if clossest_class == label else 1
            misclassified += index
        acc = 1 - misclassified / len(vq_projections)
        few_shot_accs.append(1-math.sqrt(1 - acc**2))

        if acc > 1 or acc < 0:
            print(f"WTF! {acc}")

    plot.plot_simple_line_2d(accs)
    plot.plot_simple_line_2d(few_shot_accs)
    plot.plot_simple_line_2d(scores, function=lambda x:x)

def get_full_class_list():
    class_embeddings = data["class_embeddings"]
    new_class_embeddings = avg_embeddings(data["new_class_embeddings"])

    return [new_class_embeddings[0], class_embeddings[0], [], class_embeddings[1], [], class_embeddings[2], new_class_embeddings[1], 
            class_embeddings[3], class_embeddings[4], class_embeddings[5]]

def get_dists_by_label(embeddings, labels, class_embeddings):
    dists = [[] for _ in range(10)]
    for i, embedding in enumerate(embeddings):
        class_embedding = class_embeddings[labels[i]]
        dists[labels[i]].append(np.log(np.linalg.norm(np.array(class_embedding) - np.array(embedding))))
    return dists

def plot_distance_distribution():
    train_embeddings = data["train_embeddings"]
    vq_embeddings  = data["test_embeddings"] #data["val_query_embeddings"]

    train_labels = data["train_labels"]
    vq_labels = data["test_labels"]#data["val_query_labels"]

    class_embeddings = data["class_embeddings"]#get_full_class_list()

    test_dists = get_dists_by_label(vq_embeddings, vq_labels, class_embeddings)
    train_dists = get_dists_by_label(train_embeddings, train_labels, class_embeddings)
    all_dists = au.get_zipped(test_dists, train_dists)

    num_buckets = 30

    maximum = au.deep_filter(all_dists, 2, max, float("-inf"))
    minimum = au.deep_filter(all_dists, 2, min, float("inf"))

    dim = 3 #data["config"]["d"]
    bucket_dists = [((i + 1) * (maximum - minimum) / (num_buckets) + minimum) for i in range(num_buckets)]
    buckets = []

    sphere_bucket = []
    inner_shpere = ev.hyper_sphere_volume(np.exp(minimum), dim)
    for dist in bucket_dists:
        outer_sphere = ev.hyper_sphere_volume(np.exp(dist), dim)
        diff = (outer_sphere - inner_shpere)
        diff = diff if diff < 0.14 else 0.14
        sphere_bucket.append(diff)
        inner_shpere = outer_sphere

    for dists in all_dists:
        bucket = ev.get_buckets(dists, maximum, minimum, num_buckets)
        buckets.append(bucket)
    
    buckets.append(sphere_bucket)

    plot.plot_line_2d(bucket_dists, buckets, 
                      [f"Class {i}" + (f" is val" if i == 0 or i == 6 else "") for i in range(10)] + [f"{dim} Dim Sphere"], 
                      lambda x:x)

    COLOR = plot.get_colors(10)
    series = [{"marker": ".", "color": COLOR[i], "label": f"{i}", "points":b} for i, b in enumerate(buckets)]


if __name__ == '__main__':
    #plot_pca()
    #ev.print_distance_matrix(data["class_embeddings"])
    #plot_pca_comparison()
    plot_distance_distribution()