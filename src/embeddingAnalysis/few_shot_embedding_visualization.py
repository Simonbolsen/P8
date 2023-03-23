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

data = ev.read_json_file(os.path.dirname(__file__) + "/../../embeddingData/few_shot_test_data.json")
def min_pair(iterable):
    m = float("inf")
    k = "ERROR"
    for i, v in iterable.items():
        if v < m:
            m = v
            k = i
    return k, m


def plot_pca():
    train_embeddings = data["train_embeddings"]
    vs_embeddings  = data["val_support_embeddings"]
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

    train_embeddings = data["train_embeddings"]
    class_embeddings = data["class_embeddings"]
    train_labels = data["train_labels"]

    scores = []
    accs = []
    for i in range(1, 60):
        pca = PCA(n_components = i)
        train_projections = pca.fit_transform(train_embeddings)
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
            print(f"WTF! {acc}")

    plot.plot_simple_line_2d(accs)
    plot.plot_simple_line_2d(scores, function=lambda x:x)

if __name__ == '__main__':
    #plot_pca()
    #ev.print_distance_matrix(data["class_embeddings"])
    plot_pca_comparison()