import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import os.path
from sklearn.decomposition import PCA
import Plotting.plotting_util as plot
import numpy as np
import matplotlib.cm as cm

def read_json_file(path_to_json_file):
    with open(path_to_json_file, 'r') as json_file:
        data = json_file.read()
    return json.loads(json.loads(data))

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]

data = read_json_file(os.path.dirname(__file__) + "/../../embeddingData/json_data.json")


 
pca = PCA(n_components = 4)

train_embeddings = data["train_embeddings"]
test_embeddings  = data["test_embeddings"]
class_embeddings = data["class_embeddings"]

train_projections = pca.fit_transform(train_embeddings)
test_projections  = pca.transform(test_embeddings)
class_projections = pca.transform(class_embeddings)


COLOR = plot.get_colors(10)
series = [{"marker": "." if i < 10 else "*", "color": COLOR[i % 10], "label": f"{i%10} {'C' if i < 10 else 'Misc'}lassified", "points":[]} for i in range(20)]
misclassified = 0

def print_distance_matrix():
    for i in class_embeddings:
        print("")
        for ii in class_embeddings:
            print(f"{np.linalg.norm(np.array(i) - np.array(ii)):.2f}",end=" ")

def print_distances_from_center():
    center = np.zeros(len(class_embeddings[0]))
    for i in class_embeddings:
        center += i
    center = center / 10
    for i in class_embeddings:
        print(np.linalg.norm(center - i))

def setup_series():
    for i, v in enumerate(test_embeddings):
        label = data["test_labels"][i]

        clossest_class = argmin([((np.array(v) - np.array(c))**2).sum() for c in class_embeddings])
        index = 0 if clossest_class == label else 1
        misclassified += index
        
        series[label + 10 * index]["points"].append(test_projections[i])

    print(f"{misclassified}/{len(test_embeddings)}, acc: {100.0 * (1.0 - misclassified/len(test_embeddings)):.2f}%")
    plot.plotCustomPoints(series, legend=False, axes=[0,1,2])
    plot.plotCustomPoints(series, legend=False, axes=[0,1,3])
    plot.plotCustomPoints(series, legend=False, axes=[0,2,3])
    plot.plotCustomPoints(series, legend=False, axes=[1,2,3])









# PLOT COLORED BY CLASSES WITH CLASS EMBEDDINGS
#
#xs = [[] for _ in range(11)]
#ys = [[] for _ in range(11)]
#zs = [[] for _ in range(11)]
#
#for i, v in enumerate(projections):
#    xs[data["test_labels"][i]].append(v[0])
#    ys[data["test_labels"][i]].append(v[1])
#    zs[data["test_labels"][i]].append(v[2])
#
#for i, v in enumerate(class_projections):
#    xs[10].append(v[0])
#    ys[10].append(v[1])
#    zs[10].append(v[2])

#labels = [i for i in range(10)]
#labels.append("Classes")
#plot.plotPoints(xs, ys, zs, num_of_series=11, series_labels=labels, function= lambda x: x)












