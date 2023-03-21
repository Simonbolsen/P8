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

def read_json_file(path_to_json_file):
    with open(path_to_json_file, 'r') as json_file:
        data = json_file.read()
    return json.loads(json.loads(data))

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]

data = read_json_file(os.path.dirname(__file__) + "/../../embeddingData/json_data.json")


 


train_embeddings = data["train_embeddings"]
test_embeddings  = data["test_embeddings"]
class_embeddings = data["class_embeddings"]

scores = []
accs = []

for i in range(1, 60):
    pca = PCA(n_components = i)
    train_projections = pca.fit_transform(train_embeddings)
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


test_projections  = pca.transform(test_embeddings)
class_projections = pca.transform(class_embeddings)

plot.plot_2d(accs)
plot.plot_2d(scores)

COLOR = plot.get_colors(10)
series = [{"marker": "." if i < 10 else "*", "color": COLOR[i % 10], "label": f"{i%10} {'C' if i < 10 else 'Misc'}lassified", "points":[]} for i in range(20)]

def print_distance_matrix():
    print("")
    for i in range(len(class_embeddings)):
        print(f"  {i}", end="  ")
    print("")
    for index, i in enumerate(class_embeddings):
        print(f"{index}", end=" ")
        for ii in class_embeddings:
            print(f"{np.linalg.norm(np.array(i) - np.array(ii)):.2f}",end=" ")
        print("")

def print_distances_from_center():
    center = np.zeros(len(class_embeddings[0]))
    for i in class_embeddings:
        center += i
    center = center / 10
    print()
    for i in class_embeddings:
        print(np.linalg.norm(center - i))

def plot_series():
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


#print_distance_matrix()
#print_distances_from_center()
#plot_series()

print("Done")








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












