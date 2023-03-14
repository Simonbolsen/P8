import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import os.path
from sklearn.decomposition import PCA
import Plotting.plotting_util as plot

def read_json_file(path_to_json_file):
    with open(path_to_json_file, 'r') as json_file:
        data = json_file.read()
    return json.loads(json.loads(data))

data = read_json_file(os.path.dirname(__file__) + "/../../embeddingData/json_data.json")


 
pca = PCA(n_components = 3)

projections = pca.fit_transform(data["train_embeddings"])
projections = pca.transform(data["test_embeddings"])
class_projections = pca.transform(data["class_embeddings"])

xs = [[] for _ in range(11)]
ys = [[] for _ in range(11)]
zs = [[] for _ in range(11)]

for i, v in enumerate(projections):
    xs[data["test_labels"][i]].append(v[0])
    ys[data["test_labels"][i]].append(v[1])
    zs[data["test_labels"][i]].append(v[2])

for i, v in enumerate(class_projections):
    xs[10].append(v[0])
    ys[10].append(v[1])
    zs[10].append(v[2])

labels = [i for i in range(10)]
labels.append("Classes")

plot.plotPoints(xs, ys, zs, num_of_series=11, series_labels=labels, function= lambda x: x)

























