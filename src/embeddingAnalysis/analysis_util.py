from math import ceil, log2
import numpy as np

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def get_dists(embeddings, labels, class_embeddings, func = lambda x: np.log(x)):
    dists = []
    for i, embedding in enumerate(embeddings):
        class_embedding = class_embeddings[labels[i]]
        dists.append(func(np.linalg.norm(np.array(class_embedding) - np.array(embedding))))
    return dists

def get_dists_by_label(embeddings, labels, class_embeddings):
    dists = [[] for _ in range(10)]
    for i, embedding in enumerate(embeddings):
        class_embedding = class_embeddings[labels[i]]
        dists[labels[i]].append(np.linalg.norm(np.array(class_embedding) - np.array(embedding)))
    return dists

def get_buckets(dists, maximum, minimum, num_buckets):
    buckets = [0 for _ in range(num_buckets)]
    for d in dists:
        index = int((d - minimum) * (num_buckets - 1) / (maximum - minimum))
        buckets[index] += 1/len(dists)
    return buckets

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

def print_distance_matrix(embeddings, other):

    print("")
    for i in range(len(embeddings)):
        print(f"   {i}", end="  ")
    print("")
    for index, i in enumerate(embeddings):
        print(f"{index}", end=" ")
        for ii in other:
            print(f"{np.linalg.norm(np.array(i) - np.array(ii)) < 5}",end=" ")
        print("")

def print_distances_from_center(embeddings):
    

    center = np.zeros(len(embeddings[0]))
    for i in embeddings:
        center += i
    center = center / len(embeddings)
    print()
    for i in embeddings:
        print(np.linalg.norm(center - i))


def get_exp_report_name(meta_data):
    if "meta_data" in meta_data: # If the entire json is sent
        meta_data = meta_data["meta_data"] 

    config = meta_data["config"]

    model_name = dict()
    model_name["resnet18"]  = "RN18"
    model_name["resnet50"]  = "RN50"
    model_name["resnet101"] = "RN101"

    loss_name = dict()
    loss_name["cross_entropy"] = "P"
    loss_name["simple-dist"]   = "E"
    loss_name["cosine-loss"]   = "C"
    loss_name["class-push"]    = "CP"

    return model_name[config["model_name"]] + "-" + loss_name[config["loss_func"]]



def group_by(list, predicate):
    groups = dict()
    for x in list:
        pred = predicate(x)
        if not pred in groups:
            groups[pred] = []

        groups[pred].append(x)
    return groups

def bit_length(num:int) -> int:
    return int(ceil(log2(num)))

class experiment_sorter():
    def __init__(self) -> None:
        self.sortings_dataset = [ "cifar100", "cifar10", "fashion", "mnist" ]
        self.loss_func        = [ "cosine-loss", "class-push", "simple-dist", "cross_entropy"]
        self.sortings_model   = [ "resnet101", "resnet50", "resnet18" ]

    def sort(self, experiments:list) -> list:
        experiments.sort(key=self.key)
        return experiments
    
    def key(self, meta_data):
        if "meta_data" in meta_data: # If the entire json is sent
            meta_data = meta_data["meta_data"] 

        config = meta_data["config"]

        order_id = 0
        
        # Dataset
        order_id += self.sortings_dataset.index(config["dataset"])
        order_id <<= bit_length(len(self.sortings_dataset))

        # Loss func
        order_id += self.loss_func.index(config["loss_func"])
        order_id <<= bit_length(len(self.loss_func))
        
        # Model
        order_id += self.sortings_model.index(config["model_name"])
        order_id <<= bit_length(len(self.sortings_model))

        return order_id

    def group_by_dataset(self, data:dict) -> dict[dict]:
        return group_by(data.values(), lambda x: x["meta_data"]["config"]["dataset"])
