from collections import OrderedDict
from math import ceil, log2
from typing import Generic, Optional, Tuple, TypeVar, TypedDict, Callable
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



T1 = TypeVar("T1")
T2 = TypeVar("T2")
def group_by(items:list[T1], predicate:Callable[[T1], T2]) -> dict[T2, list[T1]]:
    groups:dict[T2, list[T1]] = dict()

    for x in items:
        pred = predicate(x)
        if not pred in groups:
            groups[pred] = []

        groups[pred].append(x)
        
    return groups

Key_Type = TypeVar("Key_Type")
Content_Type = TypeVar("Content_Type")
def order_dict(groups:dict[Key_Type, Content_Type], orderer:Optional[Callable[[Tuple[Key_Type, Content_Type]], int]] = None) -> OrderedDict[Key_Type, Content_Type]:
    ordered_groups:OrderedDict[Key_Type, Content_Type] = OrderedDict()

    for key, group in sorted(list(groups.items()), key=orderer):
        ordered_groups[key] = group
        
    return ordered_groups

def bit_length(num:int) -> int:
    return int(ceil(log2(num)))

class dataset_model_loss(TypedDict):
    dataset: str
    loss_func: str
    model_name: str

class experiment_sorter():
    def __init__(self) -> None:
        self.sortings_dataset = [ "cifar100", "cifar10", "fashion", "mnist", "kmnist" ]
        self.loss_func        = [ "cosine-loss", "class-push", "simple-dist", "cross_entropy"]
        self.sortings_model   = [ "resnet101", "resnet50", "resnet18" ]

    def sort(self, experiments:list) -> list:
        experiments.sort(key=self.key)
        return experiments
    
    def get_order_id(self, dataset:Optional[str]=None, loss_func:Optional[str]=None, model_name:Optional[str]=None):
        order_id = 0
        
        # Dataset
        if dataset: order_id += self.sortings_dataset.index(dataset)
        order_id <<= bit_length(len(self.sortings_dataset))

        # Loss func
        if loss_func: order_id += self.loss_func.index(loss_func)
        order_id <<= bit_length(len(self.loss_func))
        
        # Model
        if model_name: order_id += self.sortings_model.index(model_name)
        order_id <<= bit_length(len(self.sortings_model))

        return order_id
    
    def key(self, meta_data):
        if "meta_data" in meta_data: # If the entire json is sent
            meta_data = meta_data["meta_data"] 

        config = meta_data["config"]

        dataset = config["dataset"] if "dataset" in config else None
        loss_func = config["loss_func"] if "loss_func" in config else None
        model_name = config["model_name"] if "model_name" in config else None

        return self.get_order_id(dataset, loss_func, model_name)

    # def group_by_dataset(self, data:dict) -> dict[dict]:
    #     return group_by(data.values(), lambda x: x["meta_data"]["config"]["dataset"])
    
    T = TypeVar("T")
    def group_by_dataset_loss_model(self, items:list[T], get_dataset_model_loss:Callable[[T], dataset_model_loss]) -> OrderedDict[str, OrderedDict[str, OrderedDict[str, list[T]]]]:

        datasets = group_by(items, lambda x: get_dataset_model_loss(x)["dataset"])
        ordered_datasets = order_dict(datasets, lambda x: self.get_order_id(dataset=x[0]))

        ordered_datasets_with_subsets = OrderedDict()

        for k, v in ordered_datasets.items():
            ordered_datasets_with_subsets[k] = self.group_by_loss_model(v, get_dataset_model_loss)
        
        return ordered_datasets_with_subsets
    
    def group_by_loss_model(self, items:list[T], get_dataset_model_loss:Callable[[T], dataset_model_loss]) -> OrderedDict[str, OrderedDict[str, list[T]]]:
        grouped_items = group_by(items, lambda x: get_dataset_model_loss(x)["loss_func"])
        ordered_groups = order_dict(grouped_items, lambda x: self.get_order_id(loss_func=x[0]))

        ordered_models_with_subsets = OrderedDict()

        for k, v in ordered_groups.items():
            ordered_models_with_subsets[k] = self.group_by_model(v, get_dataset_model_loss)
        
        return ordered_models_with_subsets
    
    def group_by_model(self, items:list[T], get_dataset_model_loss:Callable[[T], dataset_model_loss]) -> OrderedDict[str, list[T]]:

        grouped_items = group_by(items, lambda x: get_dataset_model_loss(x)["model_name"])
        ordered_groups = order_dict(grouped_items, lambda x: self.get_order_id(model_name=x[0]))
        
        return ordered_groups