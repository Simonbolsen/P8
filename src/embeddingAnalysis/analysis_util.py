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
