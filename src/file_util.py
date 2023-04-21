import os
import json
import pickle

def save_to_json(folder, file_name, object):
    folder_path = os.path.join(os.path.realpath(__file__), '..', folder)

    if not os.path.exists(folder_path):
        print("==> folder to save embedding does not exist... creating folder...")
        print("   ==> folder path: ", folder_path)
        os.mkdir(folder_path)
    
    with open(os.path.join(folder_path, file_name), 'w+') as outfile:
        json.dump(json.dumps(object), outfile)

def save_to_pickle(folder, file_name, object):
    folder_path = os.path.join(os.path.realpath(__file__), '..', folder)

    if not os.path.exists(folder_path):
        print("==> folder to save embedding does not exist... creating folder...")
        print("   ==> folder path: ", folder_path)
        os.mkdir(folder_path)
    
    pickle.dump(object, open(os.path.join(folder_path, file_name), "wb"))


def read_json_file(path_to_json_file):
    with open(path_to_json_file, 'r') as json_file:
        data = json_file.read()
    return json.loads(json.loads(data))

def read_pickle_file(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
        return data
