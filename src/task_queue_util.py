'''
    Simple script to select names
    for tasks
'''

import os
import jstyleson

# get the path of the current file
current_path = os.path.abspath(__file__)
# get the path to the folder the current file is in
folder_path = os.path.dirname(current_path)

path_to_tasks_json = os.path.join(folder_path, '..', '.vscode/tasks.json')
print('==> tasks: ', path_to_tasks_json)

print('==> loading json')
with open(path_to_tasks_json, 'r') as f:
    tasks = jstyleson.load(f)
    labels = {i: task['label'] for i, task in enumerate(tasks['tasks'])}
    
    for k, v in labels.items():
        print(f"{k}: {v}")
    print('\nSelect tasks (eg. 0 1 6 7): ', end='')

    selected = input().split(sep=' ')
    selected_list = [labels[int(i)] for i in selected]
    print(selected_list)
