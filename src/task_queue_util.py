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
    
    if 'y' == input('Filter? y/[n]: '):
        # split on space and remove / 
        filter_term = input('/').lower()
        
        labels_copy = labels.copy()
        for k in labels_copy:
            if filter_term not in labels_copy[k].lower():
                labels.pop(k)
        
        for k, v in labels.items():
            print(f"{k}: {v}")

    print('\nSelect tasks (eg. 0 1 6 7. or 5-10. Default: All): ', end='')

    selected = []

    cin = input()

    if '-' in cin:
        ranges = cin.split('-')
        selected = range(int(ranges[0]), int(ranges[1]) + 1)
    else:
        selected = cin.split(sep=' ')
    
    # DEFAULT
    if not selected[0]:
        selected_list = "".join(['\"' + label + '\",\n' for label in labels.values()])
    else:
        selected_list = "".join(['\"' + labels[int(i)] + '\",\n' for i in selected])
        
    print(selected_list)

