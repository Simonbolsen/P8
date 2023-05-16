
import os
import jstyleson
from embeddingAnalysis.analysis_util import get_exp_report_name
from num2tex import num2tex
from decimal import Decimal

def generate_latex_table_line(task_name, lr, dims, prox_mult):
    return task_name + ' & ' + '${:e}$'.format(num2tex(float(lr))) + ' & ' + dims + ' & ' + prox_mult + '\\\\'

tasks_to_check = [
"BEST : cl_embed_push_res_large_fashion",
"BEST : cl_embed_cosine_res_large_fashion",        
"BEST : cl_embed_simple_res_large_fashion",        
"BEST : cl_embed_push_res_med_fashion",
"BEST : cl_embed_cosine_res_med_fashion",
"BEST : cl_embed_simple_res_med_fashion",
"BEST : cl_embed_push_res_small_fashion",
"BEST : cl_embed_cosine_res_small_fashion",        
"BEST : cl_embed_simple_res_small_fashion",        
"BEST : cl_pure_res_large_fashion_mnist",
"BEST : cl_pure_res_med_fashion_mnist",
"BEST : cl_pure_res_small_fashion_mnist",
"BEST : cl_embed_push_res_large_cifar_10",
"BEST : cl_embed_cosine_res_large_cifar_10",       
"BEST : cl_embed_simple_res_large_cifar_10",
"BEST : cl_embed_push_res_med_cifar_10",
"BEST : cl_embed_cosine_res_med_cifar_10",
"BEST : cl_embed_simple_res_med_cifar_10",
"BEST : cl_embed_push_res_small_cifar_10",
"BEST : cl_embed_cosine_res_small_cifar_10",
"BEST : cl_embed_simple_res_small_cifar_10",
"BEST : cl_pure_res_large_cifar_10",
"BEST : cl_pure_res_med_cifar_10",
"BEST : cl_pure_res_small_cifar_10",
"BEST : cl_embed_simple_res_large_cifar_100",
"BEST : cl_embed_push_res_large_cifar_100",
"BEST : cl_embed_cosine_res_large_cifar_100",
"BEST : cl_embed_push_res_med_cifar_100",
"BEST : cl_embed_cosine_res_med_cifar_100",
"BEST : cl_embed_simple_res_med_cifar_100",
"BEST : cl_embed_push_res_small_cifar_100",
"BEST : cl_embed_cosine_res_small_cifar_100",
"BEST : cl_embed_simple_res_small_cifar_100",
"BEST : cl_pure_res_large_cifar_100",
"BEST : cl_pure_res_med_cifar_100",
"BEST : cl_pure_res_small_cifar_100",
"BEST : cl_embed_push_res_large_mnist",
"BEST : cl_embed_cosine_res_large_mnist",
"BEST : cl_embed_simple_res_large_mnist",
"BEST : cl_embed_push_res_med_mnist",
"BEST : cl_embed_cosine_res_med_mnist",
"BEST : cl_embed_simple_res_med_mnist",
"BEST : cl_embed_push_res_small_mnist",
"BEST : cl_embed_cosine_res_small_mnist",
"BEST : cl_embed_simple_res_small_mnist",
"BEST : cl_pure_res_large_mnist",
"BEST : cl_pure_res_med_mnist",
"BEST : cl_pure_res_small_mnist",
]

# get the path of the current file
current_path = os.path.abspath(__file__)
# get the path to the folder the current file is in
folder_path = os.path.dirname(current_path)

path_to_tasks_json = os.path.join(folder_path, '..', '.vscode/tasks.json')
print('==> tasks: ', path_to_tasks_json)

print('==> loading json')

table_lines_pr_dataset = {}
with open(path_to_tasks_json, 'r') as f:
    tasks = jstyleson.load(f)
    tasks_filtered = []
    
    for task in tasks['tasks']:
        for task_to_check in tasks_to_check:
            if task['label'] == task_to_check:
                tasks_filtered.append(task)
    
    for task in tasks_filtered:
        args = task['args']
        loss_func = args[args.index('--loss-func') + 1]
        
        model = args[args.index('--model') + 1]
        lr = args[args.index('--lr') + 1]
        dims = '-'
        prox_mult = '-'
        dataset = args[args.index('--dataset') + 1]
        
        if loss_func == 'simple-dist' or loss_func == 'cosine-loss':
            dims = args[args.index('--dims') + 1]
        elif loss_func == 'class-push':
            prox_mult = args[args.index('--prox-mult') + 1]
            dims = args[args.index('--dims') + 1]
        # todo: add pnp
        
        meta_data = {'config':{'model_name': model, 'loss_func': loss_func}}
        
        task_name = get_exp_report_name(meta_data)
        
        row = generate_latex_table_line(task_name, lr, dims, prox_mult)
        
        if dataset not in table_lines_pr_dataset.keys():
           table_lines_pr_dataset[dataset] = []
        
        table_lines_pr_dataset[dataset].append(row) 
    
    for k in table_lines_pr_dataset:
        table_lines_pr_dataset[k].sort()
        
print("\\begin{table}[]")
print("\\begin{tabular}{lrrr}")
print("Experiment ID & Learning Rate & d & $Prox_{mult}$ \\\\")
for dataset_name in table_lines_pr_dataset:
    print("\\\\"+ dataset_name + " & & & \\\\")
    for row in table_lines_pr_dataset[dataset_name]:
        print(row)

print("\\end{tabular}")
print("\\end{table}")
        
        



        
        