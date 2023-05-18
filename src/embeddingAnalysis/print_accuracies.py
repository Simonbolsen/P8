import json
import os
import matplotlib
import numpy
from analysis_util import get_exp_report_name, experiment_sorter

def get_acc_of_func(func_results):
    return max(func_results)

def print_accuracies(plot_data, formatter):

    acc_per_loss_func = plot_data["acc"][1]
    def get_display_acc(name):
        if name in acc_per_loss_func:
            return formatter(get_acc_of_func(acc_per_loss_func[name]))
        else:
            return formatter(None)

    name       = get_exp_report_name(plot_data)
    cos_acc    = get_display_acc("cosine")
    cos_cd_acc = get_display_acc("cosine_ce")
    euc_acc    = get_display_acc("eucledian")
    euc_ce_acc = get_display_acc("eucledian_ce")
    pure_acc   = get_display_acc("pure")

    print(name + " & " + cos_acc + " & " + cos_cd_acc + " & " + euc_acc + " & " + euc_ce_acc + " & " + pure_acc + " \\\\")


def get_all_data(data_path):
    data = dict()
    for entry in os.scandir(path=data_path):
        if entry.is_file():
            with open(entry, "rb") as json_file:
                data[entry.name] = json.loads(json.load(json_file))

    return data


def accuracy_formatter(acc):
    if acc == None:
        return "-"
    return str(round(acc * 100, 1)) + "\%"


def accuracy_formatter_coloured(colormap, colormap_text, acc_range):
    def rgb2hex(color):
        (r,g,b,a) = color
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        
        return "{:02x}{:02x}{:02x}".format(r,g,b)

    def norm_acc(acc):
        acc_min, acc_max = acc_range
        return (acc - acc_min) / (acc_max-acc_min)

    def formatter(acc):
        if acc == None:
            return "-"
        
        acc_text = str(round(acc * 100, 1)) + "\%"
        if acc == acc_range[1]: # if highest acc, make bold
            acc_text = "\\textbf{" + acc_text + "}"
        
        return "\\cellcolor[HTML]{" + rgb2hex(colormap(norm_acc(acc))) + "}" + acc_text
    
    return formatter

def get_acc_range_in_dataset(dataset):
    lowest = numpy.inf
    highest = -numpy.inf

    for row in dataset:
        for funcresults in row["acc"][1].values():
            lowest = min(lowest, get_acc_of_func(funcresults))
            highest = max(highest, get_acc_of_func(funcresults))

    return (lowest, highest)

if __name__ == '__main__':
    all_data = get_all_data("./src/plotting/plotData")

    sorter = experiment_sorter()
    data_per_dataset = sorter.group_by_dataset(all_data)

    colormap = lambda x: matplotlib.colormaps["magma"](x/3 + 2/3)
    colormap_text = matplotlib.colormaps["magma_r"]

    print("\\begin{table}[]")
    print("\\begin{tabular}{lccccc}")
    for dataset_name, dataset in data_per_dataset.items():
        print("\\textbf{" + dataset_name + "} & cos & $\mathrm{cos}_{cd}$ & euc & $\mathrm{euc}_{ce}$ & pure \\\\")
        dataset_accuracy_range = get_acc_range_in_dataset(dataset)
        for data in sorter.sort(dataset):
            print_accuracies(data, formatter=accuracy_formatter_coloured(colormap, colormap_text, dataset_accuracy_range))
        print("\\\\")

    print("\\end{tabular}")
    print("\\end{table}")