import json
import os
from analysis_util import get_exp_report_name


def print_accuracies(plot_data):
    def format_acc(acc): 
        return str(round(acc * 100, 1)) + "\%"
    

    acc_per_loss_func = plot_data["acc"][1]
    def get_display_acc(name):
        if name in acc_per_loss_func:
            return format_acc(max(acc_per_loss_func[name]))
        else:
            return "-"


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

def group_by(list, predicate):
    groups = dict()
    for x in list:
        pred = predicate(x)
        if not pred in groups:
            groups[pred] = []

        groups[pred].append(x)
    return groups

if __name__ == '__main__':
    all_data = get_all_data("./plots/plotData")


    data_per_dataset = group_by(all_data.values(), lambda x: x["meta_data"]["config"]["dataset"])


    print("\\begin{table}[]")
    print("\\begin{tabular}{lrrrrr}")
    print("Experiment ID & cos\_acc & cos\_cd\_acc & euc\_acc & euc\_ce\_acc & pure\\\\")
    for dataset_name, dataset in data_per_dataset.items():
        print(dataset_name + " & & & & & \\\\")
        for data in dataset:
            print_accuracies(data)

    print("\\end{tabular}")
    print("\\end{table}")