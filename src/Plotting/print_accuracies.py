import json
import os
from ray import tune
if __name__ == '__main__':
    import results_analysis_util as analysis
else:
    import Plotting.results_analysis_util as analysis


def print_accuracies(plot_data):

    name = plot_data["name"]
    def format_acc(acc): 
        return str(round(acc * 100, 1)) + "\%"
    

    acc_per_loss_func = plot_data["acc"][1]
    def get_display_acc(name):
        if name in acc_per_loss_func:
            return format_acc(max(acc_per_loss_func[name]))
        else:
            return "n/a"


    cos_acc    = get_display_acc("cosine")
    cos_cd_acc = get_display_acc("cosine_ce")
    euc_acc    = get_display_acc("eucledian")
    euc_ce_acc = get_display_acc("eucledian_ce")
    pure_acc   = get_display_acc("pure")

    print(name + " & " + cos_acc + " & " + cos_cd_acc + " & " + euc_acc + " & " + euc_ce_acc + " & " + pure_acc + " \\\\")


if __name__ == '__main__':
    data_path = "./plots/plotData"

    print("\\begin{table}[]")
    print("\\begin{tabular}{lrrrr}")
    print("Experiment ID & cos_acc & cos_cd_acc & euc_acc & euc_ce_acc & pure\\\\")

    for entry in os.scandir(path=data_path):
        if entry.is_file():
            entryname = entry.name
            with open(entry, "rb") as json_file:
                data = json.loads(json.load(json_file))

                print_accuracies(data)
                # try:
                #     print_accuracies(data)
                # except:
                #     print("Failed to plot " + entryname)

    print("\\end{tabular}")
    print("\\end{table}")