from ray import tune
import Plotting.plotting_util as plot
from Plotting.plotting_util import axis
import Plotting.results_analysis_util as analysis
import math
import os

# parameters = {
#     "lr": "Learning Rate log_10(lr)",
#     "d" : "Dimensions d",
#     "i" : "Training Iterations i",
#     "e" : "Epochs e",
#     "n" : "Linear Num n",
#     "s" : "Linear Size s",
#     "b" : "Batch Size log_2(b)",
#     "c" : "Channels c"
# }

# param1 = "lr"
# paramColor = "e"

bin1 = []
bin2 = []

bin1_size = 0.5
bin2_size = 10
binColer_size = 3

def insert_append(l, i, e):
    if i < len(l):
        l[i].append(e)
    else:
        while i > len(l):
            l.append(None)
        l.append(e)



def make_plots(experiment_id:str, save_location:str = None) -> None:
    make_config_plots(experiment_id, save_location)

def make_config_plots(experiment_id:str, save_location:str = None) -> None:
    results = tune.ExperimentAnalysis("~/ray_results/" + experiment_id, 
                                    default_metric="accuracy", 
                                    default_mode="max")
    best_results = analysis.best_iterations_per_trial(results)

    accuracies = []
    colors = axis("Iterations", [])

    visual_func = lambda x: (1 - math.sqrt(1 - x**2))
    inv_visual_func = lambda x: math.sqrt(1 - (1 -  x)**2)

    data = {}
    index = 0
    for k, result in best_results.items():
        insert_append(accuracies, index, result["data"]["accuracy"])
        insert_append(colors.data, index, int(result["data"]["training_iteration"] / binColer_size))

        for key, value in result["config"].items():
            if not key in data.keys():
                data[key] = []

            insert_append(data[key], index, value)

        index += 1

    for key, value in data.items():

        if value[0] == str(value[0]):
            continue

        figure = make_2d_plot(
            axis1=axis(key, value),
            # axis2=axis("Accuracy", [visual_func(x) for x in accuracies])
            axis2=axis("Accuracy", accuracies),
            colors=colors
        )
        
        if save_location != None:
            savePath = save_location + key
            os.makedirs(os.path.dirname(savePath), exist_ok=True)
            figure.savefig(save_location + key)
        else:
            figure.show()
        print("Plotted: " + key)

    
def make_2d_plot(axis1:axis, axis2:axis, colors:axis=None) -> plot:

    figure = plot.plotPoints2d(
        xs=axis1,
        ys=axis2,
        colors=colors,
        legend=True,
        num_of_series=len(axis2.data),
        series_labels=[f"{i * binColer_size}" for i in range(len(axis2.data))]
    )

    return figure


if __name__ == '__main__':
    
    title = "cl_pure_res_large_cub_200"
    make_plots(title, os.path.expanduser("~/ray_results/plots/") + title + "/")

