from typing import Optional, Sequence, Tuple
import numpy
from ray import tune
import math
import os
import numbers
import matplotlib.scale

if __name__ == '__main__':
    import plotting_util as plot
    from plotting_util import axis
    import results_analysis_util as analysis
else:
    import Plotting.plotting_util as plot
    from Plotting.plotting_util import axis
    import Plotting.results_analysis_util as analysis

axisOverwrites = {
    "lr": {"label": "Learning Rate lr", "scale": matplotlib.scale.LogScale(None)},
    "d" : {"label": "Dimensions d"},
    "i" : {"label": "Training Iterations i"},
    "e" : {"label": "Epochs e"},
    "n" : {"label": "Linear Num n"},
    "s" : {"label": "Linear Size s"},
    "b" : {"label": "Batch Size b"},
    "c" : {"label": "Channels c"}
}

def apply_overwrites(axis:axis, overwrites):
    for name, value in overwrites:
        setattr(axis, name, value)

binColer_size = 3

def insert_append(l:list[list], i:int, e):
    if i < len(l):
        l[i].append(e)
    else:
        while i > len(l):
            l.append([])
        l.append([e])

def simon_visual_func(x):
    return 1 - (1 - x**2)**(1/2)

def simon_inv_visual_func(x):
    return (1 - (1 - x)**2)**(1/2)

def get_simon_scale() -> matplotlib.scale.FuncScale:
    return matplotlib.scale.FuncScale(None, functions=(simon_visual_func, simon_inv_visual_func))

def make_experiment_plots(experiment_id:str, save_location:Optional[str] = None) -> None:
    accuracies, configData = get_formatted_config_data_and_accuracies(experiment_id)

    accuracies.scale = get_simon_scale()

    # Apply any overwrites
    for (key, overwrites) in axisOverwrites.items():
        if key in configData.keys():
            for (property, value) in overwrites.items():
                setattr(configData[key], property, value)

    make_plots(configData.values(), [accuracies], name=experiment_id, save_location=save_location)


def save_plot(plt:plot, save_location:str, plot_id:str):
    plot_save_path = os.path.join(save_location, plot_id)
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)

def get_formatted_config_data_and_accuracies(experiment_id:str) -> Tuple[axis, dict[str, axis]]:
    results = tune.ExperimentAnalysis("~/ray_results/" + experiment_id, 
                                    default_metric="accuracy", 
                                    default_mode="max")
    best_results = analysis.best_iterations_per_trial(results)

    accuracies:axis = axis("Accuracy", [])

    data:dict[str, axis] = dict()
    for k, result in best_results.items():
        color_index = int(result["data"]["training_iteration"] / binColer_size)
        insert_append(accuracies.data, color_index, result["data"]["accuracy"])

        for key, value in result["config"].items():
            if not isinstance(value, numbers.Number):
                continue
    
            if not key in data.keys():
                data[key] = axis(key, [])

            insert_append(data[key].data, color_index, value)
    
    return accuracies, data

def make_plots(xAxis:list[axis], yAxis:Sequence[axis], name:Optional[str], save_location:Optional[str] = None) -> None:
    for x in xAxis:
        for y in yAxis:
            figure = make_2d_plot(x, y)

            plot_name = x.label + "_" + y.label

            if name != None:
                figure.title(name)

            if save_location != None:
                save_plot(figure, save_location, plot_name)
            else:
                figure.show()
            print("Plotted: " + plot_name)
    
def make_2d_plot(axis1:axis, axis2:axis) -> plot:

    figure = plot.plotPoints2d(
        xs=axis1,
        ys=axis2,
        legend=True,
        num_of_series=len(axis2.data),
        series_labels=[f"{i * binColer_size}" for i in range(len(axis2.data))]
    )

    return figure


if __name__ == '__main__':
    
    user_dir = os.path.expanduser("~")

    data_path = os.path.join(user_dir, "ray_results")
    save_path = os.path.join(user_dir, "ray_plots")

    for entry in os.scandir(path=data_path):
        if entry.is_dir():
            try:
                print("Plotting: " + entry.name)
                make_experiment_plots(entry.name, os.path.join(save_path, entry.name))
            except:
                print("Failed to plot: " + entry.name)


    # title = "cl_pure_res_large_cub_200"
    # make_plots(title)#, os.path.expanduser("~/ray_results/plots/") + title + "/")

    # print("done")
