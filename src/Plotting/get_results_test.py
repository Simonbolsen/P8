from plotting_util import plotPoints
from ray import tune
import plotting_util as plot
import results_analysis_util as analysis
import math

results = tune.ExperimentAnalysis("~/ray_results/mnist_initial_few_shot_test3", 
                                  default_metric="accuracy", 
                                  default_mode="max")
best_results = analysis.best_iterations_per_trial(results)

# best_trials = results.trials
# best_trial = results.best_trial
# best = results.best_result
num_of_bins = 10
max_value = 30
param1 = [[] for _ in range(num_of_bins)]
param2 = [[] for _ in range(num_of_bins)]
accuracies = [[] for _ in range(num_of_bins)]

visual_func = lambda x: (1 - math.sqrt(1 - x**2))
inv_visual_func = lambda x: math.sqrt(1 - (1 -  x)**2)
failures = 0
total = 0

for k, result in best_results.items():
    if "config" in result.keys():
        lr = math.log(result["config"]["lr"],10)
        if lr < -3 and lr > -5:
            iterations = result["data"]["training_iteration"]
            epochs = result["config"]["num_of_epochs"]
            channels = result["config"]["channels"]
            dimension = result["config"]["d"]
            accuracy = result["data"]["accuracy"]
            linear_n = result["config"]["linear_n"]
            linear_size = result["config"]["linear_size"]
            batch_size = math.log(result["config"]["batch_size"],2)

            if accuracy < 0.2:
                failures += 1

            total += 1

            accuracy = visual_func(accuracy)            

            index = int((epochs / max_value) * num_of_bins)
            index = num_of_bins - 1 if index >= num_of_bins else index
            param1[index].append(batch_size)
            param2[index].append(channels)
            accuracies[index].append(accuracy)

print(f"Failures: {failures}/{total}")

# print(results.results)
plot.plotPoints(param1, param2, accuracies, ["Learning Rate", "Dimensions", "Accuracy v(a)"], 
                num_of_series=num_of_bins, series_labels=[f"{i * max_value / num_of_bins}" for i in range(num_of_bins)])