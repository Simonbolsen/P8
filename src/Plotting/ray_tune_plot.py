from plotting_util import plotPoints
from ray import tune
import plotting_util as plot
import results_analysis_util as analysis
import math

parameters = {"lr": "Learning Rate log_10(lr)", "d": "Dimensions d", "i": "Training Iterations i", 
              "e": "Epochs e", "n" : "Linear Num n", "s": "Linear Size s", "b" : "Batch Size log_2(b)", "c": "Channels c"}
param1 = "e"
param2 = "d"
paramColor = "e"

bin1 = []
bin2 = []

bin1_size = 0.5
bin2_size = 10
binColer_size = 3

def insert_append(l, i, e):
    if i < len(l):
        l[i].append(e)
    else:
        while i < len(l):
            l.append([])
        l.append([e])

results = tune.ExperimentAnalysis("~/ray_results/mnist_classification_test_comparison_loss", 
                                  default_metric="accuracy", 
                                  default_mode="max")
best_results = analysis.best_iterations_per_trial(results)

x1 = []
x2 = []
accuracies = []

visual_func = lambda x: (1 - math.sqrt(1 - x**2))
inv_visual_func = lambda x: math.sqrt(1 - (1 -  x)**2)
failures = 0
total = 0

for k, result in best_results.items():
    if "config" in result.keys():
        params = {"lr": math.log(result["config"]["lr"], 10)} #
        #params["n"] = result["config"]["linear_n"]
        params["d"] = result["config"]["d"]
        params["i"] = result["data"]["training_iteration"]
        params["e"] = result["config"]["num_of_epochs"]
        #params["s"] = result["config"]["linear_size"]
        params["b"] = math.log(result["config"]["batch_size"],2)
        params["c"] = result["config"]["channels"]
        accuracy = result["data"]["accuracy"]
           
        if True: #params["lr"] < -7 and params["lr"] > -9 and params["c"] > 150:
            if accuracy < 0.2:
                failures += 1

            total += 1

            accuracy = visual_func(accuracy)            

            index = int(params[paramColor] / binColer_size)
            insert_append(x1, index, params[param1])
            insert_append(x2, index, params[param2])
            insert_append(accuracies, index, accuracy)

print(f"Failures: {failures}/{total}")

# print(results.results)
plot.plotPoints(x1, x2, accuracies, [parameters[param1], parameters[param2], "Accuracy a"], legend = False,
                num_of_series=len(accuracies), series_labels=[f"{i * binColer_size}" for i in range(len(accuracies))])