from plotting_util import plotPoints
from ray import tune

results = tune.ExperimentAnalysis(experiment_checkpoint_path="~/ray_results/mnist_initial_test", 
                                  default_metric="accuracy", 
                                  default_mode="max")
trials = results.trials
# best_trials = results.trials
# best_trial = results.best_trial
# best = results.best_result
lrs = []
dimensions = []
accuracies = []

for result in results.results:
    print(result)

    # results.tr
    # lrs.append(i["config"]["lr"])
    # dimensions.append(i["config"]["d"])
    # accuracies.append(i["accuracy"])

# print(results.results)
print(lrs)
print(dimensions)
print(accuracies)