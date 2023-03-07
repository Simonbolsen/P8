

def best_iterations_per_trial(experiment, metric="accuracy"):
    best_iterations = {}
    trials = experiment.trial_dataframes

    for dir in trials:
        trial = trials[dir]
        best_id = trial[metric].idxmax()
        best = trial.iloc[best_id]
        
        trial_id = best["trial_id"]
        config = experiment.results[trial_id]["config"]

        best_iterations[trial_id] = {"data" : best,
                                     "config" : config}
    
    return best_iterations





