"""Run NAP experiments for a series of models."""
import subprocess
import sys

models = ["mnist", "cifar10"]
# ["mixed5", "mixed6", "mixed10", "predictions"]}
layers = {"inception_v3": ["mixed1", "mixed2", "mixed3", "mixed4", "mixed5",
                           "mixed6", "mixed7", "mixed8", "mixed9", "mixed10", "predictions"]}
#layers = {"mnist": ["conv2d_1"]}
data_sets = {"mnist": "mnist", "cifar10": "cifar10",
             "inception_v3": "imagenet2012_subset"}
splits = {"inception_v3": "validation"}
layer_aggregations = ["mean"]  # , "mean_std", 'none']
#data_set_sizes = range(1000, 12811, 1000)
data_set_sizes = range(10000, 11000, 1000)
minimum_pattern_sizes = [5, 10]  # range(5, 6, 1)

cluster_selection_epsilons = [0]  # + [pow(10, exponent) for exponent in range(-2, -1, 1)]

cluster_selection_methods = ["leaf"]
cluster_min_samples = [5, 10]

# FILTERS = " --all_filters "
#FILTERS = " --filter_range 0 2 "
FILTERS = ""
# Batch processing for computing NAPs


def run_command(comd):
    try:
        retcode = subprocess.call(comd, shell=True)
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
        else:
            print("Child returned", retcode, file=sys.stderr)
    except OSError as err:
        print("Execution failed:", err, file=sys.stderr)
        retcode = subprocess.call("model_analysis.py" + " myarg", shell=True)


# Evaluate scaling with respect to number of inputs
for model in models:
    for layer_aggregation in layer_aggregations:
        for data_set_size in data_set_sizes:
            for minimum_pattern_size in minimum_pattern_sizes:
                for cluster_selection_epsilon in cluster_selection_epsilons:
                    for cluster_min_sample in cluster_min_samples:
                        for cluster_selection_method in cluster_selection_methods:
                            cmd = (
                                f"python model_analysis.py --model {model}"
                                f" --data_set {data_sets[model]}"
                                f" --size {data_set_size} --layer_aggregation {layer_aggregation}"
                                f" --minimum_pattern_size {minimum_pattern_size}"
                                f" --cluster_min_samples {cluster_min_sample}"
                                f" --cluster_selection_epsilon {cluster_selection_epsilon}"
                                f" --cluster_selection_method {cluster_selection_method}") + FILTERS
                            if model in splits:
                                cmd += f" --split {splits[model]}"
                            if model in layers:
                                for layer in layers[model]:
                                    run_command(cmd + f" --layer {layer}")
                            else:
                                run_command(cmd)
