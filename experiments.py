"""Run NAP experiments for a series of models."""
import subprocess
import sys

models = ["mnist"]
# ["mixed5", "mixed6", "mixed10", "predictions"]}
#layers = {"inception_v3": ["predictions"]}
layers = {"mnist": ["conv2d_1"]}
data_sets = {"mnist": "mnist", "cifar10": "cifar10",
             "inception_v3": "imagenet2012_subset"}
splits = {"inception_v3": "validation"}
layer_aggregations = ["mean"]  # , "mean_std", 'none']
#data_set_sizes = range(12811, 14000, 14000)
data_set_sizes = range(2000, 3000, 14000)
minimum_pattern_sizes = range(6, 7, 1)

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
                cmd = (
                    f"python model_analysis.py --model {model} --data_set {data_sets[model]}"
                    f" --size {data_set_size} --layer_aggregation {layer_aggregation}"
                    f" --minimum_pattern_size {minimum_pattern_size}") + FILTERS
                if model in splits:
                    cmd += f" --split {splits[model]}"
                if model in layers:
                    for layer in layers[model]:
                        run_command(cmd + f" --layer {layer}")
                else:
                    run_command(cmd)
