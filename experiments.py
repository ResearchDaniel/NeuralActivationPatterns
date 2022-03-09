

import subprocess
import sys

models = ["mnist"]
layers = {"mnist": "conv2d_1", "inception_v3": "mixed8"}
data_sets = {"mnist": "mnist", "cifar10": "cifar10", "inception_v3": "imagenet2012_subset"}
splits = {"inception_v3": "train"}
aggregations = ["mean"]#, "mean_std", 'none']
data_set_sizes = range(2000, 10000, 10000)
# Batch processing for computing NAPs

# Evaluate scaling with respect to number of inputs
for model in models:
    for aggregation in aggregations:
        for data_set_size in data_set_sizes:
            try:
                cmd = f"python model_analysis.py --model {model} --data_set {data_sets[model]} --size {data_set_size} --aggregation {aggregation}"
                if model in layers:
                    cmd += f" --layer {layers[model]}"
                if model in splits:     
                    cmd += f" --split {splits[model]}"                    
                retcode = subprocess.call(cmd, shell=True)
                if retcode < 0:
                    print("Child was terminated by signal", -retcode, file=sys.stderr)
                else:
                    print("Child returned", retcode, file=sys.stderr)
            except OSError as e:
                print("Execution failed:", e, file=sys.stderr)
                retcode = call("model_analysis.py" + " myarg", shell=True)