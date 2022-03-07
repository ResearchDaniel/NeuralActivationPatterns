

import subprocess
import sys
 
models = ["mnist"]
layers = ["conv2D_1"]
aggregations = ["mean_std"]
data_set_sizes = range(10000, 60000, 10000)
# Batch processing for computing NAPs

 # Evaluate scaling with respect to number of inputs
for model, layer in zip(models, layers):
    for aggregation in aggregations:
        for data_set_size in data_set_sizes:
            try:
                retcode = subprocess.call(f"python model_analysis.py --model {model} --layers {layer} --size {data_set_size} --aggregation {aggregation}", shell=True)
                if retcode < 0:
                    print("Child was terminated by signal", -retcode, file=sys.stderr)
                else:
                    print("Child returned", retcode, file=sys.stderr)
            except OSError as e:
                print("Execution failed:", e, file=sys.stderr)
                retcode = call("model_analysis.py" + " myarg", shell=True)