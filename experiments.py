

import subprocess
import sys
 

# Batch processing for computing NAPs

 # Evaluate scaling with respect to number of inputs
for data_set_size in range(10000, 60000, 10000):
    try:
        retcode = subprocess.call(f"python model_analysis.py --size {data_set_size}", shell=True)
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
        else:
            print("Child returned", retcode, file=sys.stderr)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)
        retcode = call("model_analysis.py" + " myarg", shell=True)