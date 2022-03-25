""" Scan backend data dir for models and create a DataFrame with settings and number of patterns. """
from pathlib import Path
import os
import re
import json
import pandas as pd


# Export data for producing paper figures

DATA_DIR = '../magnifying_glass/backend/data'

m = []
l = []
ds = []
normalized = []
splt = []
sizes = []
layer_agg = []
min_pattern_size = []
num_patterns = []
min_samples = []
cluster_selection_epsilons = []
cluster_selection_methods = []
for models in os.scandir(DATA_DIR):

    model_dir = models.name

    # cifar10_cifar10_test_1000_norm_MeanAggregation_min_pattern_5_
    # min_samples_5_cluster_selection_epsilon_1e-01_leaf
    matches = re.match(
        (r"(inception_v3|.*)_(imagenet2012_subset|.*)_(.*)_(.*)_(.*)_(.*)_(.*)_(.*)_(.*)"
         r"_(.*)_(.*)_(.*)_(.*)_(.*)_(.*)_(.*)_(.*)"),
        model_dir)
    model = matches.group(1)
    dataset = matches.group(2)
    split = matches.group(3)
    input_data_size = int(matches.group(4))
    norm = (matches.group(5))
    layer_aggregation = matches.group(6)
    minimum_pattern_size = int(matches.group(9))
    min_sample = int(matches.group(12))
    cluster_selection_epsilon = float(matches.group(16))
    cluster_selection_method = matches.group(17)

    model_path = Path(DATA_DIR, model_dir, "layers")
    layer_dirs = [f.name for f in os.scandir(model_path) if f.is_dir()]

    with open(Path(DATA_DIR, model_dir, 'config.json'), encoding='utf8') as json_file:
        config = json.load(json_file)
    layers = list(
        filter(lambda layer: layer in layer_dirs, config["layers"]))

    for layer in layers:
        patterns_info = pd.read_pickle(
            Path(
                DATA_DIR, model_dir, 'layers', layer,
                'patterns_info.pkl'))

        m.append(model)
        ds.append(dataset)
        splt.append(split)
        sizes.append(input_data_size)
        normalized.append(norm)
        layer_agg.append(layer_aggregation)
        min_pattern_size.append(minimum_pattern_size)
        min_samples.append(min_sample)
        cluster_selection_epsilons.append(cluster_selection_epsilon)
        cluster_selection_methods.append(cluster_selection_method)
        l.append(layer)
        num_patterns.append(len(patterns_info))


df = pd.DataFrame({"Model": m, "Data set": ds, "Split": splt, "Input size": sizes,
                   "Normalized": normalized,
                   "Layer aggregation": layer_agg, "Minimum pattern size": min_pattern_size,
                   "Minimum samples": min_samples, "Epsilon": cluster_selection_epsilons,
                   "Method": cluster_selection_methods,
                   "Layer": l, "Number of patterns": num_patterns})
print(df)
df.to_feather("model_pattern_count.feather")
