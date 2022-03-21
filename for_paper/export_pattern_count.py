from pathlib import Path
import os
import pandas as pd
import json
import re

# Export data for producing paper figures

DATA_DIR = 'nap_microscope/backend/data'

m = []
l = []
ds = []
splt = []
sizes = []
layer_agg = []
min_pattern_size = []
num_patterns = []
for models in os.scandir(DATA_DIR):

    model_dir = models.name  # 'cifar10_cifar10_test_2000_MeanAggregation_min_pattern_5'

    matches = re.match(
        r"(inception_v3|.*)_(imagenet2012_subset|.*)_(.*)_(.*)_(.*)_(.*)_(.*)_(.*)", model_dir)
    model = matches.group(1)
    dataset = matches.group(2)
    split = matches.group(3)
    input_data_size = int(matches.group(4))
    layer_aggregation = matches.group(5)
    minimum_pattern_size = int(matches.group(8))

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
        layer_agg.append(layer_aggregation)
        min_pattern_size.append(minimum_pattern_size)
        l.append(layer)
        num_patterns.append(len(patterns_info))


df = pd.DataFrame({"Model": m, "Data set": ds, "Split": splt, "Input size": sizes,
                   "Layer aggregation": layer_agg, "Minimum pattern size": min_pattern_size,
                   "Layer": l, "Number of patterns": num_patterns})
print(df)
# df.to_csv("model_pattern_count.csv")
df.to_feather("model_pattern_count.feather")
