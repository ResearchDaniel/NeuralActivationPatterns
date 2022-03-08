import json
from pathlib import Path
from PIL import Image
import numpy as np
import nap
import nap.cache
import util
import pandas as pd
import tensorflow as tf
import pandas as pd

EXPORT_LOCATION = Path("activation_cluster_explorer/backend/data")


def export_config(image_dir, model_name, destination=EXPORT_LOCATION):
    config = {
        "data_path": image_dir
    }
    Path(destination, model_name).mkdir(parents=True, exist_ok=True)
    with open(Path(destination, model_name, "config.json"), 'w') as outfile:
        json.dump(config, outfile)


def export_dataset(file_names, labels, predictions, model_name, destination=EXPORT_LOCATION):
    path = Path(destination, model_name)
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file_name": file_names, "label": labels,
                 "prediction": predictions}).to_pickle(Path(path, "dataset.pkl"))


def export_patterns(model, model_name, X, layers, filters, agg_func, destination=EXPORT_LOCATION):
    def export_pattern(path, patterns, info):
        path.mkdir(parents=True, exist_ok=True)
        patterns.to_pickle(Path(path, "patterns.pkl"))
        info.to_pickle(Path(path, "patterns_info.pkl"))
    for layer in layers:
        path = Path(destination, model_name, "layers", str(layer))
        patterns, info = nap.cache.get_layer_patterns(
            X, model, model_name, layer, agg_func)
        export_pattern(path, patterns, info)
        if layer in filters:
            for filter in filters[layer]:
                path = Path(destination, model_name, "layers", str(layer), str(filter))
                patterns, info = nap.cache.get_filter_patterns(
                X, model, model_name, layer, filter)
                export_pattern(path, patterns, info)

def export_averages(model, model_name, X, layers, filters, agg_func, destination=EXPORT_LOCATION):
    def export_pattern_averages(base_path, patterns):
        sorted_patterns = nap.sort(patterns)
        for pattern_id, pattern in sorted_patterns.groupby('patternId'):
            if pattern_id == -1:
                continue
            path = Path(base_path, str(int(pattern_id)))
            path.mkdir(parents=True, exist_ok=True)
            to_average = util.filter_tf_dataset(X, pattern.index)
            avg = tf.keras.layers.Average()(to_average).numpy()
            export_image(path, "average", avg)

    for layer in layers:
        patterns, _ = nap.cache.get_layer_patterns(
            X, model, model_name, layer, agg_func)
        layer_path = Path(destination, model_name, "layers", str(layer))
        export_pattern_averages(layer_path, patterns)         
        if layer in filters:
            for filter in filters[layer]:
                patterns, _ = nap.cache.get_filter_patterns(
                    X, model, model_name, layer, filter)
                export_pattern_averages(Path(layer_path, 'filters', str(filter)), patterns)  
                


def export_image(path, name, array):
    image = np.squeeze(array)
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = Image.fromarray((image * 255).astype(np.uint8), 'L')
    else:
        image = Image.fromarray((image * 255).astype(np.uint8), 'RGB')
    image.save(Path(path, f"{name}.jpeg"))


def export_all(model, model_name, X, y, predictions, file_names, layers, filters, image_dir, agg_func, destination=EXPORT_LOCATION):
    export_config(image_dir, model_name, destination)
    export_dataset(file_names, y, predictions, model_name, destination)
    export_patterns(model, model_name, X, layers, filters, agg_func, destination)
    export_averages(model, model_name, X, layers, filters, agg_func, destination)
