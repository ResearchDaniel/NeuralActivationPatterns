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
import shutil
import pickle

EXPORT_LOCATION = Path("activation_cluster_explorer/backend/data")


def export_config(image_dir, model, model_name, destination=EXPORT_LOCATION):
    config = {
        "data_path": image_dir,
        "layers": [layer.name for layer in model.layers]
    }
    Path(destination, model_name).mkdir(parents=True, exist_ok=True)
    with open(Path(destination, model_name, "config.json"), 'w') as outfile:
        json.dump(config, outfile)


def export_dataset(file_names, labels, predictions, model_name, destination=EXPORT_LOCATION):
    path = Path(destination, model_name)
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file_name": file_names, "label": labels,
                 "prediction": predictions}).to_pickle(Path(path, "dataset.pkl"))


def export_patterns(model, model_name, X, layers, filters, layer_aggregation, filter_aggregation, destination=EXPORT_LOCATION):
    def export_pattern(path, patterns, info, statistics):
        path.mkdir(parents=True, exist_ok=True)
        patterns.to_pickle(Path(path, "patterns.pkl"))
        info.to_pickle(Path(path, "patterns_info.pkl"))
        pickle.dump(statistics, open(
            Path(path, "patterns_statistics.pkl"), "wb"))
    for layer in layers:
        path = Path(destination, model_name, "layers", str(layer))
        patterns, info = nap.cache.get_layer_patterns(
            X, model, model_name, layer, layer_aggregation)
        statistics = nap.cache.get_layer_patterns_activation_statistics(
            X, model, model_name, layer, layer_aggregation)
        export_pattern(path, patterns, info, statistics)
        if layer in filters:
            for filter in filters[layer]:
                path = Path(destination, model_name,
                            "layers", str(layer), "filters", filter_aggregation.__class__.__name__, str(filter))
                patterns, info = nap.cache.get_filter_patterns(
                    X, model, model_name, layer, filter, filter_aggregation)
                statistics = nap.cache.get_filter_patterns_activation_statistics(
                    X, model, model_name, layer, filter, filter_aggregation)
                export_pattern(path, patterns, info, statistics)


def export_statistics(model, model_name, X, layers, filters, destination=EXPORT_LOCATION):
    for layer in layers:
        path = Path(destination, model_name, "layers", str(layer))
        path.mkdir(parents=True, exist_ok=True)
        stats = nap.cache.get_layer_activation_statistics(
            X, model, model_name, layer)
        pickle.dump(stats, open(Path(path, "layer_statistics.pkl"), "wb"))
        if layer in filters:
            for filter in filters[layer]:
                path = Path(destination, model_name,
                            "layers", str(layer), "filters", str(filter))
                path.mkdir(parents=True, exist_ok=True)
                stats = nap.cache.get_filter_activation_statistics(
                    X, model, model_name, layer, filter)
                pickle.dump(stats, open(
                    Path(path, "filter_statistics.pkl"), "wb"))


def export_averages(image_dir, file_names, model, model_name, X, layers, filters, layer_aggregation, filter_aggregation,  destination=EXPORT_LOCATION):
    def export_pattern_averages(base_path, patterns):
        sorted_patterns = nap.sort(patterns)
        for pattern_id, pattern in sorted_patterns.groupby('patternId'):
            if pattern_id == -1:
                continue
            path = Path(base_path, str(int(pattern_id)))
            path.mkdir(parents=True, exist_ok=True)
            size = list(X.element_spec.shape)
            avg = util.average_images(
                image_dir, file_names, pattern.index, size)
            avg.save(Path(path, "average.jpeg"))

    for layer in layers:
        patterns, _ = nap.cache.get_layer_patterns(
            X, model, model_name, layer, layer_aggregation)
        layer_path = Path(destination, model_name, "layers", str(layer))
        export_pattern_averages(layer_path, patterns)
        if layer in filters:
            for filter in filters[layer]:
                patterns, _ = nap.cache.get_filter_patterns(
                    X, model, model_name, layer, filter, filter_aggregation)
                filter_path = Path(
                    layer_path, 'filters', filter_aggregation.__class__.__name__, str(filter))
                export_pattern_averages(filter_path, patterns)


def export_image(path, name, array):
    image = np.squeeze(array)
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = Image.fromarray((image * 255).astype(np.uint8), 'L')
    else:
        image = Image.fromarray((image * 255).astype(np.uint8), 'RGB')
    image.save(Path(path, f"{name}.jpeg"))


def export_max_activations(image_dir, file_names, model, model_name, X, layers, filters, N, destination=EXPORT_LOCATION):
    # Copies the N most activating input images into destination/layer/max_activations
    def export_activations(base_path, max_activations):
        base_path.mkdir(parents=True, exist_ok=True)
        for order, max_activation in enumerate(max_activations):
            path = Path(base_path, f"{order}_{file_names[max_activation]}")
            old_name = f"{image_dir}/{file_names[max_activation]}"
            shutil.copy(old_name, path)

    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        activations, f = nap.cache.get_layer_activations(
            X, model, model_name, layer)
        # [()] fetches all data into memory. Needed because slicing the filter is super-slow in hdf5
        activations = activations[()]
        max_activations = ap.layer_max_activations(
            layer, activations=activations, nSamplesPerLayer=N)
        layer_path = Path(destination, model_name, "layers",
                          str(layer), "max_activations")
        export_activations(layer_path, max_activations)

        if layer in filters:
            for filter in filters[layer]:
                max_activations = ap.filter_max_activations(
                    layer, filter, activations=activations, nSamplesPerLayer=N)
                export_activations(
                    Path(layer_path, 'filters', str(filter)), max_activations)
        f.close()


def export_all(model, model_name, X, y, predictions, file_names, layers, filters, image_dir, layer_aggregation, filter_aggregation, destination=EXPORT_LOCATION):
    export_config(image_dir, model, model_name, destination)
    export_dataset(file_names, y, predictions, model_name, destination)
    export_patterns(model, model_name, X, layers,
                    filters, layer_aggregation, filter_aggregation, destination)
    export_statistics(model, model_name, X, layers,
                      filters, destination)
    export_averages(image_dir, file_names, model, model_name,
                    X, layers, filters, layer_aggregation, filter_aggregation, destination)
