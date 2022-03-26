"""Export the results of an NAP analysis run."""
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import nap
import nap.cache
import util

EXPORT_LOCATION = Path("magnifying_glass/backend/data")


def export_config(image_dir, neural_activation, export_name, destination=EXPORT_LOCATION):
    config = {
        "data_path": image_dir,
        "layers": [layer.name for layer in neural_activation.model.layers],
        "layer_aggregation": neural_activation.layer_aggregation.__class__.__name__,
        "filter_aggregation": neural_activation.filter_aggregation.__class__.__name__,
        "min_pattern_size": neural_activation.min_pattern_size,
        "min_samples": neural_activation.min_samples,
        "cluster_selection_epsilon": neural_activation.cluster_selection_epsilon,
        "cluster_selection_method": neural_activation.cluster_selection_method
    }
    Path(destination, export_name).mkdir(parents=True, exist_ok=True)
    with open(Path(destination, export_name, "config.json"), 'w', encoding="utf8") as outfile:
        json.dump(config, outfile)


def export_dataset(file_names, labels, predictions, export_name, destination=EXPORT_LOCATION):
    path = Path(destination, export_name)
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file_name": file_names, "label": labels,
                 "prediction": predictions}).to_pickle(Path(path, "dataset.pkl"))


def export_patterns(neural_activation, model_name, export_name, input_data, layers, filters,
                    destination=EXPORT_LOCATION):
    def export_pattern(path, patterns, info, statistics):
        path.mkdir(parents=True, exist_ok=True)
        patterns.to_pickle(Path(path, "patterns.pkl"))
        info.to_pickle(Path(path, "patterns_info.pkl"))
        with open(Path(path, "patterns_statistics.pkl"), "wb") as outfile:
            pickle.dump(statistics, outfile)
    for layer in layers:
        path = Path(destination, export_name, "layers", str(layer))
        patterns, info = nap.cache.get_layer_patterns(
            input_data, neural_activation, model_name, layer)
        statistics = nap.cache.get_layer_patterns_activation_statistics(
            input_data, neural_activation, model_name, layer)
        export_pattern(path, patterns, info, statistics)
        if layer in filters:
            for model_filter in filters[layer]:
                path = Path(
                    destination, export_name, "layers", str(layer),
                    "filters", neural_activation.filter_aggregation.__class__.__name__,
                    str(model_filter))
                patterns, info = nap.cache.get_filter_patterns(
                    input_data, neural_activation, model_name, layer, model_filter)
                statistics = nap.cache.get_filter_patterns_activation_statistics(
                    input_data, neural_activation, model_name, layer, model_filter)
                export_pattern(path, patterns, info, statistics)


def export_statistics(neural_activation, model_name, export_name, input_data, layers, filters,
                      destination=EXPORT_LOCATION):
    for layer in layers:
        path = Path(destination, export_name, "layers", str(layer))
        path.mkdir(parents=True, exist_ok=True)
        stats = nap.cache.get_layer_activation_statistics(
            input_data, neural_activation, model_name, layer)
        with open(Path(path, "layer_statistics.pkl"), "wb") as outfile:
            pickle.dump(stats, outfile)
        if layer in filters:
            for model_filter in filters[layer]:
                path = Path(destination, export_name,
                            "layers", str(layer), "filters", str(model_filter))
                path.mkdir(parents=True, exist_ok=True)
                stats = nap.cache.get_filter_activation_statistics(
                    input_data, neural_activation, model_name, layer, model_filter)
                with open(Path(path, "filter_statistics.pkl"), "wb") as outfile:
                    pickle.dump(stats, outfile)


def export_pattern_averages(base_path, patterns, file_names, image_dir, input_data):
    sorted_patterns = nap.sort(patterns)
    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        if pattern_id == -1:
            continue
        path = Path(base_path, str(int(pattern_id)))
        path.mkdir(parents=True, exist_ok=True)
        size = list(input_data.element_spec.shape)
        avg = util.average_images(
            image_dir, file_names, pattern.index, size)
        avg.save(Path(path, "average.jpeg"))


def export_averages(
        image_dir, file_names, neural_activation, model_name, export_name, input_data, layers,
        filters, destination=EXPORT_LOCATION):
    for layer in layers:
        patterns, _ = nap.cache.get_layer_patterns(
            input_data, neural_activation, model_name, layer)
        layer_path = Path(destination, export_name, "layers", str(layer))
        export_pattern_averages(layer_path, patterns,
                                file_names, image_dir, input_data)
        export_filter_averages(
            image_dir, file_names, neural_activation, model_name, input_data, filters,
            layer, layer_path)


def export_filter_averages(
        image_dir, file_names, neural_activation, model_name, input_data, filters,
        layer, layer_path):
    if layer in filters:
        for model_filter in filters[layer]:
            patterns, _ = nap.cache.get_filter_patterns(
                input_data, neural_activation, model_name, layer, model_filter)
            filter_path = Path(layer_path, 'filters',
                               neural_activation.filter_aggregation.__class__.__name__,
                               str(model_filter))
            export_pattern_averages(filter_path,
                                    patterns, file_names, image_dir, input_data)


def export_image(path, name, array):
    image = np.squeeze(array)
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = Image.fromarray((image * 255).astype(np.uint8), 'L')
    else:
        image = Image.fromarray((image * 255).astype(np.uint8), 'RGB')
    image.save(Path(path, f"{name}.jpeg"))


def export_activations(base_path, max_activations, file_names, image_dir):
    # Copies the N most activating input images into destination/layer/max_activations
    base_path.mkdir(parents=True, exist_ok=True)
    for order, max_activation in enumerate(max_activations):
        path = Path(base_path, f"{order}_{file_names[max_activation]}")
        old_name = f"{image_dir}/{file_names[max_activation]}"
        shutil.copy(old_name, path)


def export_max_activations(
        image_dir, file_names, activation_pattern, model_name, export_name, input_data, layers,
        filters, number, destination=EXPORT_LOCATION):
    for layer in layers:
        activations, outfile = nap.cache.get_layer_activations(
            input_data, activation_pattern.model, model_name, layer)
        # [()] fetches all data into memory. Needed because slicing the filter is super-slow in hdf5
        layer_path = Path(destination, export_name, "layers",
                          str(layer), "max_activations")
        export_activations(layer_path, activation_pattern.layer_max_activations(
            layer, activations=activations[()], samples_per_layer=number), file_names, image_dir)

        if layer in filters:
            for model_filter in filters[layer]:
                export_activations(Path(layer_path, 'filters', str(model_filter)),
                                   activation_pattern.filter_max_activations(
                                       layer, model_filter, activations=activations,
                                       samples_per_layer=number),
                                   file_names, image_dir)
        outfile.close()


def export_all(model_name, input_data, labels, predictions, file_names, layers, filters, image_dir,
               neural_activation, n_max_activations,
               destination=EXPORT_LOCATION):
    # Differentiate between model and export name to be able to cache activation data
    # between configs.
    export_name = (f'{model_name}_{neural_activation.layer_aggregation.__class__.__name__}'
                   f'_min_pattern_{neural_activation.min_pattern_size}'
                   f'_min_samples_{neural_activation.min_samples}'
                   f'_cluster_selection_epsilon_{neural_activation.cluster_selection_epsilon:1.0e}'
                   f'_{neural_activation.cluster_selection_method}')

    export_config(image_dir, neural_activation, export_name, destination)
    export_dataset(file_names, labels, predictions, export_name, destination)
    export_patterns(neural_activation, model_name, export_name, input_data, layers,
                    filters, destination)
    # We currently do not use these statisitics and they take a LONG time time compute.
    # Remove comment when they are needed
    # export_statistics(neural_activation, model_name, export_name, input_data, layers,
    #                   filters, destination)
    export_averages(image_dir, file_names, neural_activation, model_name,
                    export_name, input_data, layers, filters, destination)

    if n_max_activations > 0:
        export_max_activations(
            image_dir, file_names, neural_activation, model_name, export_name,
            input_data, layers, filters, number=n_max_activations)
