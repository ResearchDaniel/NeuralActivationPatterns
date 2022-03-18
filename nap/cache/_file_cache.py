"""Provide caching for neural activation patterns."""
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import nap

CACHE_LOCATION = Path("results")


def activations_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_activations.h5')


def activations_agg_path(destination, model_name, layer, layer_aggregation):
    return Path(destination, model_name, layer,
                f'layer_activations_{layer_aggregation.__class__.__name__}.h5')


def activation_statistics_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_activation_statistics.pkl')


def layer_patterns_activation_statistics_path(destination, model_name, layer, min_pattern_size):
    return Path(destination, model_name, layer, f'layer_patterns_min_size_{min_pattern_size}_activation_statistics.pkl')


def filter_patterns_activation_statistics_path(destination, model_name, layer, filter_index,
                                               filter_aggregation, min_pattern_size):
    return Path(destination, model_name, layer, 'filters', filter_aggregation.__class__.__name__,
                str(filter_index), f'filter_patterns_min_size_{min_pattern_size}_activation_statistics.pkl')


def filter_activation_statistics_path(destination, model_name, layer, filter_index):
    return Path(destination, model_name, layer, 'filters', str(filter_index),
                'filter_activation_statistics.pkl')


def layer_patterns_path(destination, model_name, layer, min_pattern_size):
    return Path(destination, model_name, layer, f'layer_patterns_min_size_{min_pattern_size}.h5')


def layer_patterns_info_path(destination, model_name, layer, min_pattern_size):
    return Path(destination, model_name, layer, f'layer_patterns_min_size_{min_pattern_size}_info.h5')


def filter_patterns_path(destination, model_name, layer, filter_index, filter_aggregation, min_pattern_size):
    return Path(destination, model_name, layer, 'filters', filter_aggregation.__class__.__name__,
                str(filter_index), f'filter_patterns_min_size_{min_pattern_size}.h5')


def filter_patterns_info_path(destination, model_name, layer, filter_index, filter_aggregation, min_pattern_size):
    return Path(destination, model_name, layer, 'filters', filter_aggregation.__class__.__name__,
                str(filter_index), f'filter_patterns_min_size_{min_pattern_size}_info.h5')


# pylint: disable=R0914
def export_activations(input_data, model, model_name, layers, destination=CACHE_LOCATION, mode='w'):
    activation_patterns = nap.NeuralActivationPattern(model)
    for layer in layers:
        act_path = activations_path(destination, model_name, layer)
        act_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(act_path, mode) as file_handle:
            activations = activation_patterns.layer_activations(
                layer, input_data.take(1).batch(1))
            output_shape = list(activations.shape)
            total_num_inputs = input_data.cardinality().numpy()
            output_shape[0] = total_num_inputs
            bytes_per_float = 4
            num_activations_in_one_mb = min(total_num_inputs, max(
                1, int(1000000 / (np.prod(output_shape[1:])*bytes_per_float))))
            chunk_size = tuple([num_activations_in_one_mb] + output_shape[1:])
            dset = file_handle.create_dataset(
                "activations", output_shape, compression="gzip", chunks=chunk_size)
            iterator = 0
            for data_set in input_data.batch(1000).cache().prefetch(tf.data.AUTOTUNE):
                activation_patterns = nap.NeuralActivationPattern(model)
                num_inputs = data_set.shape[0]
                activations = activation_patterns.layer_activations(
                    layer, data_set)
                dset[iterator:iterator+num_inputs] = activations
                iterator += num_inputs


# pylint: disable=R0914
def export_layer_aggregation(input_data, model, model_name, layers, layer_aggregation,
                             destination=CACHE_LOCATION):
    activation_patterns = nap.NeuralActivationPattern(model)
    for layer in layers:
        act_path = activations_path(destination, model_name, layer)
        if not act_path.exists():
            export_activations(input_data, model, model_name, [
                               layer], destination)
        with h5py.File(act_path, 'r') as f_act:
            activations = f_act["activations"]
            # pylint: disable=E1101
            output_shape = list(activations.shape)
            agg_shape = layer_aggregation.shape(output_shape[1:])
            total_num_inputs = output_shape[0]
            agg_path = activations_agg_path(
                destination, model_name, layer, layer_aggregation)
            with h5py.File(agg_path, 'w') as f_agg:
                agg_size = np.prod(agg_shape)
                bytes_per_float = 4
                num_activations_in_one_mb = min(total_num_inputs, max(
                    1, int(1000000 / (agg_size*bytes_per_float))))
                if isinstance(agg_shape, list):
                    all_agg_shape = [total_num_inputs] + agg_shape
                    chunk_size = (num_activations_in_one_mb) + tuple(agg_shape)
                else:
                    all_agg_shape = [total_num_inputs] + [agg_shape]
                    chunk_size = (num_activations_in_one_mb, agg_shape)

                dset_aggregated = f_agg.create_dataset(
                    "activations", all_agg_shape, compression="gzip", chunks=chunk_size)
                i = 0
                for chunk in activations.iter_chunks():
                    data = activations[chunk]
                    dset_aggregated[i:i+data.shape[0]] = [
                        layer_aggregation.aggregate(
                            activation_patterns.layer(layer), activation)
                        for activation in data]
                    i += data.shape[0]


def activation_statistics(activations, axis):
    means = []
    mins = []
    maxs = []
    q1s = []
    q3s = []
    if axis is None:
        means.append(np.mean(activations))
        mins.append(np.min(activations))
        maxs.append(np.max(activations))
        q1s.append(np.quantile(activations, 0.25))
        q3s.append(np.quantile(activations, 0.75))
    else:
        for feature in range(activations.shape[axis]):
            f_act = activations[..., feature]
            means.append(np.mean(f_act))
            mins.append(np.min(f_act))
            maxs.append(np.max(f_act))
            q1s.append(np.quantile(f_act, 0.25))
            q3s.append(np.quantile(f_act, 0.75))
    iqr = [q3 - q1 for q1, q3 in zip(q1s, q3s)]

    lower = [max(q1-1.5*iqr, min_v)
             for q1, iqr, min_v in zip(q1s, iqr, mins)]
    upper = [min(q3+1.5*iqr, max_v)
             for q3, iqr, max_v in zip(q3s, iqr, maxs)]
    return {
        "min": mins,
        "max": maxs,
        "mean": means,
        "q1": q1s,
        "q3": q3s,
        "IQR": iqr,
        "lower": lower,
        "upper": upper
    }


def export_layer_activation_statistics(input_data, model, model_name, layers,
                                       destination=CACHE_LOCATION):
    for layer in layers:
        act_path = activations_path(destination, model_name, layer)
        if not act_path.exists():
            export_activations(input_data, model, model_name, [
                               layer], destination)
        with h5py.File(act_path, 'r') as f_act:
            activations = f_act["activations"]
            statistics = activation_statistics(
                activations, axis=-1)
            path = activation_statistics_path(destination, model_name, layer)
            with open(path, "wb") as output_file:
                pickle.dump(statistics, output_file)


def export_layer_patterns_activation_statistics(input_data, model, model_name, layer, patterns, min_pattern_size,
                                                destination=CACHE_LOCATION):
    act_path = activations_path(destination, model_name, layer)
    if not act_path.exists():
        export_activations(input_data, model, model_name, [layer], destination)
    with h5py.File(act_path, 'r') as f_act:
        activations = f_act["activations"]
        pattern_statistics = {}
        for index, pattern in patterns.groupby("patternId"):
            pattern_statistics[index] = activation_statistics(
                activations[pattern.index.tolist()], axis=-1)

        path = layer_patterns_activation_statistics_path(
            destination, model_name, layer, min_pattern_size)
        with open(path, "wb") as output_file:
            pickle.dump(pattern_statistics, output_file)


def export_filter_patterns_activation_statistics(input_data, model, model_name, layer, filter_index,
                                                 filter_aggregation, patterns, min_pattern_size,
                                                 destination=CACHE_LOCATION):
    act_path = activations_path(destination, model_name, layer)
    if not act_path.exists():
        export_activations(input_data, model, model_name, [layer], destination)
    with h5py.File(act_path, 'r') as f_act:
        pattern_statistics = {}
        for index, pattern in patterns.groupby("patternId"):
            pattern_statistics[index] = activation_statistics(
                f_act["activations"][pattern.index.tolist()][..., filter_index], axis=None)

        path = filter_patterns_activation_statistics_path(
            destination, model_name, layer, filter_index, filter_aggregation, min_pattern_size)
        with open(path, "wb") as output_file:
            pickle.dump(pattern_statistics, output_file)


def export_filter_activation_statistics(input_data, model, model_name, layers, filters=None,
                                        destination=CACHE_LOCATION):
    for layer in layers:
        activations, file_handle = get_layer_activations(
            input_data, model, model_name, layer, destination)
        # [()] retireves all data because slicing the filter is super-slow in hdf5
        activations = activations[()]
        if filters is None:
            # pylint: disable=E1101
            filters = range(activations.shape[-1])
        for filter_index in filters:
            statistics = activation_statistics(
                activations[..., filter_index], axis=None)
            path = filter_activation_statistics_path(
                destination, model_name, layer, filter_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as output_file:
                pickle.dump(statistics, output_file)
        file_handle.close()


def export_layer_patterns(input_data, model, model_name, layers, layer_aggregation, min_pattern_size,
                          destination=CACHE_LOCATION):
    activation_patterns = nap.NeuralActivationPattern(
        model, layer_aggregation=layer_aggregation, min_pattern_size=min_pattern_size)
    for layer in layers:
        activations, file_handle = get_layer_activations_agg(
            input_data, model, model_name, layer, layer_aggregation, destination)
        patterns, patterns_info = activation_patterns.activity_patterns(
            layer, activations=activations)
        patterns_path = layer_patterns_path(
            destination, model_name, layer, min_pattern_size)
        file_handle.close()
        patterns.to_hdf(patterns_path, f'{layer}')
        patterns_info_path = layer_patterns_info_path(
            destination, model_name, layer, min_pattern_size)
        patterns_info.to_hdf(patterns_info_path, f'{layer}')


def export_filter_patterns(input_data, model, model_name, layers, filters, filter_aggregation, min_pattern_size,
                           destination=CACHE_LOCATION):
    activation_patterns = nap.NeuralActivationPattern(
        model, filter_aggregation=filter_aggregation, min_pattern_size=min_pattern_size)
    for layer in layers:
        activations, file_handle = get_layer_activations(
            input_data, model, model_name, layer, destination)
        # [()] fetches all data into memory. Needed because slicing the filter is super-slow in hdf5
        activations = activations[()]
        if filters is None:
            # pylint: disable=E1101
            filters = range(activations.shape[-1])
        for filter_index in filters:
            patterns, patterns_info = activation_patterns.activity_patterns(
                f'{layer}:{filter_index}', activations=activations)
            path = filter_patterns_path(
                destination, model_name, layer, filter_index, filter_aggregation, min_pattern_size)
            path.parent.mkdir(parents=True, exist_ok=True)
            patterns.to_hdf(path, f'{layer}/filter_{filter_index}')
            patterns_info.to_hdf(filter_patterns_info_path(
                destination, model_name, layer, filter_index, filter_aggregation, min_pattern_size),
                f'{layer}/filter_{filter_index}')
        file_handle.close()


def get_layer_activations(input_data, model, model_name, layer, destination=CACHE_LOCATION):
    path = activations_path(destination, model_name, layer)
    if not path.exists():
        export_activations(input_data, model, model_name, [layer], destination)
    file_handle = h5py.File(path, 'r')
    return file_handle["activations"], file_handle


def get_layer_activation_statistics(input_data, model, model_name, layer,
                                    destination=CACHE_LOCATION):
    path = activation_statistics_path(destination, model_name, layer)
    if not path.exists():
        export_layer_activation_statistics(
            input_data, model, model_name, [layer], destination)
    return pickle.load(open(path, "rb"))


def get_layer_activations_agg(input_data, model, model_name, layer, layer_aggregation,
                              destination=CACHE_LOCATION):
    path = activations_agg_path(
        destination, model_name, layer, layer_aggregation)
    if not path.exists():
        export_layer_aggregation(input_data, model, model_name, [
                                 layer], layer_aggregation, destination)
    file_handle = h5py.File(path, 'r')
    return file_handle["activations"], file_handle


def get_layer_patterns(input_data, model, model_name, layer, layer_aggregation, min_pattern_size,
                       destination=CACHE_LOCATION):
    path = layer_patterns_path(
        destination, model_name, layer, min_pattern_size)
    info_path = layer_patterns_info_path(
        destination, model_name, layer, min_pattern_size)
    if not path.exists() or not info_path.exists():
        export_layer_patterns(input_data, model, model_name, [
                              layer], layer_aggregation, min_pattern_size, destination)
    return pd.read_hdf(path), pd.read_hdf(info_path)


def get_layer_patterns_activation_statistics(input_data, model, model_name, layer,
                                             layer_aggregation, min_pattern_size, destination=CACHE_LOCATION):
    path = layer_patterns_activation_statistics_path(
        destination, model_name, layer, min_pattern_size)
    if not path.exists():
        patterns, _ = get_layer_patterns(
            input_data, model, model_name, layer, layer_aggregation, min_pattern_size, destination)
        export_layer_patterns_activation_statistics(
            input_data, model, model_name, layer, patterns, min_pattern_size, destination)
    with open(path, "rb") as output_file:
        return pickle.load(output_file)


def get_filter_patterns(input_data, model, model_name, layer, filter_index, filter_aggregation, min_pattern_size,
                        destination=CACHE_LOCATION):
    path = filter_patterns_path(
        destination, model_name, layer, filter_index, filter_aggregation, min_pattern_size)
    info_path = filter_patterns_info_path(
        destination, model_name, layer, filter_index, filter_aggregation, min_pattern_size)
    if not path.exists() or not info_path.exists():
        export_filter_patterns(input_data, model, model_name, [
                               layer], [filter_index], filter_aggregation, min_pattern_size, destination)
    return pd.read_hdf(path), pd.read_hdf(info_path)


def get_filter_patterns_activation_statistics(input_data, model, model_name, layer, filter_index,
                                              filter_aggregation, min_pattern_size, destination=CACHE_LOCATION):
    path = filter_patterns_activation_statistics_path(
        destination, model_name, layer, filter_index, filter_aggregation, min_pattern_size)
    if not path.exists():
        patterns, _ = get_filter_patterns(
            input_data, model, model_name, layer, filter_index, filter_aggregation, min_pattern_size, destination)
        export_filter_patterns_activation_statistics(
            input_data, model, model_name, layer, filter_index, filter_aggregation, patterns, min_pattern_size,
            destination)
    return pickle.load(open(path, "rb"))


def get_filter_activation_statistics(input_data, model, model_name, layer, filter_index,
                                     destination=CACHE_LOCATION):
    path = filter_activation_statistics_path(
        destination, model_name, layer, filter_index)
    if not path.exists():
        export_filter_activation_statistics(
            input_data, model, model_name, [layer], [filter_index], destination)
    return pickle.load(open(path, "rb"))
