"""Provide caching for neural activation patterns."""
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

CACHE_LOCATION = Path("results")


def layer_settings_string(neural_activation):
    return (f'layer_agg_{neural_activation.layer_aggregation.__class__.__name__}'
            f'_min_size_{neural_activation.min_pattern_size}'
            f'_min_samples_{neural_activation.min_samples}'
            f'_cluster_selection_epsilon_{neural_activation.cluster_selection_epsilon:1.0e}'
            f'_{neural_activation.cluster_selection_method}')


def filter_settings_string(neural_activation):
    return (f'filter_agg_{neural_activation.filter_aggregation.__class__.__name__}'
            f'_min_size_{neural_activation.min_pattern_size}'
            f'_min_samples_{neural_activation.min_samples}'
            f'_cluster_selection_epsilon_{neural_activation.cluster_selection_epsilon:1.0e}'
            f'_{neural_activation.cluster_selection_method}')


def activations_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_activations.h5')


def activations_agg_path(destination, model_name, layer, neural_activation):
    return Path(
        destination, model_name, layer,
        f'layer_activations_{neural_activation.layer_aggregation.__class__.__name__}.h5')


def activation_statistics_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_activation_statistics.pkl')


def layer_patterns_activation_statistics_path(
        destination, model_name, layer, neural_activation):
    return Path(
        destination, model_name, layer,
        f'layer_patterns_{layer_settings_string(neural_activation)}_activation_statistics.pkl')


def filter_patterns_activation_statistics_path(destination, model_name, layer, filter_index,
                                               neural_activation):
    return Path(
        destination, model_name, layer, 'filters',
        filter_settings_string(neural_activation),
        str(filter_index),
        'filter_patterns_activation_statistics.pkl')


def filter_activation_statistics_path(destination, model_name, layer, filter_index):
    return Path(destination, model_name, layer, 'filters', str(filter_index),
                'filter_activation_statistics.pkl')


def layer_patterns_path(destination, model_name, layer, neural_activation):
    return Path(
        destination, model_name, layer,
        f'layer_patterns_{layer_settings_string(neural_activation)}.h5')


def layer_patterns_info_path(destination, model_name, layer, neural_activation):
    return Path(destination, model_name, layer,
                f'layer_patterns_{layer_settings_string(neural_activation)}_info.h5')


def filter_patterns_path(
        destination, model_name, layer, filter_index, neural_activation):
    return Path(destination, model_name, layer, 'filters',
                filter_settings_string(neural_activation), str(filter_index),
                'filter_patterns.h5')


def filter_patterns_info_path(
        destination, model_name, layer, filter_index, neural_activation):
    return Path(destination, model_name, layer, 'filters',
                filter_settings_string(neural_activation),
                str(filter_index), 'filter_patterns_info.h5')


# pylint: disable=R0914
def export_activations(input_data, neural_activation, model_name, layers,
                       destination=CACHE_LOCATION, mode='w'):
    for layer in layers:
        act_path = activations_path(destination, model_name, layer)
        act_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(act_path, mode) as file_handle:
            activations = neural_activation.layer_activations(
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
            num_units = output_shape[-1]
            maximum = [float('-inf')]*num_units
            for data_set in input_data.batch(128).cache().prefetch(tf.data.AUTOTUNE):
                num_inputs = data_set.shape[0]
                activations = neural_activation.layer_activations(
                    layer, data_set)
                unit_max = [np.max(np.abs(activations[..., unit])) for unit in range(num_units)]
                maximum = np.maximum(unit_max, maximum)
                dset[iterator:iterator+num_inputs] = activations
                iterator += num_inputs

            # Normalize units individually by their absolute max activation
            for chunk in dset.iter_chunks():
                # [()] fetches all data into memory.
                data = dset[chunk][()]
                for unit in range(num_units):
                    if maximum[unit] != 0:
                        data[..., unit] /= maximum[unit]
                dset[chunk] = data


# pylint: disable=R0914


def export_layer_aggregation(input_data, neural_activation, model_name, layers,
                             destination=CACHE_LOCATION):
    for layer in layers:
        act_path = activations_path(destination, model_name, layer)
        if not act_path.exists():
            export_activations(input_data, neural_activation, model_name, [
                               layer], destination)
        with h5py.File(act_path, 'r') as f_act:
            activations = f_act["activations"]
            # pylint: disable=E1101
            output_shape = list(activations.shape)
            agg_shape = neural_activation.layer_aggregation.shape(output_shape[1:])
            total_num_inputs = output_shape[0]
            agg_path = activations_agg_path(
                destination, model_name, layer, neural_activation)
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
                        neural_activation.layer_aggregation.aggregate(
                            neural_activation.layer(layer), activation)
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
        quantiles = np.quantile(activations, [0.25, 0.75])
        q1s.append(quantiles[0])
        q3s.append(quantiles[1])
    else:
        # Preallocate arrays
        means = [0]*activations.shape[axis]
        mins = [float('inf')]*activations.shape[axis]
        maxs = [float('-inf')]*activations.shape[axis]
        q1s = [0]*activations.shape[axis]
        q3s = [0]*activations.shape[axis]
        if isinstance(activations, h5py.Dataset):
            # Improve performance by operating on a per chunk-basis
            # h5py is super-slow when accessing the last (feature) dimension
            n_chunks = 0
            for chunk in activations.iter_chunks():
                n_chunks += 1
                # [()] fetches all data into memory.
                data = activations[chunk][()]
                for feature in range(data.shape[axis]):
                    f_act = data[..., feature]
                    means[feature] += np.mean(f_act)
                    mins[feature] = min(mins[feature], np.min(f_act))
                    maxs[feature] = max(maxs[feature], np.max(f_act))
                    quantiles = np.quantile(f_act, [0.25, 0.75])
                    # Quantiles will not be exact when computed chunked.
                    # Small bias, but much faster computation compared to computing it with all data
                    # cinot
                    # https://stackoverflow.com/questions/40291135/iterative-quantile-estimation-in-matlab
                    q1s[feature] += quantiles[0]
                    q3s[feature] += quantiles[1]

            for feature in range(activations.shape[axis]):
                means[feature] /= n_chunks
                q1s[feature] /= n_chunks
                q3s[feature] /= n_chunks

        else:
            for feature in range(activations.shape[axis]):
                f_act = activations[..., feature]
                means[feature] = np.mean(f_act)
                mins[feature] = np.min(f_act)
                maxs[feature] = np.max(f_act)
                quantiles = np.quantile(f_act, [0.25, 0.75])
                q1s[feature] = quantiles[0]
                q3s[feature] = quantiles[1]

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


def export_layer_activation_statistics(input_data, neural_activation, model_name, layers,
                                       destination=CACHE_LOCATION):
    for layer in layers:
        act_path = activations_path(destination, model_name, layer)
        if not act_path.exists():
            export_activations(input_data, neural_activation, model_name, [
                               layer], destination)
        with h5py.File(act_path, 'r') as f_act:
            activations = f_act["activations"]
            # [()] fetches all data into memory.
            # Needed because slicing the filter is super-slow in hdf5
            statistics = activation_statistics(
                activations, axis=-1)
            path = activation_statistics_path(destination, model_name, layer)
            with open(path, "wb") as output_file:
                pickle.dump(statistics, output_file)


def export_layer_patterns_activation_statistics(
        input_data, neural_activation, model_name, layer, patterns,
        destination=CACHE_LOCATION):
    act_path = activations_path(destination, model_name, layer)
    if not act_path.exists():
        export_activations(input_data, neural_activation, model_name, [layer], destination)
    with h5py.File(act_path, 'r') as f_act:
        activations = f_act["activations"]
        pattern_statistics = {}
        for index, pattern in patterns.groupby("patternId"):
            pattern_statistics[index] = activation_statistics(
                activations[pattern.index.tolist()], axis=-1)

        path = layer_patterns_activation_statistics_path(
            destination, model_name, layer, neural_activation)
        with open(path, "wb") as output_file:
            pickle.dump(pattern_statistics, output_file)


def export_filter_patterns_activation_statistics(
        input_data, neural_activation, model_name, layer, filter_index, patterns,
        destination=CACHE_LOCATION):
    act_path = activations_path(destination, model_name, layer)
    if not act_path.exists():
        export_activations(input_data, neural_activation, model_name, [layer], destination)
    with h5py.File(act_path, 'r') as f_act:
        pattern_statistics = {}
        for index, pattern in patterns.groupby("patternId"):
            pattern_statistics[index] = activation_statistics(
                f_act["activations"][pattern.index.tolist()][..., filter_index], axis=None)

        path = filter_patterns_activation_statistics_path(
            destination, model_name, layer, filter_index, neural_activation)
        with open(path, "wb") as output_file:
            pickle.dump(pattern_statistics, output_file)


def export_filter_activation_statistics(input_data, neural_activation, model_name, layers,
                                        filters=None, destination=CACHE_LOCATION):
    for layer in layers:
        activations, file_handle = get_layer_activations(
            input_data, neural_activation, model_name, layer, destination)
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


def export_layer_patterns(
        input_data, neural_activation, model_name, layers,
        destination=CACHE_LOCATION):
    for layer in layers:
        activations, file_handle = get_layer_activations_agg(
            input_data, neural_activation, model_name, layer, destination)
        patterns, patterns_info = neural_activation.activity_patterns(
            layer, activations=activations)
        patterns_path = layer_patterns_path(
            destination, model_name, layer, neural_activation)
        file_handle.close()
        patterns.to_hdf(patterns_path, f'{layer}')
        patterns_info_path = layer_patterns_info_path(
            destination, model_name, layer, neural_activation)
        patterns_info.to_hdf(patterns_info_path, f'{layer}')


def export_filter_patterns(
        input_data, neural_activation, model_name, layers, filters,
        destination=CACHE_LOCATION):
    for layer in layers:
        activations, file_handle = get_layer_activations(
            input_data, neural_activation, model_name, layer, destination)
        # [()] fetches all data into memory. Needed because slicing the filter is super-slow in hdf5
        activations = activations[()]
        if filters is None:
            # pylint: disable=E1101
            filters = range(activations.shape[-1])
        for filter_index in filters:
            patterns, patterns_info = neural_activation.activity_patterns(
                f'{layer}:{filter_index}', activations=activations)
            path = filter_patterns_path(
                destination, model_name, layer, filter_index, neural_activation)
            path.parent.mkdir(parents=True, exist_ok=True)
            patterns.to_hdf(path, f'{layer}/filter_{filter_index}')
            patterns_info.to_hdf(filter_patterns_info_path(
                destination, model_name, layer, filter_index, neural_activation),
                f'{layer}/filter_{filter_index}')
        file_handle.close()


def get_layer_activations(input_data, neural_activation, model_name, layer,
                          destination=CACHE_LOCATION):
    path = activations_path(destination, model_name, layer)
    if not path.exists():
        export_activations(input_data, neural_activation, model_name, [layer], destination)
    file_handle = h5py.File(path, 'r')
    return file_handle["activations"], file_handle


def get_layer_activation_statistics(input_data, neural_activation, model_name, layer,
                                    destination=CACHE_LOCATION):
    path = activation_statistics_path(destination, model_name, layer)
    if not path.exists():
        export_layer_activation_statistics(
            input_data, neural_activation, model_name, [layer], destination)
    return pickle.load(open(path, "rb"))


def get_layer_activations_agg(input_data, neural_activation, model_name, layer,
                              destination=CACHE_LOCATION):
    path = activations_agg_path(
        destination, model_name, layer, neural_activation)
    if not path.exists():
        export_layer_aggregation(input_data, neural_activation, model_name, [
                                 layer], destination)
    file_handle = h5py.File(path, 'r')
    return file_handle["activations"], file_handle


def get_layer_patterns(input_data, neural_activation, model_name, layer,
                       destination=CACHE_LOCATION):
    path = layer_patterns_path(
        destination, model_name, layer, neural_activation)
    info_path = layer_patterns_info_path(
        destination, model_name, layer, neural_activation)
    if not path.exists() or not info_path.exists():
        export_layer_patterns(input_data, neural_activation, model_name, [
                              layer], destination)
    return pd.read_hdf(path), pd.read_hdf(info_path)


def get_layer_patterns_activation_statistics(
        input_data, neural_activation, model_name, layer,
        destination=CACHE_LOCATION):
    path = layer_patterns_activation_statistics_path(
        destination, model_name, layer, neural_activation)
    if not path.exists():
        patterns, _ = get_layer_patterns(
            input_data, neural_activation, model_name, layer, destination)
        export_layer_patterns_activation_statistics(
            input_data, neural_activation, model_name, layer, patterns, destination)
    with open(path, "rb") as output_file:
        return pickle.load(output_file)


def get_filter_patterns(
        input_data, neural_activation, model_name, layer, filter_index,
        destination=CACHE_LOCATION):
    path = filter_patterns_path(
        destination, model_name, layer, filter_index, neural_activation)
    info_path = filter_patterns_info_path(
        destination, model_name, layer, filter_index, neural_activation)
    if not path.exists() or not info_path.exists():
        export_filter_patterns(
            input_data, neural_activation, model_name, [layer],
            [filter_index], destination)
    return pd.read_hdf(path), pd.read_hdf(info_path)


def get_filter_patterns_activation_statistics(
        input_data, neural_activation, model_name, layer, filter_index,
        destination=CACHE_LOCATION):
    path = filter_patterns_activation_statistics_path(
        destination, model_name, layer, filter_index, neural_activation)
    if not path.exists():
        patterns, _ = get_filter_patterns(
            input_data, neural_activation, model_name, layer, filter_index,
            destination)
        export_filter_patterns_activation_statistics(
            input_data, neural_activation, model_name, layer, filter_index, patterns,
            destination)
    return pickle.load(open(path, "rb"))


def get_filter_activation_statistics(input_data, neural_activation, model_name, layer, filter_index,
                                     destination=CACHE_LOCATION):
    path = filter_activation_statistics_path(
        destination, model_name, layer, filter_index)
    if not path.exists():
        export_filter_activation_statistics(
            input_data, neural_activation, model_name, [layer], [filter_index], destination)
    return pickle.load(open(path, "rb"))
