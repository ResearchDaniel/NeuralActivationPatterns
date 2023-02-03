"""Provide caching for neural activation patterns."""
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import tensorflow as tf

CACHE_LOCATION = Path("results")


def layer_settings_string(neural_activation):
    if neural_activation.unit_normalization:
        unit_normalization = "_unitnorm_"
    else:
        unit_normalization = ""
    return (f'{unit_normalization}'
            f'layer_agg_{neural_activation.layer_aggregation.__class__.__name__}'
            f'_min_size_{neural_activation.min_pattern_size}'
            f'_min_samples_{neural_activation.min_samples}'
            f'_cluster_selection_epsilon_{neural_activation.cluster_selection_epsilon:1.0e}'
            f'_{neural_activation.cluster_selection_method}'
            f'_{neural_activation.metric}')


def filter_settings_string(neural_activation):
    if neural_activation.unit_normalization:
        unit_normalization = "_unitnorm_"
    else:
        unit_normalization = ""
    return (f'{unit_normalization}'
            f'filter_agg_{neural_activation.filter_aggregation.__class__.__name__}'
            f'_min_size_{neural_activation.min_pattern_size}'
            f'_min_samples_{neural_activation.min_samples}'
            f'_cluster_selection_epsilon_{neural_activation.cluster_selection_epsilon:1.0e}'
            f'_{neural_activation.cluster_selection_method}'
            f'_{neural_activation.metric}')


def activations_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_activations.h5')


def activations_agg_path(destination, model_name, layer, neural_activation):
    if neural_activation.unit_normalization:
        unit_normalization = "_unitnorm_"
    else:
        unit_normalization = ""
    return Path(
        destination, model_name, layer,
        f'layer_activations{unit_normalization}_{neural_activation.layer_aggregation.__class__.__name__}.h5')


def activation_statistics_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_activation_statistics.pkl')


def layer_patterns_activation_statistics_path(
        destination, model_name, layer, pattern_index, activation):
    return Path(
        destination, model_name, layer,
        (f'layer_patterns_{layer_settings_string(activation)}'
         f'_activation_statistics_{pattern_index}.arrow'))


def filter_patterns_activation_statistics_path(
        destination, model_name, layer, filter_index, pattern_index, neural_activation):
    return Path(
        destination, model_name, layer, 'filters',
        filter_settings_string(neural_activation),
        str(filter_index),
        f'filter_patterns_activation_statistics_{pattern_index}.arrow')


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
            for data_set in input_data.batch(128).cache().prefetch(tf.data.AUTOTUNE):
                num_inputs = data_set.shape[0]
                activations = neural_activation.layer_activations(
                    layer, data_set)
                dset[iterator:iterator+num_inputs] = activations
                iterator += num_inputs


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
                    chunk_size = tuple([num_activations_in_one_mb] + agg_shape)
                else:
                    all_agg_shape = [total_num_inputs] + [agg_shape]
                    chunk_size = (num_activations_in_one_mb, agg_shape)

                dset_aggregated = f_agg.create_dataset(
                    "activations", all_agg_shape, compression="gzip", chunks=chunk_size)

                abs_max = np.full(agg_shape, float('-inf'))
                i = 0
                do_aggregation = neural_activation.layer_aggregation.should_aggregate(
                    output_shape
                    [1:])
                for chunk in activations.iter_chunks():
                    data = activations[chunk]
                    if do_aggregation:
                        aggregated = [
                            neural_activation.layer_aggregation.aggregate(
                                neural_activation.layer(layer),
                                activation) for activation in data]
                    else:
                        aggregated = data

                    abs_agg = np.abs(aggregated)
                    unit_max = np.max(abs_agg, axis=0)
                    abs_max = np.maximum(unit_max, abs_max)

                    dset_aggregated[i:i+data.shape[0]] = aggregated
                    i += data.shape[0]

                if neural_activation.unit_normalization:
                    # Normalize aggregated dimensions individually by their absolute max activation
                    if do_aggregation:
                        norm_val = neural_activation.layer_aggregation.normalization_value(abs_max)
                        for chunk in dset_aggregated.iter_chunks():
                            data = dset_aggregated[chunk]
                            dset_aggregated[chunk] = neural_activation.layer_aggregation.normalize(
                                data, norm_val)
                    else:
                        normalization_val = np.max(abs_max, axis=0)
                        for chunk in dset_aggregated.iter_chunks():
                            data = dset_aggregated[chunk]
                            dset_aggregated[chunk] = np.divide(
                                data, normalization_val, out=np.zeros_like(data),
                                where=~np.isclose(normalization_val, np.zeros_like(normalization_val)))


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
                    f_act = data[..., feature].ravel()
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
                f_act = activations[..., feature].ravel()
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
        act_path = activations_agg_path(destination, model_name, layer, neural_activation)
        if not act_path.exists():
            export_layer_aggregation(input_data, neural_activation, model_name, [
                layer], destination)
        with h5py.File(act_path, 'r') as f_act:
            statistics = activation_statistics(
                f_act["activations"], axis=-1)
            path = activation_statistics_path(destination, model_name, layer)
            with open(path, "wb") as output_file:
                pickle.dump(statistics, output_file)


def export_layer_patterns_activation_statistics(
        input_data, neural_activation, model_name, layer, patterns,
        destination=CACHE_LOCATION):
    act_path = activations_agg_path(destination, model_name, layer, neural_activation)
    if not act_path.exists():
        export_layer_aggregation(input_data, neural_activation, model_name, [
            layer], destination)
    with h5py.File(act_path, 'r') as f_act:
        activations = f_act["activations"]
        for index, pattern in patterns.groupby("patternId"):
            pattern_statistics = pa.Table.from_arrays(
                [list(range(neural_activation.layer_num_units(layer)))],
                names=["unit"])
            pattern_stats = activation_statistics(
                activations[pattern.index.tolist()], axis=-1)
            for key, value in pattern_stats.items():
                pattern_statistics = pattern_statistics.append_column(
                    key, pa.array(np.array(value).astype(np.float16)))
            path = layer_patterns_activation_statistics_path(
                destination, model_name, layer, index, neural_activation)
            writer = pa.ipc.new_file(path, pattern_statistics.schema)
            writer.write(pattern_statistics)
            writer.close()


def export_filter_patterns_activation_statistics(
        input_data, neural_activation, model_name, layer, filter_index, patterns,
        destination=CACHE_LOCATION):
    act_path = activations_path(destination, model_name, layer)
    if not act_path.exists():
        export_activations(input_data, neural_activation, model_name, [layer], destination)
    with h5py.File(act_path, 'r') as f_act:
        for index, pattern in patterns.groupby("patternId"):
            pattern_statistics = pa.Table.from_arrays([[filter_index]],
                                                      names=["unit"])
            pattern_stats = activation_statistics(
                f_act["activations"][pattern.index.tolist()]
                [..., filter_index],
                axis=None)
            for key, value in pattern_stats.items():
                pattern_statistics = pattern_statistics.append_column(
                    f"{index}_{key}", [np.array(value).astype(np.float16)])
            path = filter_patterns_activation_statistics_path(
                destination, model_name, layer, filter_index, index, neural_activation)
            writer = pa.ipc.new_file(path, pattern_statistics.schema)
            writer.write(pattern_statistics)
            writer.close()


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
        input_data, neural_activation, model_name, layer, pattern_index,
        destination=CACHE_LOCATION):
    path = layer_patterns_activation_statistics_path(
        destination, model_name, layer, pattern_index, neural_activation)
    if not path.exists():
        patterns, _ = get_layer_patterns(
            input_data, neural_activation, model_name, layer, destination)
        export_layer_patterns_activation_statistics(
            input_data, neural_activation, model_name, layer, patterns, destination)
    reader = pa.ipc.open_file(path)
    table = reader.read_all()
    return table


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
        input_data, neural_activation, model_name, layer, filter_index, pattern_index,
        destination=CACHE_LOCATION):
    path = filter_patterns_activation_statistics_path(
        destination, model_name, layer, filter_index, pattern_index, neural_activation)
    if not path.exists():
        patterns, _ = get_filter_patterns(
            input_data, neural_activation, model_name, layer, filter_index,
            destination)
        export_filter_patterns_activation_statistics(
            input_data, neural_activation, model_name, layer, filter_index, patterns,
            destination)
    reader = pa.ipc.open_file(path)
    table = reader.read_all()
    return table


def get_filter_activation_statistics(input_data, neural_activation, model_name, layer, filter_index,
                                     destination=CACHE_LOCATION):
    path = filter_activation_statistics_path(
        destination, model_name, layer, filter_index)
    if not path.exists():
        export_filter_activation_statistics(
            input_data, neural_activation, model_name, [layer], [filter_index], destination)
    return pickle.load(open(path, "rb"))
