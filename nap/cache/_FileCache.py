import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
import nap
from pathlib import Path
import pickle
from scipy import stats

CACHE_LOCATION = Path("results")


def activations_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_activations.h5')


def activations_agg_path(destination, model_name, layer, layer_aggregation):
    return Path(destination, model_name, layer, f'layer_activations_{layer_aggregation.__class__.__name__}.h5')


def activation_statistics_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_activation_statistics.pkl')


def layer_patterns_activation_statistics_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_patterns_activation_statistics.pkl')


def filter_patterns_activation_statistics_path(destination, model_name, layer, filter, filter_aggregation):
    return Path(destination, model_name, layer, 'filters', filter_aggregation.__class__.__name__, str(filter), 'filter_patterns_activation_statistics.pkl')


def filter_activation_statistics_path(destination, model_name, layer, filter):
    return Path(destination, model_name, layer, 'filters', str(filter), 'filter_activation_statistics.pkl')


def layer_patterns_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_patterns.h5')


def layer_patterns_info_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_patterns_info.h5')


def filter_patterns_path(destination, model_name, layer, filter, filter_aggregation):
    return Path(destination, model_name, layer, 'filters', filter_aggregation.__class__.__name__, str(filter), 'filter_patterns.h5')


def filter_patterns_info_path(destination, model_name, layer, filter, filter_aggregation):
    return Path(destination, model_name, layer, 'filters', filter_aggregation.__class__.__name__, str(filter), 'filter_patterns_info.h5')


def export_activations(X, model, model_name, layers, destination=CACHE_LOCATION, mode='w'):
    batch_size = 1000

    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        act_path = activations_path(destination, model_name, layer)
        act_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(act_path, mode) as f:
            # Dummy computation to get the output shape
            activations = ap.layer_activations(layer, X.take(1).batch(1))
            output_shape = list(activations.shape)
            total_num_inputs = X.cardinality().numpy()
            # Ignore first entry, which is batch size
            output_shape[0] = total_num_inputs
            bytes_per_float = 4
            num_activations_in_one_MB = min(total_num_inputs, max(
                1, int(1000000 / (np.prod(output_shape[1:])*bytes_per_float))))
            chunk_size = tuple([num_activations_in_one_MB] + output_shape[1:])
            dset = f.create_dataset(
                "activations", output_shape, compression="gzip", chunks=chunk_size)
            i = 0
            for ds in X.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE):
                ap = nap.NeuralActivationPattern(model)
                num_inputs = ds.shape[0]
                activations = ap.layer_activations(layer, ds)
                dset[i:i+num_inputs] = activations
                i += num_inputs


def export_layer_aggregation(X, model, model_name, layers, layer_aggregation, destination=CACHE_LOCATION):
    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        act_path = activations_path(destination, model_name, layer)
        if not act_path.exists():
            export_activations(X, model, model_name, [layer], destination)
        with h5py.File(act_path, 'r') as f_act:
            activations = f_act["activations"]
            output_shape = list(activations.shape)
            agg_shape = layer_aggregation.shape(output_shape[1:])

            total_num_inputs = output_shape[0]
            agg_path = activations_agg_path(
                destination, model_name, layer, layer_aggregation)
            with h5py.File(agg_path, 'w') as f_agg:
                agg_size = np.prod(agg_shape)
                bytes_per_float = 4
                num_activations_in_one_MB = min(total_num_inputs, max(
                    1, int(1000000 / (agg_size*bytes_per_float))))
                if isinstance(agg_shape, list):
                    all_agg_shape = [total_num_inputs] + agg_shape
                    chunk_size = (num_activations_in_one_MB) + tuple(agg_shape)
                else:
                    all_agg_shape = [total_num_inputs] + [agg_shape]
                    chunk_size = (num_activations_in_one_MB, agg_shape)

                dset_aggregated = f_agg.create_dataset(
                    "activations", all_agg_shape, compression="gzip", chunks=chunk_size)
                i = 0
                for chunk in activations.iter_chunks():
                    data = activations[chunk]
                    dset_aggregated[i:i+data.shape[0]
                                    ] = [layer_aggregation.aggregate(ap.layer(layer), activation) for activation in data]
                    i += data.shape[0]


def export_layer_activation_statistics(X, model, model_name, layers, destination=CACHE_LOCATION):
    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        act_path = activations_path(destination, model_name, layer)
        if not act_path.exists():
            export_activations(X, model, model_name, [layer], destination)
        with h5py.File(act_path, 'r') as f_act:
            activations = f_act["activations"]
            statistics = stats.describe(activations, axis=-1)
            path = activation_statistics_path(destination, model_name, layer)
            pickle.dump(statistics, open(path, "wb"))


def export_layer_patterns_activation_statistics(X, model, model_name, layer, patterns, destination=CACHE_LOCATION):
    act_path = activations_path(destination, model_name, layer)
    if not act_path.exists():
        export_activations(X, model, model_name, [layer], destination)
    with h5py.File(act_path, 'r') as f_act:
        activations = f_act["activations"]
        pattern_statistics = {}
        for id, pattern in patterns.groupby("patternId"):
            statistics = stats.describe(
                activations[pattern.index.tolist()], axis=-1)
            pattern_statistics[id] = statistics
        path = layer_patterns_activation_statistics_path(
            destination, model_name, layer)
        pickle.dump(pattern_statistics, open(path, "wb"))


def export_filter_patterns_activation_statistics(X, model, model_name, layer, filter, filter_aggregation, patterns, destination=CACHE_LOCATION):
    act_path = activations_path(destination, model_name, layer)
    if not act_path.exists():
        export_activations(X, model, model_name, [layer], destination)
    with h5py.File(act_path, 'r') as f_act:
        activations = f_act["activations"]
        pattern_statistics = {}
        for id, pattern in patterns.groupby("patternId"):
            statistics = stats.describe(
                activations[pattern.index.tolist()][..., filter], axis=None)
            pattern_statistics[id] = statistics
        path = filter_patterns_activation_statistics_path(
            destination, model_name, layer, filter, filter_aggregation)
        pickle.dump(pattern_statistics, open(path, "wb"))


def export_filter_activation_statistics(X, model, model_name, layers, filters=None, destination=CACHE_LOCATION):
    for layer in layers:
        activations, f = get_layer_activations(
            X, model, model_name, layer, destination)
        # [()] retireves all data because slicing the filter is super-slow in hdf5
        activations = activations[()]
        if filters is None:
            filters = range(activations.shape[-1])
        for filter in filters:
            statistics = stats.describe(activations[..., filter], axis=None)
            path = filter_activation_statistics_path(
                destination, model_name, layer, filter)
            path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(statistics, open(path, "wb"))
        f.close()


def export_layer_patterns(X, model, model_name, layers, layer_aggregation, destination=CACHE_LOCATION):
    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        activations, f = get_layer_activations_agg(
            X, model, model_name, layer, layer_aggregation, destination)
        patterns, patterns_info = ap.activity_patterns(
            layer, activations=activations)
        patterns_path = layer_patterns_path(destination, model_name, layer)
        f.close()
        patterns.to_hdf(patterns_path, f'{layer}')
        patterns_info_path = layer_patterns_info_path(
            destination, model_name, layer)
        patterns_info.to_hdf(patterns_info_path, f'{layer}')


def export_filter_patterns(X, model, model_name, layers, filters, filter_aggregation, destination=CACHE_LOCATION):
    ap = nap.NeuralActivationPattern(
        model, filter_aggregation=filter_aggregation)
    for layer in layers:
        activations, f = get_layer_activations(
            X, model, model_name, layer, destination)
        # [()] fetches all data into memory. Needed because slicing the filter is super-slow in hdf5
        activations = activations[()]
        if filters is None:
            filters = range(activations.shape[-1])
        for filter in filters:
            patterns, patterns_info = ap.activity_patterns(
                f'{layer}:{filter}', activations=activations)
            path = filter_patterns_path(
                destination, model_name, layer, filter, filter_aggregation)
            path.parent.mkdir(parents=True, exist_ok=True)
            patterns.to_hdf(path, f'{layer}/filter_{filter}')
            patterns_info_path = filter_patterns_info_path(
                destination, model_name, layer, filter, filter_aggregation)
            patterns_info.to_hdf(patterns_info_path,
                                 f'{layer}/filter_{filter}')
        f.close()


def get_layer_activations(X, model, model_name, layer, destination=CACHE_LOCATION):
    path = activations_path(destination, model_name, layer)
    if not path.exists():
        export_activations(X, model, model_name, [layer], destination)
    f = h5py.File(path, 'r')
    return f["activations"], f


def get_layer_activation_statistics(X, model, model_name, layer, destination=CACHE_LOCATION):
    path = activation_statistics_path(destination, model_name, layer)
    if not path.exists():
        export_layer_activation_statistics(
            X, model, model_name, [layer], destination)
    return pickle.load(open(path, "rb"))


def get_layer_activations_agg(X, model, model_name, layer, layer_aggregation, destination=CACHE_LOCATION):
    path = activations_agg_path(
        destination, model_name, layer, layer_aggregation)
    if not path.exists():
        export_layer_aggregation(X, model, model_name, [
                                 layer], layer_aggregation, destination)
    f = h5py.File(path, 'r')
    return f["activations"], f


def get_layer_patterns(X, model, model_name, layer, layer_aggregation, destination=CACHE_LOCATION):
    path = layer_patterns_path(destination, model_name, layer)
    info_path = layer_patterns_info_path(destination, model_name, layer)
    if not path.exists() or not info_path.exists():
        export_layer_patterns(X, model, model_name, [
                              layer], layer_aggregation, destination)
    return pd.read_hdf(path), pd.read_hdf(info_path)


def get_layer_patterns_activation_statistics(X, model, model_name, layer, layer_aggregation, destination=CACHE_LOCATION):
    path = layer_patterns_activation_statistics_path(
        destination, model_name, layer)
    if not path.exists():
        patterns, _ = get_layer_patterns(
            X, model, model_name, layer, layer_aggregation, destination)
        export_layer_patterns_activation_statistics(
            X, model, model_name, layer, patterns, destination)
    return pickle.load(open(path, "rb"))


def get_filter_patterns(X, model, model_name, layer, filter, filter_aggregation, destination=CACHE_LOCATION):
    path = filter_patterns_path(
        destination, model_name, layer, filter, filter_aggregation)
    info_path = filter_patterns_info_path(
        destination, model_name, layer, filter, filter_aggregation)
    if not path.exists() or not info_path.exists():
        export_filter_patterns(X, model, model_name, [
                               layer], [filter], filter_aggregation, destination)
    return pd.read_hdf(path), pd.read_hdf(info_path)


def get_filter_patterns_activation_statistics(X, model, model_name, layer, filter, filter_aggregation, destination=CACHE_LOCATION):
    path = filter_patterns_activation_statistics_path(
        destination, model_name, layer, filter, filter_aggregation)
    if not path.exists():
        patterns, _ = get_filter_patterns(
            X, model, model_name, layer, filter, filter_aggregation, destination)
        export_filter_patterns_activation_statistics(
            X, model, model_name, layer, filter, filter_aggregation, patterns, destination)
    return pickle.load(open(path, "rb"))


def get_filter_activation_statistics(X, model, model_name, layer, filter, destination=CACHE_LOCATION):
    path = filter_activation_statistics_path(
        destination, model_name, layer, filter)
    if not path.exists():
        export_filter_activation_statistics(
            X, model, model_name, [layer], [filter], destination)
    return pickle.load(open(path, "rb"))
