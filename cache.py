import h5py
import pandas as pd

def layer_activations(X, model, model_name, layer, destination=PRECOMPUTE_LOCATION):
    path = activations_path(destination, model_name, layer)
    if not path.exists():
        export_activations(X, model, model_name, [layer], destination)
    f = h5py.File(path, 'r')
    return f["activations"], f
def layer_activations_agg(X, model, model_name, layer, destination=PRECOMPUTE_LOCATION):
    path = activations_agg_path(destination, model_name, layer)
    if not path.exists():
        export_activations(X, model, model_name, [layer], destination)
    f = h5py.File(path, 'r')
    return f["activations"], f
def layer_patterns(X, model, model_name, layer, destination=PRECOMPUTE_LOCATION):
    path = layer_patterns_path(destination, model_name, layer)
    if not path.exists():
        export_layer_patterns(X, model, model_name, [layer], destination)
    return pd.read_hdf(path)
def filter_patterns(X, model, model_name, layer, filter, destination=PRECOMPUTE_LOCATION):
    path = filter_patterns_path(destination, model_name, layer, filter)
    if not path.exists():
        export_filter_patterns(X, model, model_name, [layer], [filter], destination)
    return pd.read_hdf(path)    