import h5py
import pandas as pd
import numpy as np
from pathlib import Path

CACHE_LOCATION = Path("results")

def activations_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_activations.h5')
def activations_agg_path(destination, model_name, layer, agg_func):
    return Path(destination, model_name, layer, f'layer_activations_{agg_func.__name__}.h5')    
def layer_patterns_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_patterns.h5')   
def filter_patterns_path(destination, model_name, layer, filter):
    return Path(destination, model_name, layer, 'filters', f'{filter}.h5') 

def export_activations(X, model, model_name, layers, destination=CACHE_LOCATION, mode = 'w'):
    batch_size = 1000

    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        act_path = activations_path(destination, model_name, layer)
        act_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(act_path, mode) as f:
            # Dummy computation to get the output shape
            activations = ap.layer_activations(layer, X.take(1).batch(1))
            output_shape = list(activations.shape)
            # Ignore first entry, which is batch size 
            agg_shape = nap.layer_activation_aggregation_shape(output_shape[1:])
            total_num_inputs = X.cardinality().numpy()
            
            output_shape[0] = total_num_inputs
            dset = f.create_dataset("activations", output_shape, compression="gzip")
            i = 0
            for ds in X.batch(batch_size):
                ap = nap.NeuralActivationPattern(model)
                num_inputs = ds.shape[0]
                activations = ap.layer_activations(layer, ds)
                dset[i:i+num_inputs] = activations
                i += num_inputs

def export_layer_aggregation(X, model, model_name, layers, agg_func = np.mean, destination=CACHE_LOCATION):
    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        act_path = activations_path(destination, model_name, layer)
        if not act_path.exists():
            export_activations(X, model, model_name, [layer], destination)
        with h5py.File(act_path, 'r') as f_act:
            activations = f_act["activations"]
            output_shape = list(activations.shape)
            agg_shape = nap.layer_activation_aggregation_shape(output_shape[1:], agg_func)
            total_num_inputs = output_shape[0]
            if isinstance(agg_shape, list):
                agg_shape = [total_num_inputs] + agg_shape
            else:
                agg_shape = [total_num_inputs] + [agg_shape]
            agg_path = activations_agg_path(destination, model_name, layer, agg_func)
            with h5py.File(agg_path, 'w') as f_agg:
                dset_aggregated = f_agg.create_dataset("activations", agg_shape, compression="gzip")
                for i, activation in enumerate(activations):
                    dset_aggregated[i] = nap.layer_activation_aggregation(activation[()], agg_func)


def export_layer_patterns(X, model, model_name, layers, agg_func = np.mean, destination=CACHE_LOCATION):
    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        activations, f = get_layer_activations_agg(X, model, model_name, layer, agg_func, destination)
        patterns = ap.activity_patterns(layer, activations=activations)
        patterns_path = layer_patterns_path(destination, model_name, layer)
        f.close()
        patterns.to_hdf(patterns_path, f'{layer}') 

def export_filter_patterns(X, model, model_name, layers, filters = None, destination=CACHE_LOCATION):
    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        
        activations, f = get_layer_activations(X, model, model_name, layer, destination)
        if filters is None:
            filters = range(activations.shape[-1])
        for filter in filters:
            # [()] retireves all data because slicing the filter is super-slow in hdf5  
            patterns = ap.activity_patterns(f'{layer}:{filter}', activations=activations[()])  
            path = filter_patterns_path(destination, model_name, layer, filter)
            path.parent.mkdir(parents=True, exist_ok=True)
            patterns.to_hdf(path, f'{layer}/filter_{filter}')     
        f.close()
 
def get_layer_activations(X, model, model_name, layer, destination=CACHE_LOCATION):
    path = activations_path(destination, model_name, layer)
    if not path.exists():
        export_activations(X, model, model_name, [layer], destination)
    f = h5py.File(path, 'r')
    return f["activations"], f
def get_layer_activations_agg(X, model, model_name, layer, agg_func = np.mean, destination=CACHE_LOCATION):
    path = activations_agg_path(destination, model_name, layer, agg_func)
    if not path.exists():
        export_layer_aggregation(X, model, model_name, [layer], agg_func, destination)
    f = h5py.File(path, 'r')
    return f["activations"], f
def get_layer_patterns(X, model, model_name, layer, agg_func = np.mean, destination=CACHE_LOCATION):
    path = layer_patterns_path(destination, model_name, layer)
    if not path.exists():
        export_layer_patterns(X, model, model_name, [layer], agg_func, destination)
    return pd.read_hdf(path)
def get_filter_patterns(X, model, model_name, layer, filter, destination=CACHE_LOCATION):
    path = filter_patterns_path(destination, model_name, layer, filter)
    if not path.exists():
        export_filter_patterns(X, model, model_name, [layer], [filter], destination)
    return pd.read_hdf(path)    