import nap
import export
import models

import tensorflow as tf
from pathlib import Path
import tensorflow_datasets as tfds
import argparse
print(tf.__version__)


def dir_path(string):
    if Path(string).is_dir():
        return string
    raise NotADirectoryError(string)


parser = argparse.ArgumentParser(description="Running a model analysis run.")
parser.add_argument("--data_path", type=dir_path,
                    default='D:/data/tensorflow_datasets')
parser.add_argument("--model",
                    default='mnist', choices=["mnist", "cifar10", "inception_v1", "inception_v3"])
parser.add_argument("--data_set",
                    default='mnist')
parser.add_argument("--split",
                    default='test')
parser.add_argument("--layer")
parser.add_argument("--filter", type=int)
parser.add_argument("--aggregation",
                    default='mean', choices=["mean", "mean_std", "max", "none"])
parser.add_argument("--size", type=int,
                    default=2000)
args = parser.parse_args()


def create_aggregation_function(name):
    if name == "mean":
        return nap.MeanAggregation()
    elif name == "mean_std":
        return nap.MeanStdAggregation()
    elif name == "max":
        return nap.MeanAggregation()
    elif name == "none":
        return nap.NoAggregation()
    raise Exception(f"Invalid aggregation function: {name}")


def setup_model(model_name, data_path, data_set, data_set_size, split):
    X, y, file_names, image_dir = models.get_data_set(data_path, data_set, data_set_size, split)
    if model_name == "mnist":
        model, X = models.setup_mnist(X)
    elif model_name == "cifar10":
        model, X = models.setup_cifar10(X)
    elif model_name == "inception_v1":
        model, X = models.setup_inception_v1(X)
    elif model_name == "inception_v3":
        model, X = models.setup_inception_v3(X)
    else: 
        raise Exception(f"Invalid model: {model}")
    print(model.summary())
    model_name = f"{model_name}_{data_set}_{split}_{X.cardinality().numpy()}"
    return model, model_name, X, y, file_names, image_dir
    
# Model / data parameters


data_set_size = args.size
model, model_name, X, y, file_names, image_dir = setup_model(args.model, args.data_path, args.data_set, data_set_size, args.split)
model_name = f"{model_name}_leaf"
if args.layer is None:   
    layers = [layer.name for layer in model.layers] 
else:
    layers = [args.layer]
if args.filter is None:   
    filters = {}
else:
    filters  = {f"{layer}": [args.filter] for layer in layers}


# layers = ['Mixed_4b_Concatenated', 'Mixed_5b_Concatenated']
# layer = 'Mixed_4b_Concatenated'

agg_func = create_aggregation_function(args.aggregation)
if agg_func is None:
    model_name = f"{model_name}_no_agg"
else:
    model_name = f"{model_name}_{agg_func.__class__.__name__}"

# nap.cache.export_activations(X, model, model_name, layers=layers)
# nap.cache.export_layer_aggregation(X, model, model_name, layers=layers, agg_func=None)
# nap.cache.export_layer_patterns(X, model, model_name, layers=layers)
# nap.cache.export_filter_patterns(X, model, model_name, [layer], [filterId])
# X = X.take(10)
# y = y.take(10)
# X = X.batch(1)
y = list(tfds.as_numpy(y))
# ap = nap.NeuralActivationPattern(model)
# ap.layer_summary("conv2d", X, y)


#layer_analysis(model, model_name, X, y, layer)
# filter_analysis(model, model_name, X, y, layer, filterId)
files = file_names 
predictions = tf.argmax(model.predict(
    X.batch(128).cache().prefetch(tf.data.AUTOTUNE)), axis=1).numpy()
# for x in X.batch(128).cache().prefetch(tf.data.AUTOTUNE):
#     predictions = tf.keras.applications.inception_v3.decode_predictions(model.predict(
#         x), top=1)
#     print(predictions)
export.export_all(model, model_name, X, y, predictions,
                  files, layers, filters, str(image_dir), agg_func=agg_func)
