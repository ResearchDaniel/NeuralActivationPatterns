"""Analyzing a model using NAPs."""
import argparse
import pickle
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

import export
import models
import nap
import util

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
parser.add_argument("--layer_aggregation",
                    default='mean', choices=["mean", "mean_std", "max", "none"])
parser.add_argument("--all_filters", default=False, action='store_true')
parser.add_argument("--filter_range", type=int, nargs=2, required=False)
parser.add_argument("--filter_aggregation",
                    default='none', choices=["mean", "mean_std", "max", "none"])

parser.add_argument("--size", type=int,
                    default=2000)
parser.add_argument("--n_max_activations", type=int,
                    default=0)
args = parser.parse_args()


def create_aggregation_function(name):
    if name == "mean":
        return nap.MeanAggregation()
    if name == "mean_std":
        return nap.MeanStdAggregation()
    if name == "max":
        return nap.MaxAggregation()
    if name == "none":
        return nap.NoAggregation()
    raise Exception(f"Invalid aggregation function: {name}")


def setup_model(model_type, data_path, data_set, data_set_size, split):
    processing_data, labels, file_names, image_dir = models.get_data_set(
        data_path, data_set, data_set_size, split)
    if model_type == "mnist":
        model, processing_data = models.setup_mnist(processing_data)
    elif model_type == "cifar10":
        model, processing_data = models.setup_cifar10(processing_data)
    elif model_type == "inception_v1":
        model, processing_data = models.setup_inception_v1(processing_data)
    elif model_type == "inception_v3":
        model, processing_data = models.setup_inception_v3(processing_data)
    else:
        raise Exception(f"Invalid model: {model}")
    print(model.summary())
    clean_data_set_name = data_set.replace("/", "-")
    model_string = (f"{model_type}_{clean_data_set_name}_{split}_"
                    f"{processing_data.cardinality().numpy()}")
    return model, model_string, processing_data, labels, file_names, image_dir

# Model / data parameters


data_size = args.size
ml_model, model_name, X, y, files, data_dir = setup_model(
    args.model, args.data_path, args.data_set, data_size, args.split)
model_name = f"{model_name}_leaf"
if args.layer is None:
    layers = [layer.name for layer in ml_model.layers]
else:
    layers = [args.layer]
filters = {}
if args.all_filters:
    for layer in layers:
        shape = list(ml_model.get_layer(layer).output.shape)
        filters[f"{layer}"] = list(range(0, shape[-1]))
elif args.filter_range is not None:
    for layer in layers:
        filters[f"{layer}"] = list(range(
            args.filter_range[0], args.filter_range[1]))

# layers = ['Mixed_4b_Concatenated', 'Mixed_5b_Concatenated']
# layer = 'Mixed_4b_Concatenated'

layer_aggregation = create_aggregation_function(args.layer_aggregation)
filter_aggregation = create_aggregation_function(args.filter_aggregation)
model_name = f"{model_name}_{layer_aggregation.__class__.__name__}"

# nap.cache.export_activations(X, ml_model, model_name, layers=layers)
# nap.cache.export_layer_aggregation(X, ml_model, model_name, layers=layers, layer_aggregation=None)
# nap.cache.export_layer_patterns(X, ml_model, model_name, layers=layers)
# nap.cache.export_filter_patterns(X, ml_model, model_name, [layer], [filterId])
# X = X.take(10)
# y = y.take(10)
# X = X.batch(1)
y = list(tfds.as_numpy(y))
# ap = nap.NeuralActivationPattern(ml_model)
# ap.layer_summary("conv2d", X, y)


# layer_analysis(ml_model, model_name, X, y, layer)
# filter_analysis(ml_model, model_name, X, y, layer, filterId)


if args.n_max_activations > 0:
    export.export_max_activations(data_dir, files, ml_model, model_name,
                                  X, layers, filters, number=args.n_max_activations)

ONLY_MAX_ACTIVATIONS = False
if ONLY_MAX_ACTIVATIONS:
    # Only consider highest activating images
    N = 100
    X, y, files, predictions = util.keep_max_activations(
        ml_model, model_name, X, y, files, layers[0], filters[layers[0]][0], N)
    model_name += f"_{N}_max_activations"
else:
    # Predictions can take some time for larger models so we cache them on disk
    predictions_path = Path("results", model_name, "predictions.pkl")
    if predictions_path.exists():
        with open(predictions_path, "rb") as predictions_file:
            predictions = pickle.load(predictions_file)
    else:
        predictions = tf.argmax(ml_model.predict(
            X.batch(128).cache().prefetch(tf.data.AUTOTUNE)), axis=1).numpy()
        with open(predictions_path, "wb") as output:
            pickle.dump(predictions, output)

export.export_all(ml_model, model_name, X, y, predictions,
                  files, layers, filters, str(data_dir),
                  layer_aggregation=layer_aggregation, filter_aggregation=filter_aggregation)
