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
parser.add_argument(
    "--model", default='mnist',
    choices=["mnist", "cifar10", "inception_v3", "resnet50", "resnet-rs50"])
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
parser.add_argument("--minimum_pattern_size", type=int,
                    default=5)
parser.add_argument("--cluster_min_samples", type=int, default=5)
parser.add_argument("--cluster_selection_epsilon", type=float, default=0)
parser.add_argument("--cluster_selection_method", default="leaf", choices=["leaf", "eom"])
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
    elif model_type == "inception_v3":
        model, processing_data = models.setup_inception_v3(processing_data)
    elif model_type == "resnet50":
        model, processing_data = models.setup_resnet50(processing_data)
    elif model_type == "resnet-rs50":
        model, processing_data = models.setup_resnet_rs50(processing_data)
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
model_name += '_norm'
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

layer_aggregation = create_aggregation_function(args.layer_aggregation)
filter_aggregation = create_aggregation_function(args.filter_aggregation)
y = list(tfds.as_numpy(y))


ONLY_MAX_ACTIVATIONS = False
if ONLY_MAX_ACTIVATIONS:
    # Only consider highest activating images
    N = 100
    X, y, files, predictions = util.keep_max_activations(
        ml_model, model_name, X, y, files, layers[0], filters[layers[0]][0], N)
    model_name += f"_{N}_max_activations"
else:
    # Predictions can take some time for larger models so we cache them on disk
    predictions_dir = Path("results", model_name)
    predictions_path = Path("results", model_name, "predictions.pkl")
    if predictions_path.exists():
        with open(predictions_path, "rb") as predictions_file:
            predictions = pickle.load(predictions_file)
    else:
        predictions = tf.argmax(ml_model.predict(
            X.batch(128).cache().prefetch(tf.data.AUTOTUNE)), axis=1).numpy()
        predictions_dir.mkdir(parents=True, exist_ok=True)
        with open(predictions_path, "wb") as output:
            pickle.dump(predictions, output)

neural_activation_pattern = nap.NeuralActivationPattern(
    ml_model, layer_aggregation, filter_aggregation, args.minimum_pattern_size,
    min_samples=args.cluster_min_samples, cluster_selection_epsilon=args.cluster_selection_epsilon,
    cluster_selection_method=args.cluster_selection_method)
export.export_all(model_name, X, y, predictions, files, layers, filters, str(data_dir),
                  neural_activation_pattern, n_max_activations=args.n_max_activations)
