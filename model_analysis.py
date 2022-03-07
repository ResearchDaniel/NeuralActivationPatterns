from matplotlib import image
import nap
from nap import cache
import numpy as np
import export

import util
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from pathlib import Path
import h5py
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


# Model / data parameters


def setup_mnist(image_dir, data_set_size):
    ds, info = tfds.load(
        'mnist', split='train', as_supervised=True, shuffle_files=False, with_info=True)
    if not image_dir.exists():
        util.export_images(image_dir, ds)

    ds = ds.take(data_set_size)

    X = ds.map(lambda image, label: image)
    y = ds.map(lambda image, label: label)
    file_names = tf.data.Dataset.from_tensor_slices(
        [f"{i}.jpeg" for i in range(0, ds.cardinality().numpy())])

    def normalize_img(image):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.
    X = X.map(lambda row: normalize_img(row),
              num_parallel_calls=tf.data.AUTOTUNE)

    model_save_name = 'mnist_classifier'
    path = F"{model_save_name}"

    model_name = f"MNIST_Classifier_{ds.cardinality().numpy()}"
    model = keras.models.load_model(path)
    model.summary()
    return model, model_name, X, y, file_names


def setup_cifar10(image_dir, data_set_size):
    ds, info = tfds.load(
        'cifar10', split='test', as_supervised=True, shuffle_files=False, with_info=True)
    if not image_dir.exists():
        util.export_images(image_dir, ds)

    ds = ds.take(data_set_size)

    X = ds.map(lambda image, label: image)
    y = ds.map(lambda image, label: label)
    file_names = tf.data.Dataset.from_tensor_slices(
        [f"{i}.jpeg" for i in range(0, ds.cardinality().numpy())])

    def normalize_img(image):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.
    X = X.map(lambda row: normalize_img(row),
              num_parallel_calls=tf.data.AUTOTUNE)

    model_save_name = 'cifar10_classifier'
    path = F"{model_save_name}"

    model_name = f"CIFAR-10_Classifier_{ds.cardinality().numpy()}"
    model = keras.models.load_model(path)
    model.summary()
    return model, model_name, X, y, file_names


def setup_inception_v1(data_set_size):

    import tensorflow as tf
    # from tf_slim.nets import inception_v1
    from inception_v1 import InceptionV1
    # import tensorflow_hub as hub
    # print(tf.__version__)
    ds, info = tfds.load('imagenet2012', split='train', shuffle_files=False,
                         with_info=True, data_dir=args.data_path)
    ds = ds.take(data_set_size)
    # ds = ds.take(1000)
    print(info.features)
    # model = tf.keras.Sequential([
    #     hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v1/classification/4")
    # ])
    # tf.keras.backend.set_image_data_format('channels_last')
    model_name = f"InceptionV1_{ds.cardinality().numpy()}"
    model = InceptionV1()
    # i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
    # x = tf.keras.layers.Resizing(224, 224)(i)
    # x = tf.keras.applications.imagenet_utils.preprocess_input(x)
    # x = model(x)
    # model = tf.keras.Model(inputs=[i], outputs=[x])

    def transform_images(image, new_size):
        img = tf.keras.layers.Rescaling(scale=1./255)(image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [new_size, new_size])
        return img
        return tf.keras.applications.imagenet_utils.preprocess_input(
            tf.keras.layers.Resizing(new_size, new_size)(image), mode='torch')
        # return tf.keras.layers.Resizing(new_size, new_size)(image)
        return tf.keras.applications.imagenet_utils.preprocess_input(
            tf.cast(image, tf.float32), mode='torch')
    X = ds.map(lambda elem: elem['image'])
    y = ds.map(lambda elem: elem['label'])
    file_names = ds.map(lambda elem: elem['file_name'])
    X = X.map(lambda row: transform_images(row, 224),
              num_parallel_calls=tf.data.AUTOTUNE)
    print(model.summary())
    return model, model_name, X, y, file_names


def setup_inception_v3():
    import tensorflow_datasets as tfds
    import tensorflow as tf
    print(tf.__version__)
    ds, info = tfds.load('imagenet2012', split='train', shuffle_files=False,
                         with_info=True, data_dir=args.data_path)
    ds = ds.take(100000)
    print(info.features)
    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape

    def transform_images(image, new_size):
        return tf.keras.applications.inception_v3.preprocess_input(tf.keras.layers.Resizing(new_size, new_size)(image))
    X = ds.map(lambda elem: elem['image'])
    y = ds.map(lambda elem: elem['label'])
    # , num_parallel_calls=tf.data.AUTOTUNE)
    X = X.map(lambda row: transform_images(row, 299))
    model_name = "InceptionV3"
    model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')
    print(model.summary())
    return model, model_name, X, y


data_set_size = args.size
image_dir = Path(args.data_path, "MNIST")
model, model_name, X, y, file_names = setup_mnist(image_dir, data_set_size)

# image_dir = Path(args.data_path, "CIFAR10")
# model, model_name, X, y, file_names = setup_cifar10(image_dir, data_set_size)
layers = [layer.name for layer in model.layers]
#layers = ["conv2d_1"]
layer = "conv2d_1"
filterId = 0

# image_dir = Path(args.data_path)
# model, model_name, X, y, file_names = setup_inception_v1(data_set_size)
# layers = ['Mixed_4b_Concatenated', 'Mixed_5b_Concatenated']
# layer = 'Mixed_4b_Concatenated'
# filterId = 409

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
files = list(tfds.as_numpy(file_names))
predictions = tf.argmax(model.predict(
    X.batch(128).cache().prefetch(tf.data.AUTOTUNE)), axis=1).numpy()
export.export_all(model, model_name, X, y, predictions,
                  files, layers, str(image_dir), agg_func=agg_func)
