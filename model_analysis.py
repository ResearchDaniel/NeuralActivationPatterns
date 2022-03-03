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
args = parser.parse_args()


def plotly_annotation_index(nRows, nCols, i):
    # The first row correspond to the last annotation in Plotly...
    row = nRows - int(i / nCols) - 1
    col = i % nCols
    return row*nCols + col


def show_image_grid(images, labels, title, images_per_row, img_scale):
    import plotly.express as px
    if images[0].shape[2] > 1:
        imSize = (images[0].shape[0], images[0].shape[1], images[0].shape[2])
    else:
        imSize = (images[0].shape[0], images[0].shape[1])
    # Get indices of the largest
    sampleSize = len(images)
    nCols = min(sampleSize, images_per_row)
    nRows = int(sampleSize / nCols)
    img = np.array([image.reshape(imSize) for image in images])
    fig = px.imshow(img, facet_col=0, binary_string=True, facet_col_wrap=nCols,
                    # width=int(nCols*imSize[0]*img_scale),
                    # height=50+int(nRows*10 + nRows*imSize[1]*img_scale),
                    # facet_row_spacing = min(1/(nRows - 1), )
                    title=title
                    )
    # Set facet titles
    for i, im in enumerate(labels):
        fig.layout.annotations[plotly_annotation_index(
            nRows, nCols, i)]['text'] = im
    # fig.update_yaxes(automargin=False, title=title)
    # fig.update_xaxes(automargin=True)
    fig.update_layout(margin=dict(b=0))
    # fig.update_annotations(standoff=10)
    fig.show()


def show_pattern(average, representatives, representativeLabels, outliers, outlierLabels, title):
    import plotly.express as px
    if representatives[0].shape[2] > 1:
        imSize = (representatives[0].shape[0],
                  representatives[0].shape[1], representatives[0].shape[2])
    else:
        imSize = (representatives[0].shape[0], representatives[0].shape[1])
    images = np.array([average.reshape(imSize)] + [rep.reshape(imSize)
                      for rep in representatives] + [otl.reshape(imSize) for otl in outliers])
    sampleSize = len(images)
    images_per_row = 10
    nCols = min(sampleSize, images_per_row)
    nRows = int(sampleSize / nCols)
    fig = px.imshow(images, facet_col=0, binary_string=True, title=title)
    fig.layout.annotations[0]['text'] = "Average"
    for i, _ in enumerate(representatives):
        idx = plotly_annotation_index(nRows, nCols, 1+i)
        fig.layout.annotations[idx]['text'] = representativeLabels[i]
    for i, _ in enumerate(outliers):
        idx = plotly_annotation_index(nRows, nCols, len(representatives)+1+i)
        fig.layout.annotations[idx]['text'] = outlierLabels[i]
    fig.show()


def show_images(images, labels, title, images_per_row=10, img_scale=7.0):
    show_image_grid(images, labels, title, images_per_row, img_scale)


def show_outliers(model_name, X, y, patterns, quantile=0.95, n=None):
    outliers = nap.outliers(patterns, quantile, n)
    images = util.filter_tf_dataset(X, outliers)
    labels = [f"{y[i]} | id:{i}" for i in outliers]
    title = F"{model_name}, layer: {layer}, outliers"
    show_images(images, labels, title)
# Model / data parameters


def setup_mnist(image_dir):
    ds, info = tfds.load(
        'mnist', split='test', as_supervised=True, shuffle_files=False, with_info=True)
    if not image_dir.exists():
        util.export_images(image_dir, ds)

    ds = ds.take(2000)

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


def setup_cifar10(image_dir):
    ds, info = tfds.load(
        'cifar10', split='test', as_supervised=True, shuffle_files=False, with_info=True)
    if not image_dir.exists():
        util.export_images(image_dir, ds)

    ds = ds.take(2000)

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


def setup_inception_v1():

    import tensorflow as tf
    # from tf_slim.nets import inception_v1
    from inception_v1 import InceptionV1
    # import tensorflow_hub as hub
    # print(tf.__version__)
    ds, info = tfds.load('imagenet2012', split='train', shuffle_files=False,
                         with_info=True, data_dir=args.data_path)
    ds = ds.take(100000)
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


# image_dir = Path(args.data_path, "MNIST")
# model, model_name, X, y, file_names = setup_mnist(image_dir)
image_dir = Path(args.data_path, "CIFAR10")
model, model_name, X, y, file_names = setup_cifar10(image_dir)
layers = [layer.name for layer in model.layers]
#layers = ["conv2d_1"]
layer = "conv2d_1"
filterId = 0

# model, model_name, X, y, file_names = setup_inception_v1()
# layers = ['Mixed_4b_Concatenated', 'Mixed_5b_Concatenated']
# layer = 'Mixed_4b_Concatenated'
# filterId = 409
# export.export_activations(X, model, model_name, layers=layers)
# export.export_layer_aggregation(X, model, model_name, layers=layers, agg_func=None)
# export.export_layer_patterns(X, model, model_name, layers=layers)
# export.export_filter_patterns(X, model, model_name, [layer], [filterId])
# X = X.take(10)
# y = y.take(10)
# X = X.batch(1)
y = list(tfds.as_numpy(y))
# ap = nap.NeuralActivationPattern(model)
# ap.layer_summary("conv2d", X, y)


def filter_analysis(model, model_name, X, y, layer, filter):
    patterns = nap.cache.get_filter_patterns(
        X, model, model_name, layer, filter)
    # Show pattern representatives for filter
    sorted_patterns = nap.sort(patterns)

    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        if pattern_id == -1:
            continue
        to_average = util.filter_tf_dataset(X, pattern.index)
        avg = tf.keras.layers.Average()(to_average).numpy()
        centerIndices = pattern.head(1).index
        outliersIndices = pattern.tail(3).index
        centers = util.filter_tf_dataset(X, centerIndices)
        outliers = util.filter_tf_dataset(X, outliersIndices)
        centerLabels = [f"Representative | {y[i]}" for i in centerIndices]
        outlierLabels = [f"Outlier | {y[i]}" for i in outliersIndices]

        show_pattern(avg, centers, centerLabels, outliers, outlierLabels,
                     F"{model_name}, Layer {layer}, Filter: {filter}, Pattern: {pattern_id}, Size: {len(pattern)}")


def layer_analysis(model, model_name, X, y, layer):
    patterns = nap.cache.get_layer_patterns(X, model, model_name, layer)
    ap = nap.NeuralActivationPattern(model)
    # ap.layer_summary(layer, X, y, patterns).show()
    # activations = export.get_layer_activations(X, model, model_name, layer)
    # print(ap.layer_max_activations(layer, X, activations=activations))
    #

    show_outliers(model_name, X, y, patterns, n=100)

    # Show a sample subset from each pattern
    nSamplesPerLayer = 10
    print(nap.sample(patterns))
    pattern_samples = nap.head(patterns, nSamplesPerLayer)
    # for pattern_id, pattern in pattern_samples.groupby('patternId'):
    #     if pattern_id == -1:
    #         continue
    #     pattern_indices = pattern.index
    #     images = util.filter_tf_dataset(X, pattern_indices)
    #     labels = [f"{y[i]} | id:{i}" for i in pattern_indices]
    #     title = F"Layer: {layer}, pattern: {pattern_id}, size: {len(pattern)}"
    #     show_images(images, labels, title)

    # Show pattern representatives for layer
    sorted_patterns = nap.sort(patterns)
    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        if pattern_id == -1:
            continue

        to_average = util.filter_tf_dataset(X, pattern.index)
        avg = tf.keras.layers.Average()(to_average).numpy()
        centerIndices = pattern.head(1).index
        outliersIndices = pattern.tail(3).index
        # outliersIndices = nap.outliers(pattern, n=3)
        centers = util.filter_tf_dataset(X, centerIndices)
        outliers = util.filter_tf_dataset(X, outliersIndices)

        centerLabels = [f"Representative | {y[i]}" for i in centerIndices]
        outlierLabels = [f"Outlier | {y[i]}" for i in outliersIndices]

        show_pattern(avg, centers, centerLabels, outliers, outlierLabels,
                     F"{model_name}, Layer {layer}, Pattern: {pattern_id}, Size: {len(pattern)}")


# layer_analysis(model, model_name, X, y, layer)
# filter_analysis(model, model_name, X, y, layer, filterId)
files = list(tfds.as_numpy(file_names))
predictions = tf.argmax(model.predict(X.batch(128)), axis=1).numpy()
export.export_all(model, model_name, X, y, predictions,
                  files, layers, str(image_dir))
