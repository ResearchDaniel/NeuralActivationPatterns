"""Provides different models to use for pattern analysis."""
import os
import pickle
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

import util


def get_data_set(data_path, data_set, data_set_size, split='test'):
    data, ds_stats = tfds.load(
        data_set, split=split, shuffle_files=False, with_info=True, data_dir=data_path)
    data = data.take(data_set_size)
    image_dir = Path(data_path, data_set, split)

    if not image_dir.exists() or len(os.listdir(image_dir)) < data.cardinality().numpy():
        util.export_images(image_dir, data)
    label_path = Path(image_dir, "label_names.pkl")
    if not label_path.exists():
        ds_info_label = ds_stats.features["label"]
        label_names = {label: ds_info_label.int2str(
            label) for label in range(0, ds_info_label.num_classes)}
        with open(label_path, 'wb') as handle:
            pickle.dump(label_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

    file_names = []
    for i, item in enumerate(data):
        if "id" in item:
            file_name = item['id'].numpy().decode("utf-8") + ".jpeg"
        elif "file_name" in item:
            file_name = item['file_name'].numpy().decode("utf-8")
        else:
            file_name = f"{i}.jpeg"
        file_names.append(file_name)

    return (data.map(lambda elem: elem['image']),
            data.map(lambda elem: elem['label']),
            file_names, image_dir)


def setup_mnist(data_set):

    def normalize_img(image):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.
    data_set = data_set.map(normalize_img,
                            num_parallel_calls=tf.data.AUTOTUNE)

    model_save_name = 'models/mnist_classifier'
    path = F"{model_save_name}"
    model = tf.keras.models.load_model(path)
    return model, data_set


def setup_cifar10(data_set):

    def normalize_img(image):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.

    data_set = data_set.map(normalize_img,
                            num_parallel_calls=tf.data.AUTOTUNE)

    model_save_name = 'models/cifar10_classifier'
    path = F"{model_save_name}"
    model = tf.keras.models.load_model(path)
    model.summary()
    return model, data_set


def setup_inception_v3(data_set):
    def transform_images(image, new_size):
        return tf.keras.applications.inception_v3.preprocess_input(
            tf.keras.layers.Resizing(new_size, new_size)(image))
    data_set = data_set.map(lambda row: transform_images(row, 299),
                            num_parallel_calls=tf.data.AUTOTUNE)

    model = tf.keras.applications.InceptionV3(
        include_top=True, weights='imagenet')

    return model, data_set


def setup_resnet50(data_set):
    def transform_images(image, new_size):
        return tf.keras.applications.resnet50.preprocess_input(
            tf.keras.layers.Resizing(new_size, new_size)(image))
    data_set = data_set.map(lambda row: transform_images(row, 224),
                            num_parallel_calls=tf.data.AUTOTUNE)

    model = tf.keras.applications.resnet50.ResNet50(
        include_top=True, weights='imagenet')

    return model, data_set


def setup_resnet50_v2(data_set):
    def transform_images(image, new_size):
        return tf.keras.applications.resnet_v2.preprocess_input(
            tf.keras.layers.Resizing(new_size, new_size)(image))
    data_set = data_set.map(lambda row: transform_images(row, 224),
                            num_parallel_calls=tf.data.AUTOTUNE)

    model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=True, weights='imagenet')

    return model, data_set
