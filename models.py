import tensorflow_datasets as tfds
import tensorflow as tf
import util
from pathlib import Path
import pickle
import os


def get_data_set(data_path, data_set, data_set_size, split='test'):
    ds, info = tfds.load(
        data_set, split=split, shuffle_files=False, with_info=True, data_dir=data_path)
    ds = ds.take(data_set_size)
    image_dir = Path(data_path, data_set, split)

    if not image_dir.exists() or len(os.listdir(image_dir)) < ds.cardinality().numpy():
        util.export_images(image_dir, ds)
    label_path = Path(image_dir, "label_names.pkl")
    if not label_path.exists():
        label_names = {label: info.features["label"].int2str(
            label) for label in range(0, info.features["label"].num_classes)}
        with open(label_path, 'wb') as handle:
            pickle.dump(label_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X = ds.map(lambda elem: elem['image'])
    y = ds.map(lambda elem: elem['label'])
    file_names = []
    for i, item in enumerate(ds):
        if "id" in item:
            file_name = item['id'].numpy().decode("utf-8") + ".jpeg"
        elif "file_name" in item:
            file_name = item['file_name'].numpy().decode("utf-8")
        else:
            file_name = f"{i}.jpeg"
        file_names.append(file_name)

    return X, y, file_names, image_dir


def setup_mnist(X):

    def normalize_img(image):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.
    X = X.map(lambda row: normalize_img(row),
              num_parallel_calls=tf.data.AUTOTUNE)

    model_save_name = 'mnist_classifier'
    path = F"{model_save_name}"
    model = tf.keras.models.load_model(path)
    return model, X


def setup_cifar10(X):

    def normalize_img(image):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.

    X = X.map(lambda row: normalize_img(row),
              num_parallel_calls=tf.data.AUTOTUNE)

    model_save_name = 'cifar10_classifier'
    path = F"{model_save_name}"
    model = tf.keras.models.load_model(path)
    model.summary()
    return model, X


def setup_inception_v1(X):
    from inception_v1 import InceptionV1

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
    X = X.map(lambda row: transform_images(row, 224),
              num_parallel_calls=tf.data.AUTOTUNE)
    return model, X


def setup_inception_v3(X):
    def transform_images(image, new_size):
        return tf.keras.applications.inception_v3.preprocess_input(tf.keras.layers.Resizing(new_size, new_size)(image))
    X = X.map(lambda row: transform_images(row, 299),
              num_parallel_calls=tf.data.AUTOTUNE)

    model = tf.keras.applications.InceptionV3(
        include_top=True, weights='imagenet')

    return model, X
