import tensorflow_datasets as tfds
import tensorflow as tf
import util


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
    model = tf.keras.models.load_model(path)
    model.summary()
    return model, model_name, X, y, file_names


def setup_cifar10(image_dir, data_set_size):
    ds, info = tfds.load(
        'cifar10', split='train', as_supervised=True, shuffle_files=False, with_info=True)
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
    model = tf.keras.models.load_model(path)
    model.summary()
    return model, model_name, X, y, file_names


def setup_inception_v1(image_dir, data_set_size):

    import tensorflow as tf
    # from tf_slim.nets import inception_v1
    from inception_v1 import InceptionV1
    # import tensorflow_hub as hub
    # print(tf.__version__)
    ds, info = tfds.load('imagenet2012', split='train', shuffle_files=False,
                         with_info=True, data_dir=image_dir)
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


def setup_inception_v3(image_dir, data_set_size):
    print(tf.__version__)
    ds, info = tfds.load('imagenet2012', split='train', shuffle_files=False,
                         with_info=True, data_dir=image_dir)
    ds = ds.take(data_set_size)
    print(info.features)
    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape

    def transform_images(image, new_size):
        return tf.keras.applications.inception_v3.preprocess_input(tf.keras.layers.Resizing(new_size, new_size)(image))
    X = ds.map(lambda elem: elem['image'])
    y = ds.map(lambda elem: elem['label'])
    file_names = ds.map(lambda elem: elem['file_name'])
    X = X.map(lambda row: transform_images(row, 299), num_parallel_calls=tf.data.AUTOTUNE)

    model_name = f"InceptionV3_{ds.cardinality().numpy()}"
    model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')
    print(model.summary())
    return model, model_name, X, y, file_names

