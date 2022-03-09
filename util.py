from PIL import Image
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pickle
import ast


def filter_tf_dataset(dataset, indices):
    """ Only keep supplied indices in dataset

    Args:
        dataset (Tensorflow.data.Dataset): Tensorflow 
        indices (list): List like integer array

    Returns:
        list: List of tensors
    """

    # https://stackoverflow.com/questions/66410340/filter-tensorflow-dataset-by-id?noredirect=1&lq=1
    m_X_ds = dataset.enumerate()  # Create index,value pairs in the dataset.
    keys_tensor = tf.constant(np.asarray(indices, dtype=np.int32))
    vals_tensor = tf.ones_like(keys_tensor)  # Ones will be casted to True.

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=0)  # If index not in table, return 0.

    def hash_table_filter(index, value):
        # 1 if index in arr, else 0.
        table_value = table.lookup(tf.cast(index, tf.int32))
        index_in_arr = tf.cast(table_value, tf.bool)  # 1 -> True, 0 -> False
        return index_in_arr

    filtered_ds = m_X_ds.filter(hash_table_filter)
    # Reorder to same order as 'indices'
    items = list(tfds.as_numpy(filtered_ds))
    mapping = dict(items)
    return [mapping[x] for x in indices]


def average_images(image_dir, file_names, indices, image_out_size):
    """ Average images located in image_dir.
        Images will be resized to image_out_size.
    Args:
        image_dir (Path): Location of images.
        file_names (_type_): List of file names in image_dir.
        indices (_type_): Indices to file_names of images to average.
        image_out_size (list): Width, height, and number of channels of resulting image average.  

    Returns:
        Image: Average image with size according to image_out_size
    """
    num_channels = image_out_size[-1]
    if num_channels == 1:
        arr = np.zeros(image_out_size[0:2], np.float64)
    else:
        arr = np.zeros(image_out_size, np.float64)
    N = len(indices)
    # Build up average pixel intensities, casting each image as an array of floats
    for idx in indices:
        im = Image.open(
            Path(image_dir, file_names[idx])).resize(image_out_size[0:2])
        imarr = np.array(im, dtype=np.float64)
        arr = arr+imarr/N

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)

    # Generate, save and preview final image
    if len(image_out_size) == 2 or num_channels == 1:
        out = Image.fromarray(arr, mode='L')
    else:
        out = Image.fromarray(arr, mode='RGB')
    return out


def export_images(image_dir, dataset):

    image_dir.mkdir(parents=True, exist_ok=True)
    for i, item in dataset.enumerate():
        if type(item) is dict:
            image = np.squeeze(item["image"].numpy())
            if "file_name" in item:
                file_name = item["file_name"].numpy().decode("utf-8")
            elif "id" in item:
                # Exist in CIFAR10
                file_name = item["id"].numpy().decode("utf-8") + ".jpeg"
            else:
                file_name = f"{i}.jpeg"

        else:
            image = np.squeeze(item[0].numpy())
            file_name = f"{i}.jpeg"
        if len(image.shape) == 2 or image.shape[-1] == 1:
            image = Image.fromarray((image).astype(np.uint8), 'L')
        else:
            image = Image.fromarray((image).astype(np.uint8), 'RGB')
        image.save(Path(image_dir, file_name))


def export_imagenet_labels(label_path):
    if not label_path.exists():
        with open('imagenet/imagenet1000_clsidx_to_labels.txt') as file:
            # Convert text to dict
            label_names = ast.literal_eval(file.read())
            with open(label_path, 'wb') as handle:
                pickle.dump(label_names, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
