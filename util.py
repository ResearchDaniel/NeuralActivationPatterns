from PIL import Image
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


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


def export_images(image_dir, dataset):

    image_dir.mkdir(parents=True, exist_ok=True)
    for i, item in dataset.enumerate():
        image = np.squeeze(item[0].numpy())
        if image.shape[-1] == 1:
            image = Image.fromarray((image).astype(np.uint8), 'L')
        else:
            image = Image.fromarray((image).astype(np.uint8), 'RGB')
        image.save(Path(image_dir, f"{i}.jpeg"))
