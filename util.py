

def filter_tf_dataset(dataset, indices):
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import numpy as np
    # https://stackoverflow.com/questions/66410340/filter-tensorflow-dataset-by-id?noredirect=1&lq=1
    m_X_ds = dataset.enumerate()  # Create index,value pairs in the dataset.
    keys_tensor = tf.constant(np.asarray(indices, dtype=np.int32))
    vals_tensor = tf.ones_like(keys_tensor)  # Ones will be casted to True.

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=0)  # If index not in table, return 0.


    def hash_table_filter(index, value):
        table_value = table.lookup(tf.cast(index, tf.int32))  # 1 if index in arr, else 0.
        index_in_arr =  tf.cast(table_value, tf.bool) # 1 -> True, 0 -> False
        return index_in_arr

    filtered_ds = m_X_ds.filter(hash_table_filter)
    # Reorder to same order as 'indices'
    items = list(tfds.as_numpy(filtered_ds))
    mapping = dict(items)
    return [mapping[x] for x in indices]  


