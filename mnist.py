import nap
import numpy as np
import export
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from pathlib import Path
import h5py
import tensorflow_datasets as tfds
print(tf.__version__)


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
    nRows =  int(sampleSize / nCols)
    img = np.array([image.reshape(imSize) for image in images])
    fig = px.imshow(img, facet_col=0, binary_string=True, facet_col_wrap=nCols, 
                    #width=int(nCols*imSize[0]*img_scale), 
                    #height=50+int(nRows*10 + nRows*imSize[1]*img_scale), 
                    #facet_row_spacing = min(1/(nRows - 1), )
                    title=title
                    )
    # Set facet titles
    for i, im in enumerate(labels):
        fig.layout.annotations[plotly_annotation_index(nRows, nCols, i)]['text'] = im
    #fig.update_yaxes(automargin=False, title=title)
    #fig.update_xaxes(automargin=True)
    fig.update_layout(margin=dict(b=0))
    #fig.update_annotations(standoff=10)
    fig.show()

def show_pattern(average, representatives, representativeLabels, outliers, outlierLabels, title):
    import plotly.express as px
    import plotly.subplots as sp
    if representatives[0].shape[2] > 1:
        imSize = (representatives[0].shape[0], representatives[0].shape[1], representatives[0].shape[2])
    else:
        imSize = (representatives[0].shape[0], representatives[0].shape[1])
    images = np.array([average.reshape(imSize)] + [rep.reshape(imSize) for rep in representatives] + [otl.reshape(imSize) for otl in outliers])
    sampleSize = len(images)
    images_per_row = 10
    nCols = min(sampleSize, images_per_row)
    nRows =  int(sampleSize / nCols)
    fig = px.imshow(images, facet_col=0, binary_string=True, title = title)
    fig.layout.annotations[0]['text'] = "Average"
    for i, _ in enumerate(representatives):
        idx = plotly_annotation_index(nRows, nCols, 1+i)
        fig.layout.annotations[idx]['text'] = representativeLabels[i]
    for i, _ in enumerate(outliers):
        idx = plotly_annotation_index(nRows, nCols, len(representatives)+1+i)
        fig.layout.annotations[idx]['text'] = outlierLabels[i]
    fig.show()

def show_images(images, labels, title, images_per_row = 10, img_scale = 7.0):
    show_image_grid(images, labels, title, images_per_row, img_scale)

def show_outliers(model_name, X, y, patterns, quantile = 0.95, n = None):
    outliers = nap.outliers(patterns, quantile, n)
    images = filter_tf_dataset(X, outliers)
    labels = [f"{y[i]} | id:{i}" for i in outliers]
    title = F"{model_name}, layer: {layer}, outliers"
    show_images(images, labels, title)
# Model / data parameters
def setupMNIST():
    import tensorflow_datasets as tfds
    
    ds, info = tfds.load('mnist', split='test', as_supervised=True, shuffle_files=False, with_info=True)
    ds = ds.take(2000)

    X = ds.map(lambda image, label: image)
    y = ds.map(lambda image, label: label)
    def normalize_img(image):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.
    X = X.map(lambda row:normalize_img(row), num_parallel_calls=tf.data.AUTOTUNE)

    model_save_name = 'mnist_classifier'
    path = F"{model_save_name}" 
    #model.save(path)
    model_name = f"MNIST_Classifier_{ds.cardinality().numpy()}"
    model = keras.models.load_model(path)
    # X = np.stack(list(X.as_numpy_iterator()), axis=0 )
    # y = np.array(list(y.as_numpy_iterator()))
    # y = keras.utils.to_categorical(y, 10)
    # score = model.evaluate(X, y, verbose=0)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])
    #model.save_w(f"{model_save_name}")
    model.summary()  
    return model, model_name, X, y

def setupInceptionV1():
    
    import tensorflow as tf
    import tf_slim as slim
    #from tf_slim.nets import inception_v1
    from inception_v1 import InceptionV1
    #import tensorflow_hub as hub
    # print(tf.__version__)
    ds, info = tfds.load('imagenet2012', split='train', shuffle_files=False, with_info=True, data_dir='D:/data/tensorflow_datasets')
    ds = ds.take(100000)
    #ds = ds.take(1000)
    # print(info.features)
    # model = tf.keras.Sequential([
    #     hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v1/classification/4")
    # ])
    #tf.keras.backend.set_image_data_format('channels_last')
    model_name = "InceptionV1"
    model = InceptionV1()
    # i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
    # x = tf.keras.layers.Resizing(224, 224)(i)
    # x = tf.keras.applications.imagenet_utils.preprocess_input(x)
    # x = model(x)
    # model = tf.keras.Model(inputs=[i], outputs=[x])
    def transform_images(image, new_size):
        img = tf.keras.layers.Rescaling(scale=1./255)(image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [new_size,new_size])
        return img
        return tf.keras.applications.imagenet_utils.preprocess_input(
             tf.keras.layers.Resizing(new_size, new_size)(image), mode='torch')
        #return tf.keras.layers.Resizing(new_size, new_size)(image)
        return tf.keras.applications.imagenet_utils.preprocess_input(
            tf.cast(image, tf.float32), mode='torch')
    X = ds.map(lambda elem: elem['image'])
    y = ds.map(lambda elem: elem['label'])
    X = X.map(lambda row:transform_images(row, 224), num_parallel_calls=tf.data.AUTOTUNE)
    print(model.summary())
    return model, model_name, X, y

def setupInceptionV3():
    import tensorflow_datasets as tfds
    import tensorflow as tf
    print(tf.__version__)
    ds, info = tfds.load('imagenet2012', split='train', shuffle_files=False, with_info=True, data_dir='D:/data/tensorflow_datasets')
    ds = ds.take(100000)
    print(info.features)
    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape

    def transform_images(image, new_size):
        return tf.keras.applications.inception_v3.preprocess_input(tf.keras.layers.Resizing(new_size, new_size)(image))
    X = ds.map(lambda elem: elem['image'])
    y = ds.map(lambda elem: elem['label'])
    X = X.map(lambda row:transform_images(row, 299)) #, num_parallel_calls=tf.data.AUTOTUNE)
    model_name = "InceptionV3"
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    print(model.summary())
    return model, model_name, X, y

EXPORT_LOCATION = Path("results")


def activations_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_activations.h5')
def activations_agg_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_activations_aggregated.h5')    
def layer_patterns_path(destination, model_name, layer):
    return Path(destination, model_name, layer, 'layer_patterns.h5')   
def filter_patterns_path(destination, model_name, layer, filter):
    return Path(destination, model_name, layer, 'filters', f'{filter}.h5')   

def precomputeActivations(dataset, model, model_name, layers, destination=EXPORT_LOCATION, mode = 'w'):
    batch_size = 1000

    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        act_path = activations_path(destination, model_name, layer)
        act_path.parent.mkdir(parents=True, exist_ok=True)
        agg_path = activations_agg_path(destination, model_name, layer)
        with h5py.File(act_path, mode) as f, h5py.File(agg_path, mode) as f_agg:
            # Dummy computation to get the output shape
            activations = ap.layer_activations(layer, dataset.take(1).batch(1))
            output_shape = list(activations.shape)
            # Ignore first entry, which is batch size 
            agg_shape = nap.layer_activation_aggregation_shape(output_shape[1:])
            total_num_inputs = dataset.cardinality().numpy()
            
            output_shape[0] = total_num_inputs
            dset = f.create_dataset("activations", output_shape, compression="gzip")
            if isinstance(agg_shape, list):
                agg_shape = [total_num_inputs] + agg_shape
            else:
                agg_shape = [total_num_inputs] + [agg_shape]

            dset_aggregated = f_agg.create_dataset("activations", agg_shape, compression="gzip")
            i = 0
            for ds in dataset.batch(batch_size):
                ap = nap.NeuralActivationPattern(model)
                num_inputs = ds.shape[0]
                activations = ap.layer_activations(layer, ds)
                dset[i:i+num_inputs] = activations
                dset_aggregated[i:i+num_inputs] = nap.layer_activations_aggregation(activations)
                i += num_inputs

def precomputeLayerAggregation(dataset, model, model_name, layers, agg_func = np.mean, destination=EXPORT_LOCATION):
    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        activations, f = layer_activations(dataset, model, model_name, layer, destination)
        output_shape = list(activations.shape)
        agg_shape = nap.layer_activation_aggregation_shape(output_shape[1:], agg_func)
        total_num_inputs = output_shape[0]
        if isinstance(agg_shape, list):
            agg_shape = [total_num_inputs] + agg_shape
        else:
            agg_shape = [total_num_inputs] + [agg_shape]
        agg_path = activations_agg_path(destination, model_name, layer)
        with h5py.File(agg_path, 'w') as f_agg:
            dset_aggregated = f_agg.create_dataset("activations", agg_shape, compression="gzip")
            for i, activation in enumerate(activations):
                dset_aggregated[i] = nap.layer_activation_aggregation(activation[()], agg_func)
        f.close()


def precomputeLayerPatterns(dataset, model, model_name, layers, destination=EXPORT_LOCATION):
    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        activations, f = layer_activations_agg(dataset, model, model_name, layer, destination)
        patterns = ap.activity_patterns(layer, activations=activations)
        patterns_path = layer_patterns_path(destination, model_name, layer)
        f.close()
        patterns.to_hdf(patterns_path, f'{layer}') 

def precomputeFilterPatterns(dataset, model, model_name, layers, filters = None, destination=EXPORT_LOCATION):
    ap = nap.NeuralActivationPattern(model)
    for layer in layers:
        # [()] retireves all data because slicing the filter is super-slow in hdf5  
        activations, f = layer_activations(dataset, model, model_name, layer, destination)
        if filters is None:
            filters = range(activations.shape[-1])
        for filter in filters:
            
            patterns = ap.activity_patterns(f'{layer}:{filter}', activations=activations)  
            path = filter_patterns_path(destination, model_name, layer, filter)
            path.parent.mkdir(parents=True, exist_ok=True)
            patterns.to_hdf(path, f'{layer}/filter_{filter}')     
        f.close()
 
def layer_activations(dataset, model, model_name, layer, destination=EXPORT_LOCATION):
    path = activations_path(destination, model_name, layer)
    if not path.exists():
        precomputeActivations(dataset, model, model_name, [layer], destination)
    f = h5py.File(path, 'r')
    return f["activations"], f
def layer_activations_agg(dataset, model, model_name, layer, destination=EXPORT_LOCATION):
    path = activations_agg_path(destination, model_name, layer)
    if not path.exists():
        precomputeActivations(dataset, model, model_name, [layer], destination)
    f = h5py.File(path, 'r')
    return f["activations"], f
def layer_patterns(X, model, model_name, layer, destination=EXPORT_LOCATION):
    path = layer_patterns_path(destination, model_name, layer)
    if not path.exists():
        precomputeLayerPatterns(X, model, model_name, [layer], destination)
    return pd.read_hdf(path)
def filter_patterns(X, model, model_name, layer, filter, destination=EXPORT_LOCATION):
    path = filter_patterns_path(destination, model_name, layer, filter)
    if not path.exists():
        precomputeFilterPatterns(X, model, model_name, [layer], [filter], destination)
    return pd.read_hdf(path)    


    
    
# model, model_name, X, y = setupMNIST()
# layers = ["conv2d", "max_pooling2d", "conv2d_1", "max_pooling2d_1", "flatten", "dropout", "dense"]
# layers = ["conv2d_1"]
# layer = "conv2d_1" 
# filterId = 0

model, model_name, X, y = setupInceptionV1()
layers = ['Mixed_4b_Concatenated']
layer = 'Mixed_4b_Concatenated'
filterId = 409
#precomputeActivations(X, model, model_name, layers=layers)
#precomputeLayerAggregation(X, model, model_name, layers=layers, agg_func=None)
#precomputeLayerPatterns(X, model, model_name, layers=layers)
#precomputeFilterPatterns(X, model, model_name, [layer], [filterId])
# X = X.take(10)
# y = y.take(10)
# X = X.batch(1)
y = list(tfds.as_numpy(y))
# ap = nap.NeuralActivationPattern(model)
# ap.layer_summary("conv2d", X, y)

def filter_analysis(model, model_name, X, y, layer, filter):
    patterns = filter_patterns(X, model, model_name, layer, filter)
    # Show pattern representatives for filter  
    sorted_patterns = nap.sort(patterns)

    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        if pattern_id == -1:
            continue
        to_average = filter_tf_dataset(X, pattern.index)
        avg = tf.keras.layers.Average()(to_average).numpy()
        centerIndices = pattern.head(1).index
        outliersIndices = pattern.tail(3).index
        centers = filter_tf_dataset(X, centerIndices)
        outliers = filter_tf_dataset(X, outliersIndices)
        centerLabels = [f"Representative | {y[i]}" for i in centerIndices]
        outlierLabels = [f"Outlier | {y[i]}" for i in outliersIndices]

        show_pattern(avg, centers, centerLabels, outliers, outlierLabels, F"{model_name}, Layer {layer}, Filter: {filter}, Pattern: {pattern_id}, Size: {len(pattern)}")


def filter_tf_dataset(dataset, indices):
    # https://stackoverflow.com/questions/66410340/filter-tensorflow-dataset-by-id?noredirect=1&lq=1
    m_X_ds = dataset.enumerate()  # Create index,value pairs in the dataset.
    keys_tensor = tf.constant(indices)
    vals_tensor = tf.ones_like(keys_tensor)  # Ones will be casted to True.

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=0)  # If index not in table, return 0.


    def hash_table_filter(index, value):
        table_value = table.lookup(index)  # 1 if index in arr, else 0.
        index_in_arr =  tf.cast(table_value, tf.bool) # 1 -> True, 0 -> False
        return index_in_arr

    filtered_ds = m_X_ds.filter(hash_table_filter)
    # Reorder to same order as 'indices'
    items = list(tfds.as_numpy(filtered_ds))
    mapping = dict(items)
    return [mapping[x] for x in indices]

def layer_analysis(model, model_name, X, y, layer):
    patterns = layer_patterns(X, model, model_name, layer)
    ap = nap.NeuralActivationPattern(model)
    #ap.layer_summary(layer, X, y, patterns).show()
    #activations = layer_activations(X, model, model_name, layer)
    #print(ap.layer_max_activations(layer, X, activations=activations))
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
    #     images = filter_tf_dataset(X, pattern_indices)
    #     labels = [f"{y[i]} | id:{i}" for i in pattern_indices]
    #     title = F"Layer: {layer}, pattern: {pattern_id}, size: {len(pattern)}"
    #     show_images(images, labels, title)

    # Show pattern representatives for layer  
    sorted_patterns = nap.sort(patterns)
    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        if pattern_id == -1:
            continue
        
        to_average = filter_tf_dataset(X, pattern.index)
        avg = tf.keras.layers.Average()(to_average).numpy()
        centerIndices = pattern.head(1).index
        #outliersIndices = pattern.tail(3).index
        outliersIndices = nap.outliers(pattern, n=3)
        centers = filter_tf_dataset(X, centerIndices)
        outliers = filter_tf_dataset(X, outliersIndices)
        
        centerLabels = [f"Representative | {y[i]}" for i in centerIndices]
        outlierLabels = [f"Outlier | {y[i]}" for i in outliersIndices]
        
        show_pattern(avg, centers, centerLabels, outliers, outlierLabels, F"{model_name}, Layer {layer}, Pattern: {pattern_id}, Size: {len(pattern)}")


#layer_analysis(model, model_name, X, y, layer)
filter_analysis(model, model_name, X, y, layer, filterId)

#export.export_all(ap, X, y, "MNIST", ["conv2d", "conv2d_1", "dense"])

