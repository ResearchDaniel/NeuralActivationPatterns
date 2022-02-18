import nap
import numpy as np
import export
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

def show_image_grid(images, labels, img_idx, title, images_per_row, img_scale):
    import plotly.express as px
    if images[0].shape[2] > 1:
        imSize = (images[0].shape[0], images[0].shape[1], images[0].shape[2])
    else:
        imSize = (images[0].shape[0], images[0].shape[1])
    # Get indices of the largest 
    sampleSize = len(img_idx)
    nCols = min(sampleSize, images_per_row)
    nRows =  1 +sampleSize % images_per_row
    img = np.array([images[idx].reshape(imSize) for idx in img_idx])
    fig = px.imshow(img, facet_col=0, binary_string=True, facet_col_wrap=nCols, 
                    #width=int(nCols*imSize[0]*img_scale), 
                    #height=50+int(nRows*10 + nRows*imSize[1]*img_scale), 
                    title=title
                    )
    # Set facet titles
    for i, im in enumerate(img_idx):
        fig.layout.annotations[i]['text'] = f"{labels[im]}"
    #fig.update_yaxes(automargin=False, title=title)
    #fig.update_xaxes(automargin=True)
    fig.update_layout(margin=dict(b=0))
    #fig.update_annotations(standoff=10)
    fig.show()

def show_pattern(average, representatives, outliers, images, labels, title):
    import plotly.express as px
    import plotly.subplots as sp
    if images[0].shape[2] > 1:
        imSize = (images[0].shape[0], images[0].shape[1], images[0].shape[2])
    else:
        imSize = (images[0].shape[0], images[0].shape[1])
    img = np.array([average.reshape(imSize)] + [images[idx].reshape(imSize) for idx in representatives] + [images[idx].reshape(imSize) for idx in outliers])
    fig = px.imshow(img, facet_col=0, binary_string=True, title = title)
    fig.layout.annotations[0]['text'] = "Average"
    for i, im in enumerate(representatives):
        fig.layout.annotations[1+i]['text'] = f"Representative | {labels[im]}"
    for i, im in enumerate(outliers):
        fig.layout.annotations[len(representatives)+1+i]['text'] = f"Outlier | {labels[im]}"
    fig.show()

def show_images(images, labels, layer_img_idx, titles, images_per_row = 10, img_scale = 7.0):
    for layer, img_idx in enumerate(layer_img_idx):
        show_image_grid(images, labels, img_idx, titles[layer], images_per_row, img_scale)

# Model / data parameters
def setupMNIST():
    import tensorflow_datasets as tfds
    import tensorflow as tf
    ds, info = tfds.load('mnist', split='test', as_supervised=True, shuffle_files=False, with_info=True)
    X = ds.map(lambda image, label: image)
    y = ds.map(lambda image, label: label)
    def normalize_img(image):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.
    X = X.map(lambda row:normalize_img(row), num_parallel_calls=tf.data.AUTOTUNE)

    model_save_name = 'mnist_classifier'
    path = F"{model_save_name}" 
    #model.save(path)
    model = keras.models.load_model(path)

    #model.save_w(f"{model_save_name}")
    model.summary()  
    return model, X, y

def setupInceptionV3():
    import tensorflow_datasets as tfds
    import tensorflow as tf
    ds, info = tfds.load('imagenet_v2', split='test', shuffle_files=False, with_info=True)
    print(info.features)
    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape

    def transform_images(image, new_size):
        return tf.keras.applications.inception_v3.preprocess_input(tf.keras.layers.Resizing(new_size, new_size)(image))
    X = ds.map(lambda elem: elem['image'])
    y = ds.map(lambda elem: elem['label'])
    X = X.map(lambda row:transform_images(row, 299), num_parallel_calls=tf.data.AUTOTUNE)
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    print(model.summary())
    return model, X, y

def precomputeActivations(dataset, model, layers, path, mode = 'w'):
    num_shards = 10
    import tensorflow as tf
    import h5py
    with h5py.File(f'{path}.h5', mode) as f, h5py.File(f'{path}_aggregated.h5', mode) as f_agg:

        ap = nap.NeuralActivationPattern(model)

        for layer in layers:
            activations = ap.layer_activations(layer, dataset.take(1).batch(1))
            # Ignore first entry, which is batch size 
            agg_shape = nap.layer_activation_aggregation_shape(output_shape[1:])
            total_num_inputs = dataset.cardinality().numpy()
            
            output_shape[0] = total_num_inputs
            dset = f.create_dataset(f"{layer}", output_shape, compression="gzip")
            if isinstance(agg_shape, list):
                agg_shape = [total_num_inputs] + agg_shape
            else:
                agg_shape = [total_num_inputs] + [agg_shape]

            dset_aggregated = f_agg.create_dataset(f"{layer}", agg_shape, compression="gzip")
            i = 0
            for shard in range(num_shards):
                ds = dataset.shard(num_shards, shard).batch(1)
                ap = nap.NeuralActivationPattern(model)
                num_inputs = ds.cardinality().numpy()
                activations = ap.layer_activations(layer, ds)
                dset[i:i+num_inputs] = activations
                #dset_aggregated[i:i+num_inputs] = nap.layer_activation_aggregation(activations)
                dset_aggregated[i:i+num_inputs] = [activation.flatten() for activation in activations]
                i += num_inputs

def precomputePatterns(aggregated_activations_path, layer):
    import h5py
    with h5py.File(f'{aggregated_activations_path}.h5', 'r') as f_agg:
        activations_agg = f_agg[f'{layer}']
        ap = nap.NeuralActivationPattern(model)
        patterns = ap.layer_patterns(layer, agg_activations=activations_agg)
        patterns.to_feather(f'{aggregated_activations_path}_patterns_{layer}.feather')        

model, X, y = setupMNIST()

# precomputeActivations(X, model, ["conv2d", "max_pooling2d", "conv2d_1", "max_pooling2d_1", "flatten", "dropout", "dense"], 'results/mnist_activations')
# path = 'results/mnist_activations'
# #model, X, y = setupInceptionV3()
# #path = 'results/inceptionv3_activations'
# # precomputeActivations(X, model, ["mixed3","mixed4", "mixed5"], path)

X = X.take(10)
y = y.take(10)
X = X.batch(1)
y = list(y.as_numpy_iterator())
ap = nap.NeuralActivationPattern(model)

ap.layer_summary("conv2d", X, y)
def filter_analysis():
    # For now, simply test that these functions works
    layer = "conv2d_1"
    filterId = 0
    filter_patterns = ap.activity_patterns("conv2d:0", X)
    filter_patterns = ap.activity_patterns("0:0", X)
    # Show pattern representatives for filter  
    sorted_patterns = nap.sort(ap.filter_patterns(layer, filterId, X))
    
    images = list(X.unbatch().as_numpy_iterator())
    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        avg = nap.average(X, pattern.index)
        centers = pattern.head(1).index
        outliers = pattern.tail(3).index
        show_pattern(avg, centers, outliers, images, y, F"Layer {layer}, Filter: {filterId}, Pattern: {pattern_id}, Size: {len(pattern)}")


def layer_analysis():
    ap.layer_summary(5, X, y).show()
    print(ap.layer_max_activations(0, X))
    
    layerId = 5
    nSamplesPerLayer = 10
    patterns = ap.layer_patterns(layerId, X)
    # Show a sample subset from each pattern 
    print(nap.sample(patterns))
    pattern_samples = nap.head(patterns, nSamplesPerLayer) 
    titles = []
    patterns = []
    images = list(X.unbatch().as_numpy_iterator())

    for pattern_id, pattern in pattern_samples.groupby('patternId'):
        patterns.append(pattern.index)
        titles.append(F"Pattern: {pattern_id}, size: {len(pattern)}")
    #show_images(images, y, patterns, titles)

    # Show pattern representatives for layer  
    sorted_patterns = nap.sort(ap.layer_patterns(layerId, X))
    
    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        avg = nap.average(X, pattern.index)
        centers = pattern.head(1).index
        outliers = pattern.tail(3).index
        show_pattern(avg, centers, outliers, images, y, F"Layer {layerId}, Pattern: {pattern_id}, Size: {len(pattern)}")


#layer_analysis()
#filter_analysis()

export.export_all(ap, X, y, "MNIST", ["conv2d", "conv2d_1", "dense"])

