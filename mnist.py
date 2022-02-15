import nap
import numpy as np
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
    ds, info = tfds.load('mnist', split='test', shuffle_files=False, batch_size=-1, with_info=True)
    print(info.features['label'].num_classes)
    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape
    x_test = tfds.as_numpy(ds['image']).astype("float32")  / 255
    y_test_l = tfds.as_numpy(ds['label'])

    model_save_name = 'mnist_classifier'
    path = F"{model_save_name}" 
    #model.save(path)
    model = keras.models.load_model(path)
    #model.save_w(f"{model_save_name}")
    #model.layer_summary()  
    return model, x_test, y_test_l

def setupInceptionV3():
    import tensorflow_datasets as tfds
    import tensorflow as tf
    ds, info = tfds.load('imagenet_v2', split='test', shuffle_files=False, with_info=True)
    print(info.features)
    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape
    ds = ds.take(2)
    x_test = np.asarray([tf.keras.applications.inception_v3.preprocess_input(tf.keras.layers.Resizing(299, 299)(v['image'])) for v in ds])
    y_test_l = np.asarray([tfds.as_numpy(v['label']) for v in ds])
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    print(model.summary())
    return model, x_test, y_test_l

model, x_test, y_test_l = setupMNIST()
#model, x_test, y_test_l = setupInceptionV3()
      

ap = nap.NeuralActivationPattern(x_test[:100], y_test_l[:100], model)
ap.layer_summary("conv2d")
def filter_analysis():
    # For now, simply test that these functions works
    layerId = 5
    filterId = 0
    filter_patterns = ap.activity_patterns("conv2d:0")
    filter_patterns = ap.activity_patterns("0:0")
    filter_patterns = nap.sort(ap.filter_patterns(layerId, filterId))
    # Show pattern representatives for filter  
    sorted_patterns = nap.sort(ap.filter_patterns(layerId, filterId))

    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        avg = ap.average(pattern.index)
        centers = pattern.head(1).index
        outliers = pattern.tail(3).index
        show_pattern(avg, centers, outliers, x_test, y_test_l, F"Layer {layerId}, Filter: {filterId}, Pattern: {pattern_id}, Size: {len(pattern)}")


def layer_analysis():
    ap.layer_summary(5).show()
    print(ap.layer_max_activations(0))
    
    layerId = 5
    nSamplesPerLayer = 10
    # Show a sample subset from each pattern 
    print(ap.sample(layerId))
    pattern_samples = ap.head(layerId, nSamplesPerLayer) 
    titles = []
    patterns = []
    for pattern_id, pattern in pattern_samples.groupby('patternId'):
        patterns.append(pattern.index)
        titles.append(F"Pattern: {pattern_id}, size: {len(pattern)}")
    #show_images(x_test, y_test_l, patterns, titles)

    # Show pattern representatives for layer  
    sorted_patterns = nap.sort(ap.layer_patterns(layerId))
    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        avg = ap.average(pattern.index)
        centers = pattern.head(1).index
        outliers = pattern.tail(3).index
        show_pattern(avg, centers, outliers, x_test, y_test_l, F"Layer {layerId}, Pattern: {pattern_id}, Size: {len(pattern)}")


layer_analysis()