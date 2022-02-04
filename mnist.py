import nap
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

def show_image_grid(images, labels, img_idx, title, images_per_row, img_scale):
    import plotly.express as px
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
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train_l), (x_test, y_test_l) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train_l, num_classes)
y_test = keras.utils.to_categorical(y_test_l, num_classes)


model_save_name = 'mnist_classifier'
path = F"{model_save_name}" 
#model.save(path)
model = keras.models.load_model(path)
#model.save_w(f"{model_save_name}")
#model.layer_summary()        

ap = nap.NeuralActivationPattern(x_test[:100], y_test_l[:100], model)
ap.layer_summary(0)
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
    sorted_patterns = nap.sort(ap.filter_patterns(layerId))
    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        avg = ap.average(pattern.index)
        centers = pattern.head(1).index
        outliers = pattern.tail(3).index
        show_pattern(avg, centers, outliers, x_test, y_test_l, F"Layer {layerId}, Pattern: {pattern_id}, Size: {len(pattern)}")


layer_analysis()