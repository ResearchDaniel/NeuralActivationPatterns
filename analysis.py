import nap
import numpy as np
import util
import tensorflow as tf

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
    nRows = int(sampleSize / nCols)
    img = np.array([image.reshape(imSize) for image in images])
    fig = px.imshow(img, facet_col=0, binary_string=True, facet_col_wrap=nCols,
                    # width=int(nCols*imSize[0]*img_scale),
                    # height=50+int(nRows*10 + nRows*imSize[1]*img_scale),
                    # facet_row_spacing = min(1/(nRows - 1), )
                    title=title
                    )
    # Set facet titles
    for i, im in enumerate(labels):
        fig.layout.annotations[plotly_annotation_index(
            nRows, nCols, i)]['text'] = im
    # fig.update_yaxes(automargin=False, title=title)
    # fig.update_xaxes(automargin=True)
    fig.update_layout(margin=dict(b=0))
    # fig.update_annotations(standoff=10)
    fig.show()


def show_pattern(average, representatives, representativeLabels, outliers, outlierLabels, title):
    import plotly.express as px
    if representatives[0].shape[2] > 1:
        imSize = (representatives[0].shape[0],
                  representatives[0].shape[1], representatives[0].shape[2])
    else:
        imSize = (representatives[0].shape[0], representatives[0].shape[1])
    images = np.array([average.reshape(imSize)] + [rep.reshape(imSize)
                      for rep in representatives] + [otl.reshape(imSize) for otl in outliers])
    sampleSize = len(images)
    images_per_row = 10
    nCols = min(sampleSize, images_per_row)
    nRows = int(sampleSize / nCols)
    fig = px.imshow(images, facet_col=0, binary_string=True, title=title)
    fig.layout.annotations[0]['text'] = "Average"
    for i, _ in enumerate(representatives):
        idx = plotly_annotation_index(nRows, nCols, 1+i)
        fig.layout.annotations[idx]['text'] = representativeLabels[i]
    for i, _ in enumerate(outliers):
        idx = plotly_annotation_index(nRows, nCols, len(representatives)+1+i)
        fig.layout.annotations[idx]['text'] = outlierLabels[i]
    fig.show()


def show_images(images, labels, title, images_per_row=10, img_scale=7.0):
    show_image_grid(images, labels, title, images_per_row, img_scale)


def show_outliers(model_name, X, y, layer, patterns, quantile=0.95, n=None):
    outliers = nap.outliers(patterns, quantile, n)
    images = util.filter_tf_dataset(X, outliers)
    labels = [f"{y[i]} | id:{i}" for i in outliers]
    title = F"{model_name}, layer: {layer}, outliers"
    show_images(images, labels, title)

def filter_analysis(model, model_name, X, y, layer, filter):
    patterns = nap.cache.get_filter_patterns(
        X, model, model_name, layer, filter)
    # Show pattern representatives for filter
    sorted_patterns = nap.sort(patterns)

    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        if pattern_id == -1:
            continue
        to_average = util.filter_tf_dataset(X, pattern.index)
        avg = tf.keras.layers.Average()(to_average).numpy()
        centerIndices = pattern.head(1).index
        outliersIndices = pattern.tail(3).index
        centers = util.filter_tf_dataset(X, centerIndices)
        outliers = util.filter_tf_dataset(X, outliersIndices)
        centerLabels = [f"Representative | {y[i]}" for i in centerIndices]
        outlierLabels = [f"Outlier | {y[i]}" for i in outliersIndices]

        show_pattern(avg, centers, centerLabels, outliers, outlierLabels,
                     F"{model_name}, Layer {layer}, Filter: {filter}, Pattern: {pattern_id}, Size: {len(pattern)}")


def layer_analysis(model, model_name, X, y, layer):
    patterns = nap.cache.get_layer_patterns(X, model, model_name, layer)
    ap = nap.NeuralActivationPattern(model)
    # ap.layer_summary(layer, X, y, patterns).show()
    # activations = export.get_layer_activations(X, model, model_name, layer)
    # print(ap.layer_max_activations(layer, X, activations=activations))
    #

    show_outliers(model_name, X, y, layer, patterns, n=100)

    # Show a sample subset from each pattern
    nSamplesPerLayer = 10
    print(nap.sample(patterns))
    pattern_samples = nap.head(patterns, nSamplesPerLayer)
    # for pattern_id, pattern in pattern_samples.groupby('patternId'):
    #     if pattern_id == -1:
    #         continue
    #     pattern_indices = pattern.index
    #     images = util.filter_tf_dataset(X, pattern_indices)
    #     labels = [f"{y[i]} | id:{i}" for i in pattern_indices]
    #     title = F"Layer: {layer}, pattern: {pattern_id}, size: {len(pattern)}"
    #     show_images(images, labels, title)

    # Show pattern representatives for layer
    sorted_patterns = nap.sort(patterns)
    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        if pattern_id == -1:
            continue

        to_average = util.filter_tf_dataset(X, pattern.index)
        avg = tf.keras.layers.Average()(to_average).numpy()
        centerIndices = pattern.head(1).index
        outliersIndices = pattern.tail(3).index
        # outliersIndices = nap.outliers(pattern, n=3)
        centers = util.filter_tf_dataset(X, centerIndices)
        outliers = util.filter_tf_dataset(X, outliersIndices)

        centerLabels = [f"Representative | {y[i]}" for i in centerIndices]
        outlierLabels = [f"Outlier | {y[i]}" for i in outliersIndices]

        show_pattern(avg, centers, centerLabels, outliers, outlierLabels,
                     F"{model_name}, Layer {layer}, Pattern: {pattern_id}, Size: {len(pattern)}")

