"""Provide functionality for the analysis of NAPs."""
import numpy as np
import plotly.express as px
import tensorflow as tf

import nap
import util


def plotly_annotation_index(n_rows, n_cols, i):
    # The first row correspond to the last annotation in Plotly...
    row = n_rows - int(i / n_cols) - 1
    col = i % n_cols
    return row*n_cols + col


def show_image_grid(images, labels, title, images_per_row, img_scale):
    if images[0].shape[2] > 1:
        im_size = (images[0].shape[0], images[0].shape[1], images[0].shape[2])
    else:
        im_size = (images[0].shape[0], images[0].shape[1])
    # Get indices of the largest
    sample_size = len(images)
    n_cols = min(sample_size, images_per_row)
    n_rows = int(sample_size / n_cols)
    img = np.array([image.reshape(im_size) for image in images])
    fig = px.imshow(img, facet_col=0, binary_string=True, facet_col_wrap=n_cols,
                    width=int(n_cols*im_size[0]*img_scale),
                    height=50+int(n_rows*10 + n_rows*im_size[1]*img_scale),
                    title=title
                    )
    # Set facet titles
    for i, curr_im in enumerate(labels):
        fig.layout.annotations[plotly_annotation_index(
            n_rows, n_cols, i)]['text'] = curr_im
    fig.update_layout(margin=dict(b=0))
    fig.show()


def show_pattern(average, representatives, representative_labels, outliers, outlier_labels, title):
    if representatives[0].shape[2] > 1:
        im_size = (representatives[0].shape[0],
                   representatives[0].shape[1], representatives[0].shape[2])
    else:
        im_size = (representatives[0].shape[0], representatives[0].shape[1])
    images = np.array([average.reshape(im_size)] + [rep.reshape(im_size)
                      for rep in representatives] + [otl.reshape(im_size) for otl in outliers])
    sample_size = len(images)
    images_per_row = 10
    n_cols = min(sample_size, images_per_row)
    n_rows = int(sample_size / n_cols)
    fig = px.imshow(images, facet_col=0, binary_string=True, title=title)
    fig.layout.annotations[0]['text'] = "Average"
    for i, _ in enumerate(representatives):
        fig.layout.annotations[plotly_annotation_index(
            n_rows, n_cols, 1+i)]['text'] = representative_labels[i]
    for i, _ in enumerate(outliers):
        fig.layout.annotations[plotly_annotation_index(
            n_rows, n_cols, len(representatives)+1+i)]['text'] = outlier_labels[i]
    fig.show()


def show_images(images, labels, title, images_per_row=10, img_scale=7.0):
    show_image_grid(images, labels, title, images_per_row, img_scale)


def show_outliers(model_name, input_data, labels, layer, patterns, quantile=0.95, number=None):
    outliers = nap.outliers(patterns, quantile, number)
    images = util.filter_tf_dataset(input_data, outliers)
    labels = [f"{labels[i]} | id:{i}" for i in outliers]
    title = F"{model_name}, layer: {layer}, outliers"
    show_images(images, labels, title)


def filter_analysis(model, model_name, input_data, labels, layer, filter_index, min_pattern_size):
    # Show pattern representatives for filter
    sorted_patterns = nap.sort(nap.cache.get_filter_patterns(
        input_data, model, model_name, layer, filter_index, "mean", min_pattern_size))

    for pattern_id, pattern in sorted_patterns.groupby('patternId'):
        if pattern_id == -1:
            continue
        to_average = util.filter_tf_dataset(input_data, pattern.index)
        avg = tf.keras.layers.Average()(to_average).numpy()
        centers = util.filter_tf_dataset(input_data, pattern.head(1).index)
        outliers = util.filter_tf_dataset(input_data, pattern.tail(3).index)
        center_labels = [
            f"Representative | {labels[i]}" for i in pattern.head(1).index]
        outlier_labels = [
            f"Outlier | {labels[i]}" for i in pattern.tail(3).index]

        show_pattern(avg, centers, center_labels, outliers, outlier_labels,
                     (F"{model_name}, Layer {layer}, Filter: {filter}"
                      F", Pattern: {pattern_id}, Size: {len(pattern)}"))


def layer_analysis(
        model, model_name, input_data, labels, layer, aggregation=nap.MeanAggregation(),
        min_pattern_size=5):
    patterns, _ = nap.cache.get_layer_patterns(
        input_data, model, model_name, layer, aggregation, min_pattern_size)

    show_outliers(model_name, input_data, labels, layer, patterns, number=100)

    # Show a sample subset from each pattern
    print(nap.sample(patterns))

    # Show pattern representatives for layer
    for pattern_id, pattern in nap.sort(patterns).groupby('patternId'):
        if pattern_id == -1:
            continue

        avg = tf.keras.layers.Average()(
            util.filter_tf_dataset(input_data, pattern.index)).numpy()
        centers = util.filter_tf_dataset(input_data, pattern.head(1).index)
        outliers = util.filter_tf_dataset(input_data, pattern.tail(3).index)

        center_labels = [
            f"Representative | {labels[i]}" for i in pattern.head(1).index]
        outlier_labels = [
            f"Outlier | {labels[i]}" for i in pattern.tail(3).index]

        show_pattern(avg, centers, center_labels, outliers, outlier_labels,
                     F"{model_name}, Layer {layer}, Pattern: {pattern_id}, Size: {len(pattern)}")
