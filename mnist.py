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
    nRows =  max(1, int(sampleSize / images_per_row))
    img = np.array([images[idx].reshape(imSize) for idx in img_idx])
    fig = px.imshow(img, facet_col=0, binary_string=True, facet_col_wrap=nCols, 
                    #width=int(nCols*imSize[0]*img_scale), 
                    # height=220+int(nRows*80 + nRows*imSize[1]*img_scale), 
                    title=title).update_layout(margin=dict(l=5, b=0))
    # Set facet titles
    for i, im in enumerate(img_idx):
        fig.layout.annotations[i]['text'] = f"{labels[im]}"
    fig.show()

def show_cluster(average, representatives, outliers, images, labels, title):
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

def show_images(images, labels, layer_img_idx, titles, images_per_row = 10, img_scale = 2.0):
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
#model.summary()        

nap = nap.NeuralActivationPattern(x_test[:100], y_test_l[:100], model)
#print(nap.clusters.columns.levels)
# print(nap.get_max_activations(0))
#for layerId, layer in enumerate(model.layers):
# for layerId, df in nap.clusters.groupby(level=0, axis=1):
#     for cluster in df[layerId].groupby('clusterId'):
#         cluster.sort_values(by='probability', ascending=False)
#         print(len(cluster))

#nap.summary(0).show()
layerId = 0
nSamplesPerLayer = 10
# Show a sample subset from each cluster 
#print(nap.sample(layerId))
cluster_samples = nap.head(layerId, nSamplesPerLayer) 
titles = []
clusters = []
for cluster_id, cluster in cluster_samples.groupby('clusterId'):
    clusters.append(cluster.index)
    titles.append(F"Cluster: {cluster_id}, size: {len(cluster)}")
# for cluster_id, cluster_size in zip(cluster_ids, cluster_sizes):
#     titles.append(F"Cluster: {cluster_id}, size: {cluster_size}")
show_images(x_test, y_test_l, clusters, titles)

# Show cluster representatives for each layer  
#for layerId, layer in enumerate(model.layers):
# cluster_representatives, clusterIds, cluster_sizes = nap.get_cluster_representatives(layerId)
# cluster_labels = [f"Cluster {clusterId}" for clusterId in clusterIds]
# nClusters = len(cluster_representatives)
# indices = list(range(nClusters))
#show_images(cluster_representatives, cluster_labels, [indices], [f"Layer {layerId}, Clusters: {nClusters}"], img_scale=0.1)

for layerId, layer in enumerate(model.layers):
    cluster_representatives = []
    cluster_outliers = []
    cluster_labels = [f"Cluster {clusterId}" for clusterId in nap.get_clusters(layerId)['clusterId']]
    nClusters = len(nap.get_clusters(layerId))
    titles = []
    for clusterId, cluster in nap.get_clusters(layerId):
        cluster_sorted = cluster.sort_values(by='probability', ascending=False)
        print(cluster_sorted)
        # Most/least likely to belong to belong to the cluster (relative to its center)
        cluster_representative = cluster_sorted.head(2).index.values
        cluster_representatives.append(cluster_representative)
        cluster_outliers.append(cluster_sorted.tail(2).index.values)
        cluster_size = len(cluster)
        titles.append(F"Cluster: {clusterId}, size: {cluster_size}")

    # cluster_representatives = [[cluster[0], cluster[-1]] for cluster in clusters]
    # cluster_labels = [f"Cluster {clusterId}" for clusterId in clusterIds]
    # nClusters = len(clusters)
    # titles = [F"Cluster: {cluster_id}, size: {cluster_size}" for cluster_id, cluster_size in zip(clusterIds, cluster_sizes)]
    #print(nap.get_clusters(layerId))
    show_images(x_test, y_test_l, cluster_representatives, titles, img_scale=0.1)
    