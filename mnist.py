import nap
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def show_images(images, labels, layer_img_idx, titles, images_per_row = 10, img_scale = 0.05):
    for layer, img_idx in enumerate(layer_img_idx):
        import matplotlib.pyplot as plt
        imSize = int(img_scale*images[0].shape[1])
        from mpl_toolkits.axes_grid1 import ImageGrid
        # Get indices of the largest 
        sampleSize = len(img_idx)
        nCols = min(sampleSize, images_per_row)
        nRows =  max(1, int(sampleSize / images_per_row))

        fig = plt.figure(figsize=(nCols*imSize, nRows*imSize))
        grid = ImageGrid(fig, 111, nrows_ncols=(nRows, nCols))
        for ax, im in zip(grid, img_idx):
            ax.imshow(images[im])
            ax.set_title(labels[im])
        print(titles[layer])
        plt.title(label = titles[layer])
        plt.grid(False)
        plt.show()




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
model.summary()        

nap = nap.NeuralActivationPattern(x_test[:1000], y_test_l[:1000], model)

print(nap.get_max_activations(0))

#nap.summary(0).show()
layerId = 0
nSamplesPerLayer = 10
# Show a sample subset from each cluster 
clusters, cluster_ids, cluster_sizes = nap.get_cluster_activations(layerId) 
# titles = []
# for cluster_id, cluster_size in zip(cluster_ids, cluster_sizes):
#     titles.append(F"Cluster: {cluster_id}, size: {cluster_size}")
#show_images(x_test, y_test_l, clusters, titles)

# Show cluster representatives for each layer  
#for layerId, layer in enumerate(model.layers):
cluster_representatives, clusterIds, cluster_sizes = nap.get_cluster_representatives(layerId)
cluster_labels = [f"Cluster {clusterId}" for clusterId in clusterIds]
nClusters = len(cluster_representatives)
indices = list(range(nClusters))
show_images(cluster_representatives, cluster_labels, [indices], [f"Layer {layerId}, Clusters: {nClusters}"], img_scale=0.1)