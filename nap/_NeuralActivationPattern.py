def lazyproperty(func):
    name = '_lazy_' + func.__name__
    @property
    def lazy(self):
        if hasattr(self, name):
            return getattr(self, name)
        value = func(self)
        setattr(self, name, value)
        return value

    return lazy

import numpy as np
import pandas as pd
import plotly.express as px
class NeuralActivationPattern:

    def __init__(self, X, y, model, agg_func = np.max):
        self.X = X
        self.y = y
        self.model = model
        layers = model.layers[:]
        # Extracts the outputs of the layers
        layer_outputs = [layer.output for layer in layers] 
        # Creates a model that will return these outputs, given the model input
        from keras import models    
        self.activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 

    def get_max_activations(self, layerId, nSamplesPerLayer = 10, agg_func = np.max):
        """ Get indices to the inputs with the highest aggregated activation within specified layer.

        Returns:
            list: One list of indices to input data with highest activation.
        """        
        # for each layer
        activations = self.layer_activations[layerId]
        # for each input
        agg_activations = [agg_func(activation[:]) for activation in activations]
        # Get indices of images that activate the most
        largestActivationsIndices = list(reversed(np.argsort(agg_activations)[-nSamplesPerLayer:]))

        return largestActivationsIndices

    def get_cluster_activations(self, layerId, maxSamplesPerLayer = 10):
        cluster_sample_indices = []
        layer_clusters = self.clusters[layerId]
        cluster_ids = []
        cluster_sizes = []
        for clusterId, cluster in layer_clusters.groupby('clusterId'):
            cluster_size = cluster.size
            sampleSize = min(cluster_size, maxSamplesPerLayer)
            image_id_samples = cluster.sample(sampleSize).index
            cluster_sample_indices.append(image_id_samples)
            cluster_ids.append(clusterId)
            cluster_sizes.append(cluster_size)

        return cluster_sample_indices, cluster_ids, cluster_sizes

    def generate_representative(self, indices):
        img = np.zeros(self.X[0].shape, float)
        for c in indices:
            img = img + np.array(self.X[c]) / len(indices)
        return img    
    def get_cluster_representatives(self, layer):
        
        layer_clusters = self.clusters[layer]
        cluster_ids = []
        cluster_sizes = []
        cluster_representatives = []

        for clusterId, cluster in layer_clusters.groupby('clusterId'):
            cluster_representatives.append(self.generate_representative(cluster.index))
            cluster_ids.append(clusterId)
            cluster_sizes.append(cluster.size)
        return cluster_representatives, cluster_ids, cluster_sizes       
    def activation_aggregation(self, agg_func = np.mean):
        """ Aggregate activations for each layer. 
            Convolutional layers are aggregated per feature. 

            Returns a list of arrays with sizes according to the number of features in each layer.    
        """ 
        aggregated_activations = []
        # for each layer
        for layer_activation in self.layer_activations:
            layer_activations_aggregated = []
            for activations in layer_activation: 
                if (len(activations.shape) == 3): 
                    # Convolutional layer
                    aggregated_activation = [agg_func(activations[:, :,feature].flatten()) for feature in range(activations.shape[-1])]
                    layer_activations_aggregated.append(aggregated_activation)
                else:
                    # aggregated_activation = [agg_func(activations.flatten())]
                    # aggregated_activations.append(aggregated_activation)
                    layer_activations_aggregated.append(activations.flatten())
            aggregated_activations.append(layer_activations_aggregated)
        return aggregated_activations
    @lazyproperty  
    def layer_activations(self):
        layer_activations = self.activation_model.predict(self.X)
        # Print info about the activations
        print(pd.DataFrame({"Layer": [layer.name for layer in self.model.layers], 'Activation shape': [activation.shape for activation in layer_activations]}))
        return layer_activations

    @lazyproperty  
    def activations_agg(self):
        return self.activation_aggregation(agg_func = np.average)

    @lazyproperty  
    def clusters(self):
        import hdbscan
        layer_clusters = []
        for layer, activations in enumerate(self.activations_agg):
            clusterer = hdbscan.HDBSCAN()
            clusterer.fit(activations)
            print(F"Layer {layer}, number of clusters: {clusterer.labels_.max() + 1}")
            clusters = pd.DataFrame({"clusterId": clusterer.labels_})
            layer_clusters.append(clusters)
        return layer_clusters

    def summary(self, layer):
        layer_clusters = self.clusters[layer]
        dd = pd.DataFrame({'cluster': layer_clusters.clusterId, 'label': self.y})
        # Count the number of labels falling into each cluster
        dd_agg = dd.groupby(['cluster', 'label']).agg(counts=pd.NamedAgg(column='label', aggfunc='count')).reset_index()
        # Normalized count over all labels 
        dd_agg['counts_norm'] = dd_agg['counts'] / dd_agg['counts'].max()
        # Normalized number of clusters within each label
        dd_agg['counts_within_label_norm'] = dd_agg['counts'] / dd_agg.groupby('label')['counts'].transform('max')
        #print(dd_agg)

        fig = px.scatter(dd_agg, x='label', y = 'cluster', size='counts_within_label_norm', symbol_sequence=['square'], 
            title='Number of labels belonging to each cluster - size normalized per layer',
            height= 220 + dd['cluster'].unique().shape[0]*20).update_xaxes(type='category').update_yaxes(type='category')

        #fig.show()
        return fig
        # HDBSCAN is noise aware – it has a notion of data samples that are not assigned to any cluster. This is handled by assigning these samples the label -1. 
        #px.histogram(dd, x='label', facet_row='cluster', title= 'Layer {}: {}'.format(clusterOnLayer, model.layers[clusterOnLayer].name),
        #height = 2000,
        #facet_row_spacing=0.005000).update_yaxes(matches=None)


