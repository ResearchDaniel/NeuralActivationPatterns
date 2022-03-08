import numpy as np
import plotly.express as px
import pandas as pd
from . import AggregationInterface


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


def sort(patterns):
    """ Returns patterns sorted according to probability of belonging to a pattern
    """
    return patterns.sort_values(['patternId', 'probability'], ascending=False)


def sample(patterns, frac=0.1):
    """ Samples according to the probability of belonging to a pattern
        Returns: DataFrame
    """
    # Pandas cannot sample with probabilities summing to 0, so replace those
    df = patterns.copy()
    grp = df.groupby('patternId')
    df['probability'] = grp['probability'].transform(
        lambda x: x if x.mean() > 0 else 0.1)
    return grp.sample(frac=frac, weights=df.probability)


def head(patterns, n):
    """ Returns, from each pattern, the n items most likey to belong to the pattern.
    """
    return sort(patterns).groupby('patternId').head(n)


def tail(patterns, n):
    """ Returns, from each pattern, the n items (outliers) least likey to belong to the pattern.

    """
    return self.sorted(patterns).groupby('patternId').tail(n)


def outliers(patterns, quantile=0.95, n=None):
    """ Returns indices of possible outliers, i.e., the ones with highest outlier score (GLOSH algorithm).
    """
    if n is not None:
        return patterns.nlargest(n, 'outlier_score').index
    else:
        threshold = patterns.outlier_score.quantile(quantile)
        outliers = np.where(patterns.outlier_score > threshold)[0]
        return outliers


def average(X, indices):
    img = np.zeros(X[0].shape, float)
    for c in indices:
        img = img + np.array(X[c]) / len(indices)
    return img


def layer_activation_aggregation_shape(activation_shape, agg_func=np.mean):
    """ Aggregate activations in a layer. 
        Convolutional layers are aggregated per feature. 

        Returns a list of arrays with sizes according to the number of features in each layer.    
    """
    if agg_func is None:
        return np.prod(activation_shape)
    else:
        if (len(activation_shape) == 3):
            # Convolutional layer
            return activation_shape[-1]
        else:
            return np.prod(activation_shape)


def layer_activation_aggregation(activations, agg_func=np.mean):
    """ Aggregate activations in a layer. 
        Convolutional layers are aggregated per feature.  
    """
    if agg_func is None:
        return activations.flatten()

    if (len(activations.shape) == 3):
        # Convolutional-like layer
        return [agg_func(activations[:, :, feature].flatten()) for feature in range(activations.shape[-1])]
    else:

        # aggregated_activation = [agg_func(activations.flatten())]
        # aggregated_activations.append(aggregated_activation)
        return activations.flatten()


def layer_activations_aggregation(layer_activations, agg_func=np.mean):
    """ Aggregate activations in a layer. 
        Convolutional layers are aggregated per feature. 

        Returns a list of arrays with sizes according to the number of features in each layer.    
    """
    if not agg_func:
        return [activations.flatten() for activations in layer_activations]
    layer_activations_aggregated = []
    for activations in layer_activations:
        layer_activations_aggregated.append(
            layer_activation_aggregation(activations, agg_func))

    return layer_activations_aggregated


class NeuralActivationPattern:
    """ Computes neural network activation patterns using clustering.
    """

    def __init__(self, model, agg_func=np.mean):
        self.model = model
        self.agg_func = agg_func

    def layerIdx(self, layer):
        """ Get layer index given either its layer name or its index
            Returns:
                int: Layer index. 
        """
        if isinstance(layer, int) or isinstance(layer, float):
            return int(layer)
        else:
            layer_name = layer
            layer_names = [layer.name for layer in self.model.layers]
            return layer_names.index(layer_name)

    def layer(self, layer):
        return self.model.layers[self.layerIdx(layer)]

    def activity_patterns(self, path, X=None, activations=None):
        """ Get activity patterns for a layer, or a filter within a layer
            Returns:
                DataFrame: Columns [patternId, probability] and index according to input id. 
        """
        layer_filter = path.split(":")
        if len(layer_filter) < 1 or len(layer_filter) > 2:
            raise ValueError(
                f"Expected path format layer_name:filter_number, got {path}")
        if layer_filter[0].isdigit():
            layerId = int(layer_filter[0])
        else:
            layer_name = layer_filter[0]
            layer_names = [layer.name for layer in self.model.layers]
            layerId = layer_names.index(layer_name)
        if len(layer_filter) == 1:
            return self.layer_patterns(layerId, X, activations)
        elif len(layer_filter) == 2:
            filterId = int(layer_filter[1])
            return self.filter_patterns(layerId, filterId, X, activations)

    def layer_max_activations(self, layer, X=None, activations=None, nSamplesPerLayer=10, agg_func=np.max):
        """ Get indices to the inputs with the highest aggregated activation within specified layer.

        Returns:
            list: Indices to input data with the highest activations.
        """
        layerId = self.layerIdx(layer)
        if activations is None:
            activations = self.layer_activations(layer, X)
        # for each input
        agg_activations = [agg_func(activation[:])
                           for activation in activations]
        # Get indices of images that activate the most
        largestActivationsIndices = list(
            reversed(np.argsort(agg_activations)[-nSamplesPerLayer:]))

        return largestActivationsIndices

    def filter_max_activations(self, layer, filterId, X=None, activations=None, nSamplesPerLayer=10, agg_func=np.max):
        """ Get indices to the inputs with the highest aggregated activation within specified layer and filter.
            Only works for convolutional layers.
        Returns:
            list: Indices to input data with the highest activations.
        """
        if activations is None:
            activations = self.layer_activations(layer, X)
        # for each finput
        agg_activations = [agg_func(activation[:, :, filterId])
                           for activation in activations]
        # Get indices of images that activate the most
        largestActivationsIndices = list(
            reversed(np.argsort(agg_activations)[-nSamplesPerLayer:]))

        return largestActivationsIndices

    def activation_model(self, layer):
        from tensorflow import keras
        layerId = layerIdx(layer)
        return keras.models.Model(inputs=model.input, outputs=model.layers[layerId])

    def layer_output_shape(self, layer):
        layerIdx = self.layerIdx(layer)
        return self.model.layers[layerIdx].output.shape

    def layer_activations(self, layer, X):
        layerIdx = self.layerIdx(layer)
        layer_output = self.model.layers[layerIdx].output
        # Creates a model that will return these outputs, given the model input
        from tensorflow import keras
        activation_model = keras.models.Model(
            inputs=self.model.input, outputs=layer_output)

        layer_activations = activation_model.predict(X)
        # Print info about the activations
        #print(pd.DataFrame({"Layer": self.model.layers[layerIdx].name, 'Activation shape': [activation.shape for activation in layer_activations]}))
        return layer_activations

    def layer_patterns(self, layer, X=None, agg_activations=None):
        if not agg_activations:
            activations = self.layer_activations(layer, X)
            agg_activations = self.agg_func.aggregate(
                self.layer(layer), activations)
        import hdbscan
        clusterer = hdbscan.HDBSCAN(cluster_selection_method='leaf')
        clusterer.fit(agg_activations)
        print(
            F"Layer {layer}, number of patterns: {clusterer.labels_.max() + 1}")
        patterns = pd.DataFrame({"patternId": clusterer.labels_,
                                "probability": clusterer.probabilities_, "outlier_score": clusterer.outlier_scores_})
        pattern_info = pd.DataFrame(
            {"pattern_persistence": clusterer.cluster_persistence_})
        return patterns, pattern_info

    def filter_patterns(self, layer, filterId, X=None, activations=None):
        import hdbscan
        if activations is None:
            activations = self.layer_activations(layer, X)

        filter_activations = [np.ndarray.reshape(
            activation[..., filterId], -1) for activation in activations]
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(filter_activations)
        print(
            F"Layer {layer}, filter: {filterId}, number of patterns: {clusterer.labels_.max() + 1}")
        patterns = pd.DataFrame({"patternId": clusterer.labels_,
                                "probability": clusterer.probabilities_, "outlier_score": clusterer.outlier_scores_})
        pattern_info = pd.DataFrame(
            {"pattern_persistence": clusterer.cluster_persistence_})
        return patterns, pattern_info

    def layer_summary(self, layer, X, y, layer_patterns=None):
        if layer_patterns is None:
            layer_patterns = self.layer_patterns(layer, X)
        layer_patterns = pd.concat(
            [layer_patterns, pd.DataFrame({'label': y})], axis=1)
        # Count the number of labels falling into each pattern
        patterns = layer_patterns.groupby(['patternId', 'label']).agg(
            counts=pd.NamedAgg(column='label', aggfunc='count')).reset_index()

        patterns_grp = patterns.groupby('patternId')
        # Max number of labels within each pattern
        max_pattern_labels = patterns_grp['counts'].transform('max')
        # Normalized count of labels within each pattern
        patterns['counts_norm'] = patterns['counts'] / max_pattern_labels

        patterns_sorted = np.sort(layer_patterns['patternId'].unique())[::-1]
        labels_sorted = np.sort(layer_patterns['label'].unique())
        # size = max(220 + labels.shape[0]*20,
        #            220 + patterns.shape[0]*20)
        fig = px.scatter(patterns, x='label', y='patternId', size='counts', symbol_sequence=['square'],
                         category_orders={"patternId": patterns_sorted,
                                          "label": labels_sorted},
                         title=f'Layer {layer}. Number of labels belonging to each pattern',
                         size_max=12,
                         #width= size,
                         #height= size,
                         template="plotly_white"
                         )
        fig.update_xaxes(tickson="boundaries", type='category', showline=True, mirror=True,
                         # Make a square plot that fits to the data
                         # sets the range of xaxis
                         range=[-0.5, len(labels_sorted) + 0.5],
                         constrain="domain",  # compresses the xaxis by decreasing its "domain"
                         )
        fig.update_yaxes(tickson="boundaries", type='category',
                         showline=True, mirror=True, scaleanchor="x", scaleratio=1)
        # fig.show()
        return fig
        # HDBSCAN is noise aware – it has a notion of data samples that are not assigned to any pattern. This is handled by assigning these samples the label -1.
        # px.histogram(dd, x='label', facet_row='pattern', title= 'Layer {}: {}'.format(patternOnLayer, model.layers[patternOnLayer].name),
        #height = 2000,
        # facet_row_spacing=0.005000).update_yaxes(matches=None)
