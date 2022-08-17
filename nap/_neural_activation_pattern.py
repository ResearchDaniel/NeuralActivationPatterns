"""Main functionality providing neural activation patterns."""
import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow import keras

# pylint: disable=R0401
from . import MeanAggregation, NoAggregation


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
    data_frame = patterns.copy()
    grp = data_frame.groupby('patternId')
    data_frame['probability'] = grp['probability'].transform(
        lambda x: x if x.mean() > 0 else 0.1)
    return grp.sample(frac=frac, weights=data_frame.probability)


def head(patterns, num_samples):
    """ Returns, from each pattern, the n items most likey to belong to the pattern.
    """
    return sort(patterns).groupby('patternId').head(num_samples)


def tail(patterns, num_samples):
    """ Returns, from each pattern, the n items (outliers) least likey to belong to the pattern.

    """
    return sort(patterns).groupby('patternId').tail(num_samples)


def outliers(patterns, quantile=0.95, num_samples=None):
    """Returns indices of possible outliers, i.e.,
    the ones with highest outlier score (GLOSH algorithm).
    """
    if num_samples is not None:
        return patterns.nlargest(num_samples, 'outlier_score').index
    threshold = patterns.outlier_score.quantile(quantile)
    outlier_samples = np.where(patterns.outlier_score > threshold)[0]
    return outlier_samples


def average(samples, indices):
    img = np.zeros(samples[0].shape, float)
    for index in indices:
        img = img + np.array(samples[index]) / len(indices)
    return img


class NeuralActivationPattern:
    """ Computes neural network activation patterns using clustering.
    """

    def __init__(
            self, model, layer_aggregation=MeanAggregation, filter_aggregation=NoAggregation,
            min_pattern_size=5, min_samples=5, cluster_selection_epsilon=0,
            cluster_selection_method="leaf"):
        self.model = model
        self.layer_aggregation = layer_aggregation
        self.filter_aggregation = filter_aggregation
        self.min_pattern_size = min_pattern_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method

    def layer_idx(self, layer):
        """Get layer index given either its layer name or its index.

            Returns:
                int: Layer index.
        """
        if isinstance(layer, (float, int)):
            return int(layer)
        layer_name = layer
        layer_names = [layer.name for layer in self.model.layers]
        return layer_names.index(layer_name)

    def layer(self, layer):
        return self.model.layers[self.layer_idx(layer)]

    def activity_patterns(self, path, input_data=None, activations=None):
        """Get activity patterns for a layer, or a filter within a layer.

            Returns:
                DataFrame: Columns [patternId, probability] and index
                    according to input id.
        """
        layer_filter = path.split(":")
        if len(layer_filter) < 1 or len(layer_filter) > 2:
            raise ValueError(
                f"Expected path format layer_name:filter_number, got {path}")
        if layer_filter[0].isdigit():
            layer_id = int(layer_filter[0])
        else:
            layer_name = layer_filter[0]
            layer_names = [layer.name for layer in self.model.layers]
            layer_id = layer_names.index(layer_name)
        if len(layer_filter) == 1:
            return self.layer_patterns(layer_id, input_data, activations)
        filter_id = int(layer_filter[1])
        return self.filter_patterns(layer_id, filter_id, input_data, activations)

    def layer_max_activations(self, layer, input_data=None, activations=None, samples_per_layer=10,
                              agg_func=np.max):
        """Get indices to the inputs with the highest aggregated activation within specified layer.

        Returns:
            list: Indices to input data with the highest activations.
        """
        if activations is None:
            activations = self.layer_activations(layer, input_data)
        # for each input
        agg_activations = [agg_func(activation[:])
                           for activation in activations]
        # Get indices of images that activate the most
        largest_activations_indices = list(
            reversed(np.argsort(agg_activations)[-samples_per_layer:]))

        return largest_activations_indices

    def filter_max_activations(self, layer, filter_id, input_data=None, activations=None,
                               samples_per_layer=10, agg_func=np.max):
        """ Get indices to the inputs with the highest aggregated activation within specified
            layer and filter. Only works for convolutional layers.

        Returns:
            list: Indices to input data with the highest activations.
        """
        if activations is None:
            activations = self.layer_activations(layer, input_data)
        # for each input
        agg_activations = [agg_func(activation[..., filter_id])
                           for activation in activations]
        # Get indices of images that activate the most
        largest_activations_indices = list(
            reversed(np.argsort(agg_activations)[-samples_per_layer:]))

        return largest_activations_indices

    def activation_model(self, layer):
        layer_id = self.layer_idx(layer)
        return keras.models.Model(inputs=self.model.input, outputs=self.model.layers[layer_id])

    def layer_output_shape(self, layer):
        layer_idx = self.layer_idx(layer)
        return self.model.layers[layer_idx].output.shape

    def layer_num_units(self, layer):
        return self.layer_output_shape(layer)[-1]

    def layer_activations(self, layer, input_data):
        """Compute input data activations for a given layer. Equation 1 in the paper.
        """
        layer_idx = self.layer_idx(layer)
        layer_output = self.model.layers[layer_idx].output
        # Creates a model that will return these outputs, given the model input
        activation_model = keras.models.Model(
            inputs=self.model.input, outputs=layer_output)

        layer_activations = activation_model.predict(input_data)
        # Print info about the activations
        return layer_activations

    def layer_patterns(self, layer, input_data=None, agg_activations=None):
        """ Computes Neural Activation Patterns for the given layer. Equation 3 in the paper.
        """
        if agg_activations is None:
            activations = self.layer_activations(layer, input_data)
            agg_activations = self.layer_aggregation.aggregate(
                self.layer(layer), activations)

        clusterer = hdbscan.HDBSCAN(
            cluster_selection_method=self.cluster_selection_method,
            min_cluster_size=self.min_pattern_size,
            cluster_selection_epsilon=self.cluster_selection_epsilon, min_samples=self.min_samples)
        agg_2d = np.reshape(
            agg_activations,
            [agg_activations.shape[0],
             np.prod(agg_activations.shape[1:])])
        clusterer.fit(agg_2d)
        print(
            F"Layer {layer}, number of patterns: {clusterer.labels_.max() + 1}")
        patterns = pd.DataFrame({"patternId": clusterer.labels_,
                                 "probability": clusterer.probabilities_,
                                "outlier_score": clusterer.outlier_scores_})
        pattern_info = pd.DataFrame(
            {"pattern_persistence": clusterer.cluster_persistence_})
        return patterns, pattern_info

    def filter_patterns(self, layer, filter_id, input_data=None, activations=None):
        if activations is None:
            activations = self.layer_activations(layer, input_data)
        # Extract filter activations for each input
        filter_activations = activations[:, ..., filter_id]
        # Aggregate activations per input
        agg_activations = self.filter_aggregation.aggregate(self.layer(layer), filter_activations)
        agg_2d = np.reshape(
            agg_activations,
            [agg_activations.shape[0],
             np.prod(agg_activations.shape[1:])])
        clusterer = hdbscan.HDBSCAN(
            cluster_selection_method=self.cluster_selection_method,
            min_cluster_size=self.min_pattern_size,
            cluster_selection_epsilon=self.cluster_selection_epsilon, min_samples=self.min_samples)
        clusterer.fit(agg_2d)

        print(
            F"Layer {layer}, filter: {filter_id}, number of patterns: {clusterer.labels_.max()+1}")
        patterns = pd.DataFrame({"patternId": clusterer.labels_,
                                 "probability": clusterer.probabilities_,
                                "outlier_score": clusterer.outlier_scores_})
        pattern_info = pd.DataFrame(
            {"pattern_persistence": clusterer.cluster_persistence_})
        return patterns, pattern_info

    def layer_summary(self, layer, input_data, label, layer_patterns=None):
        if layer_patterns is None:
            layer_patterns = self.layer_patterns(layer, input_data)
        layer_patterns = pd.concat(
            [layer_patterns, pd.DataFrame({'label': label})], axis=1)
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
        fig = px.scatter(patterns, x='label', y='patternId', size='counts',
                         symbol_sequence=['square'], category_orders={"patternId":
                                                                      patterns_sorted,
                                                                      "label": labels_sorted},
                         title=f'Layer {layer}. Number of labels belonging to each pattern',
                         size_max=12,
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
        return fig
