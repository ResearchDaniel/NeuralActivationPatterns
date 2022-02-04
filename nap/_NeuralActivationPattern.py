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


def sort(patterns):
    return patterns.sort_values(['patternId', 'probability'], ascending=False)

def layer_activation_aggregation(layer_activations, agg_func = np.mean):
    """ Aggregate activations in a layer. 
        Convolutional layers are aggregated per feature. 

        Returns a list of arrays with sizes according to the number of features in each layer.    
    """ 
    layer_activations_aggregated = []
    for activations in layer_activations: 
        if (len(activations.shape) == 3): 
            # Convolutional layer
            aggregated_activation = [agg_func(activations[:, :,feature].flatten()) for feature in range(activations.shape[-1])]
            layer_activations_aggregated.append(aggregated_activation)
        else:
            # aggregated_activation = [agg_func(activations.flatten())]
            # aggregated_activations.append(aggregated_activation)
            layer_activations_aggregated.append(activations.flatten())
    return layer_activations_aggregated    

class NeuralActivationPattern:

    def __init__(self, X, y, model, agg_func = np.mean):
        self.X = X
        self.y = y
        self.model = model
        self.agg_func = agg_func
        layers = model.layers[:]
        # Extracts the outputs of the layers
        layer_outputs = [layer.output for layer in layers] 
        # Creates a model that will return these outputs, given the model input
        from tensorflow import keras
        self.activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs) 

    def activity_patterns(self, path):
        """ Get activity patterns for a layer, or a filter within a layer
            Returns:
                DataFrame: Columns [patternId, probability] and index according to input id. 
        """
        layer_filter = path.split(":")
        if len(layer_filter) < 1 or len(layer_filter) > 2:
            raise ValueError(f"Expected path format layer_name:filter_number, got {path}")
        if layer_filter[0].isdigit():
            layerId = int(layer_filter[0])
        else:
            layer_name = layer_filter[0]
            layer_names = [layer.name for layer in self.model.layers]
            layerId = layer_names.index(layer_name)
        if len(layer_filter) == 1:
            return self.layer_patterns(layerId)
        elif len(layer_filter) == 2:
            filterId = int(layer_filter[1])
            return self.filter_patterns(layerId, filterId)

    def layer_max_activations(self, layerId, nSamplesPerLayer = 10, agg_func = np.max):
        """ Get indices to the inputs with the highest aggregated activation within specified layer.

        Returns:
            list: Indices to input data with the highest activations.
        """        
        activations = self.layer_activations[layerId]
        # for each input
        agg_activations = [agg_func(activation[:]) for activation in activations]
        # Get indices of images that activate the most
        largestActivationsIndices = list(reversed(np.argsort(agg_activations)[-nSamplesPerLayer:]))

        return largestActivationsIndices
    def filter_max_activations(self, layerId, filterId, nSamplesPerLayer = 10, agg_func = np.max):
        """ Get indices to the inputs with the highest aggregated activation within specified layer and filter.
            Only works for convolutional layers.
        Returns:
            list: Indices to input data with the highest activations.
        """        
        activations = self.layer_activations[layerId]
        # for each finput
        agg_activations = [agg_func(activation[:,:,filterId]) for activation in activations]
        # Get indices of images that activate the most
        largestActivationsIndices = list(reversed(np.argsort(agg_activations)[-nSamplesPerLayer:]))

        return largestActivationsIndices


    def sorted(self, layerId):
        """ Returns patterns sorted according to probability of belonging to a pattern
        """
        return self.patterns[layerId].sort_values(['patternId', 'probability'], ascending=False)

    def head(self, layerId, n):
            """ Returns, from each pattern, the n items most likey to belong to the pattern.
            """
            return self.sorted(layerId).groupby('patternId').head(n)
    def tail(self, layerId, n):
            """ Returns, from each pattern, the n items (outliers) least likey to belong to the pattern.
                
            """
            return self.sorted(layerId).groupby('patternId').tail(n)
    def sample(self, layerId, frac = 0.1):
        """ Samples according to the probability of belonging to a pattern
         Returns:
            DataFrame: 
        """
        # Pandas cannot sample with probabilities summing to 0, so replace those
        df = self.patterns[layerId].copy()
        grp = df.groupby('patternId')
        df['probability'] = grp['probability'].transform(lambda x: x if x.mean() > 0 else 0.1 )

        return grp.sample(frac=frac,weights=df.probability)

    def average(self, indices):
        img = np.zeros(self.X[0].shape, float)
        for c in indices:
            img = img + np.array(self.X[c]) / len(indices)
        return img  
        

    @lazyproperty  
    def layer_activations(self):
        layer_activations = self.activation_model.predict(self.X)
        # Print info about the activations
        print(pd.DataFrame({"Layer": [layer.name for layer in self.model.layers], 'Activation shape': [activation.shape for activation in layer_activations]}))
        return layer_activations

    def layer_patterns(self, layerId):
        return self.patterns[layerId]

    @lazyproperty  
    def patterns(self):
        import hdbscan
        layer_patterns = [] 
        data = {}

        layerIds = []
        for layerId, activations in enumerate(self.layer_activations):
            agg_activations = layer_activation_aggregation(activations, self.agg_func)
            clusterer = hdbscan.HDBSCAN()
            clusterer.fit(agg_activations)
            print(F"Layer {layerId}, number of patterns: {clusterer.labels_.max() + 1}")
            layerIds.append(layerId)
            #data[(layer, "input_index")] = list(range(len(clusterer.labels_)))
            data[(layerId, "patternId")] = clusterer.labels_
            data[(layerId, "probability")] = clusterer.probabilities_
 
        return pd.DataFrame(data.values(), index=pd.MultiIndex.from_tuples(data.keys(), names=['layerId', 'input_index'])).transpose()

    def filter_patterns(self, layerId, filterId):
        import hdbscan
        activations = self.layer_activations[layerId]
        filter_activations = [activation[:,:,filterId].flatten() for activation in activations]
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(filter_activations)
        print(F"Layer {layerId}, filter: {filterId}, number of patterns: {clusterer.labels_.max() + 1}")
 
        return pd.DataFrame({"patternId":clusterer.labels_, "probability": clusterer.probabilities_})
    
    def layer_summary(self, layer):
        layer_patterns = pd.concat([self.patterns[layer], pd.DataFrame({'label': self.y})], axis=1)
        # Count the number of labels falling into each pattern
        patterns = layer_patterns.groupby(['patternId', 'label']).agg(counts=pd.NamedAgg(column='label', aggfunc='count')).reset_index()
        
        patterns_grp = patterns.groupby('patternId')
        # Max number of labels within each pattern
        max_pattern_labels = patterns_grp['counts'].transform('max')
        # Normalized count of labels within each pattern
        patterns['counts_norm'] = patterns['counts'] / max_pattern_labels

        patterns_sorted = np.sort(layer_patterns['patternId'].unique())[::-1]
        labels_sorted = np.sort(layer_patterns['label'].unique())
        # size = max(220 + labels.shape[0]*20,
        #            220 + patterns.shape[0]*20)
        fig = px.scatter(patterns, x='label', y = 'patternId', size='counts', symbol_sequence=['square'], 
            category_orders={"patternId": patterns_sorted,
                                "label": labels_sorted},
            title='Number of labels belonging to each pattern',
            size_max=12,
            #width= size,
            #height= size,
            template="plotly_white"
            )
        fig.update_xaxes(tickson="boundaries", type='category', showline=True, mirror=True,
            # Make a square plot that fits to the data
            range=[-0.5, len(labels_sorted) + 0.5],  # sets the range of xaxis
            constrain="domain",  # compresses the xaxis by decreasing its "domain"
        )
        fig.update_yaxes(tickson="boundaries", type='category', showline=True, mirror=True, scaleanchor = "x", scaleratio = 1)
        #fig.show()
        return fig
        # HDBSCAN is noise aware – it has a notion of data samples that are not assigned to any pattern. This is handled by assigning these samples the label -1. 
        #px.histogram(dd, x='label', facet_row='pattern', title= 'Layer {}: {}'.format(patternOnLayer, model.layers[patternOnLayer].name),
        #height = 2000,
        #facet_row_spacing=0.005000).update_yaxes(matches=None)


