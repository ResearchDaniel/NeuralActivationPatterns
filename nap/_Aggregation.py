import numpy as np
from itertools import chain


class AggregationInterface:
    """Interface for NAP aggregation computers"""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        pass

    def aggregate(self, layer, activations) -> np.ndarray:
        """Aggregate layer activations"""
        pass


class NoAggregation(AggregationInterface):
    """Performs no aggregation, only flattens the input array."""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        return np.prod(activation_shape)

    def aggregate(self, layer, activations) -> np.ndarray:
        return activations.flatten()


class MeanAggregation(AggregationInterface):
    """Averages convolutional layers, does nothing for other types."""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        if len(activation_shape) == 3:
            # Convolutional layer
            return activation_shape[-1]
        else:
            return np.prod(activation_shape)

    def aggregate(self, layer, activations) -> np.ndarray:
        if (len(activations.shape) == 3):
            # Convolutional-like layer
            return [np.mean(activations[:, :, feature].flatten()) for feature in range(activations.shape[-1])]
        else:
            return activations.flatten()


class MeanStdAggregation(AggregationInterface):
    """Compute average and standard deviation for convolutional layers, does nothing for other types."""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        if len(activation_shape) == 3:
            # Convolutional layer
            return 2*activation_shape[-1]
        else:
            return np.prod(activation_shape)

    def aggregate(self, layer, activations) -> np.ndarray:
        if len(activations.shape) == 3:
            # Convolutional-like layer
            mean_std = []
            for feature in range(activations.shape[-1]):
                activation = activations[:, :, feature].flatten()
                mean_std.append(np.mean(activation))
                mean_std.append(np.std(activation))
            return mean_std
        else:
            return activations.flatten()


class MaxAggregation(AggregationInterface):
    """Maximum activation for convolutional layers, does nothing for other types."""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        if len(activation_shape) == 3:
            # Convolutional layer
            return activation_shape[-1]
        else:
            return np.prod(activation_shape)

    def aggregate(self, layer, activations) -> np.ndarray:
        """Aggregate layer activations"""
        if len(activations.shape) == 3:
            # Convolutional-like layer
            return [np.max(activations[:, :, feature].flatten()) for feature in range(activations.shape[-1])]
        else:
            return activations.flatten()
