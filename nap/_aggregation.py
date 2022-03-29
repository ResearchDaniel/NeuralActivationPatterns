"""Aggregation funcions to reduce dimensionality."""
import numpy as np


class AggregationInterface:
    """Interface for NAP aggregation computers"""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""

    def aggregate(self, layer, activations) -> np.ndarray:
        """Aggregate layer activations"""


class NoAggregation(AggregationInterface):
    """Performs no aggregation, only flattens the input array."""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        return activation_shape

    def aggregate(self, layer, activations) -> np.ndarray:
        return activations


class MeanAggregation(AggregationInterface):
    """Averages convolutional layers, does nothing for other types."""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        if len(activation_shape) == 3:
            # Convolutional layer
            return activation_shape[-1]
        if len(activation_shape) == 2 and activation_shape[1] > 1:
            return 1
        return np.prod(activation_shape)

    def aggregate(self, layer, activations) -> np.ndarray:
        if len(activations.shape) == 3:
            # Convolutional-like layer
            return [np.mean(activations[:, :, feature].ravel())
                    for feature in range(activations.shape[-1])]
        if len(activations.shape) == 2 and activations.shape[1] > 1:
            return [np.mean(activations)]
        return activations.ravel()


class MeanStdAggregation(AggregationInterface):
    """Compute average and standard deviation for convolutional layers,
    does nothing for other types.
    """

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        if len(activation_shape) == 3:
            # Convolutional layer
            return 2*activation_shape[-1]
        if len(activation_shape) == 2 and activation_shape[1] > 1:
            return 2
        return np.prod(activation_shape)

    def aggregate(self, layer, activations) -> np.ndarray:
        if len(activations.shape) == 3:
            # Convolutional-like layer
            mean_std = []
            for feature in range(activations.shape[-1]):
                activation = activations[..., feature].ravel()
                mean_std.append(np.mean(activation))
                mean_std.append(np.std(activation))
            return mean_std
        if len(activations.shape) == 2 and activations.shape[1] > 1:
            return [np.mean(activations), np.std(activation)]
        return activations.ravel()


class MaxAggregation(AggregationInterface):
    """Maximum activation for convolutional layers, does nothing for other types."""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        if len(activation_shape) == 3:
            # Convolutional layer
            return activation_shape[-1]
        if len(activation_shape) == 2 and activation_shape[1] > 1:
            return 1
        return np.prod(activation_shape)

    def aggregate(self, layer, activations) -> np.ndarray:
        """Aggregate layer activations"""
        if len(activations.shape) == 3:
            # Convolutional-like layer
            return [np.max(activations[..., feature].ravel())
                    for feature in range(activations.shape[-1])]
        if len(activations.shape) == 2 and activations.shape[1] > 1:
            return [np.max(activations)]
        return activations.ravel()
