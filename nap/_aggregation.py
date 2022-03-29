"""Aggregation funcions to reduce dimensionality."""
import numpy as np


class AggregationInterface:
    """Interface for NAP aggregation computers"""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""

    def aggregate(self, layer, activations) -> np.ndarray:
        """Aggregate layer activations"""

    def normalization_value(self, abs_max):
        """Value to normalize aggregations"""

    def normalize(self, aggregated_activations, normalization_val) -> np.ndarray:
        """Normalize aggregated activations"""

    @staticmethod
    def should_aggregate(shape):
        if len(shape) != 1:
            return True
        return False


class NoAggregation(AggregationInterface):
    """Performs no aggregation, only flattens the input array."""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        return activation_shape

    def aggregate(self, layer, activations) -> np.ndarray:
        return activations

    def normalization_value(self, abs_max):
        # Per unit maximum
        return [np.max(abs_max[..., unit].ravel())
                for unit in range(abs_max.shape[-1])]

    def normalize(self, aggregated_activations, normalization_val) -> np.ndarray:
        return np.divide(
            aggregated_activations, normalization_val, out=np.zeros_like(aggregated_activations),
            where=normalization_val != 0)


class MeanAggregation(AggregationInterface):
    """Averages convolutional layers, does nothing for other types."""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        if self.should_aggregate(activation_shape):
            # Convolutional layer
            return activation_shape[-1]
        return activation_shape

    def aggregate(self, layer, activations) -> np.ndarray:
        return [np.mean(activations[:, :, feature].ravel())
                for feature in range(activations.shape[-1])]

    def normalization_value(self, abs_max):
        return abs_max

    def normalize(self, aggregated_activations, normalization_val) -> np.ndarray:
        return np.divide(
            aggregated_activations, normalization_val, out=np.zeros_like(aggregated_activations),
            where=normalization_val != 0)


class MeanStdAggregation(AggregationInterface):
    """Compute average and standard deviation for convolutional layers,
    does nothing for other types.
    """

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        if self.should_aggregate(activation_shape):
            # Convolutional layer
            return [2, activation_shape[-1]]
        # Do nothing
        return activation_shape

    def aggregate(self, layer, activations) -> np.ndarray:
        mean = []
        std = []
        for feature in range(activations.shape[-1]):
            activation = activations[..., feature].ravel()
            mean.append(np.mean(activation))
            std.append(np.std(activation))
        return [mean, std]

    def normalization_value(self, abs_max):
        std_norm = np.sqrt(abs_max[0])
        return [abs_max[0], std_norm]

    def normalize(self, aggregated_activations, normalization_val) -> np.ndarray:
        # Divide mean by abs(mean) and standard deviation by sqrt(mean)
        return np.divide(
            aggregated_activations, normalization_val, out=np.zeros_like(aggregated_activations),
            where=normalization_val != 0)


class MaxAggregation(AggregationInterface):
    """Maximum activation for convolutional layers, does nothing for other types."""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        if self.should_aggregate(activation_shape):
            # Convolutional layer
            return activation_shape[-1]
        return activation_shape

    def aggregate(self, layer, activations) -> np.ndarray:
        """Aggregate layer activations"""
        return [np.max(activations[..., feature].ravel())
                for feature in range(activations.shape[-1])]

    def normalization_value(self, abs_max):
        return abs_max

    def normalize(self, aggregated_activations, normalization_val) -> np.ndarray:
        return np.divide(
            aggregated_activations, normalization_val, out=np.zeros_like(aggregated_activations),
            where=normalization_val != 0)
