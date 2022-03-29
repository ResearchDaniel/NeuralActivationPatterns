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
        if len(shape) == 3:
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
        return np.max(abs_max)

    def normalize(self, aggregated_activations, normalization_val) -> np.ndarray:
        if normalization_val != 0:
            return aggregated_activations / normalization_val
        return aggregated_activations


class MeanAggregation(AggregationInterface):
    """Averages convolutional layers, does nothing for other types."""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        if len(activation_shape) == 3:
            # Convolutional layer
            return activation_shape[-1]
        if len(activation_shape) == 2 and activation_shape[1] > 1:
            return 1
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
        if len(activation_shape) == 3:
            # Convolutional layer
            return [2, activation_shape[-1]]
        if len(activation_shape) == 2 and activation_shape[1] > 1:
            return 2
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
        std_norm = np.divide(1, np.sqrt(abs_max[0]), out=np.zeros_like(abs_max[1]),
                             where=abs_max[0] != 0)
        return [abs_max[0], std_norm]

    def normalize(self, aggregated_activations, normalization_val) -> np.ndarray:
        # Divide mean by 1/abs(mean) and standard deviation by 1/sqrt(mean), see normalization_value
        return np.divide(
            aggregated_activations, normalization_val, out=np.zeros_like(aggregated_activations),
            where=normalization_val != 0)


class MaxAggregation(AggregationInterface):
    """Maximum activation for convolutional layers, does nothing for other types."""

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        if len(activation_shape) == 3:
            # Convolutional layer
            return activation_shape[-1]
        if len(activation_shape) == 2 and activation_shape[1] > 1:
            return 1
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
