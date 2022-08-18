"""Location disentanglement functions."""
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
        """Normalize aggregated activations. Equation 2 in the paper."""

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
            where=~np.isclose(normalization_val, np.zeros_like(normalization_val)))


class MeanAggregation(AggregationInterface):
    """Averages convolutional layers, does nothing for other types.
       Referred to as Feature amount in the paper.
       The average of all elements of the activation matrix is extracted,
       thus reflecting the amount of a feature.
    """

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
            where=~np.isclose(normalization_val, np.zeros_like(normalization_val)))


class MeanStdAggregation(AggregationInterface):
    """Compute average and standard deviation for convolutional layers,
        does nothing for other types.
        Referred to as Feature amount and spread in the paper.
        Uses the average in combination with standard deviation to disambiguate cases
        where a single high activation value would produce the
        same result as many low activation values.
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
            where=~np.isclose(normalization_val, np.zeros_like(normalization_val)))


class MaxAggregation(AggregationInterface):
    """ Maximum activation for convolutional layers, does nothing for other types.
        Referred to as Peak feature strength in the paper.
        The maximum is extracted across all elements of the activation matrix.
        The input with highest activation generally tells us what
        the network considers most important.
    """

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
            where=~np.isclose(normalization_val, np.zeros_like(normalization_val)))


class MinMaxAggregation(AggregationInterface):
    """ Compute minimum and maximum for convolutional layers, does nothing for other types.
        Referred to as Feature range in the paper.
        The minimum and maximum are extracted across all elements of the activation matrix.
        For layers with negative activations, this allows for taking opposite features into account.
        However, for layers with activation functions suppressing negative values,
        the minimum value will in many cases be zero and therefore provide limited value.
    """

    def shape(self, activation_shape) -> list:
        """Return what the resulting shape of the aggragation will be for layer."""
        if self.should_aggregate(activation_shape):
            # Convolutional layer
            return [2, activation_shape[-1]]
        # Do nothing
        return activation_shape

    def aggregate(self, layer, activations) -> np.ndarray:
        minimum = []
        maximum = []
        for feature in range(activations.shape[-1]):
            activation = activations[..., feature].ravel()
            minimum.append(np.min(activation))
            maximum.append(np.max(activation))
        return [minimum, maximum]

    def normalization_value(self, abs_max):
        return abs_max

    def normalize(self, aggregated_activations, normalization_val) -> np.ndarray:
        return np.divide(
            aggregated_activations, normalization_val, out=np.zeros_like(aggregated_activations),
            where=~np.isclose(normalization_val, np.zeros_like(normalization_val)))
