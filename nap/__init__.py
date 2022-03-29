"""Providing functionality for computing nerual activation patterns."""
from ._aggregation import AggregationInterface
from ._aggregation import NoAggregation
from ._aggregation import MeanAggregation
from ._aggregation import MeanStdAggregation
from ._aggregation import MaxAggregation
from ._aggregation import MinMaxAggregation

from ._neural_activation_pattern import NeuralActivationPattern
from ._neural_activation_pattern import sort
from ._neural_activation_pattern import sample
from ._neural_activation_pattern import head
from ._neural_activation_pattern import outliers
from ._neural_activation_pattern import tail
from ._neural_activation_pattern import average
