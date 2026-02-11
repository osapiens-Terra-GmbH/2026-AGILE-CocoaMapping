from abc import ABC, abstractmethod
import copy
import random
from typing import Literal
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EvaluationMetrics(ABC):
    """Base class for evaluation metrics.

    This class implements the common functionality for all evaluation metrics, which are:
    - Aggregating tp, fp, tn, fn values
    - Adding pos_preds, neg_preds, label values to the metrics to be aggregated
    - Computing metrics such as f1, recall, etc., as a dictionary of {metric_name: metric_value}
    - Adding other evaluation metric object

    The inherited classes should implement the _normalize method, which defines how to convert pos_preds, neg_preds, label values into tp, fp, tn, fn values.
    For example, for pixelwise metrics, we do it by pixels, for area normalized metrics, we divide by the area of the sample, and so on.

    We also use the METRICS_REGISTRY dictionary to register the metrics by their type (type attribute)
    """
    type: str
    """The type of the evaluation metrics. Used to identify the metrics in the registry."""

    def __init__(self):
        """Initialize the evaluation metrics with zero tp, fp, tn, fn values."""
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def __add__(self, other: 'EvaluationMetrics' | Literal[0]) -> 'EvaluationMetrics':
        """Add two evaluation metrics."""
        if not ((isinstance(other, EvaluationMetrics) and self.type == other.type) or other == 0):
            raise ValueError("One can only add EvaluationMetrics to EvaluationMetrics with the same type.")

        if other == 0:
            return copy.deepcopy(self)

        result = self.__class__()  # same class as self
        result.tp = self.tp + other.tp
        result.fp = self.fp + other.fp
        result.tn = self.tn + other.tn
        result.fn = self.fn + other.fn
        return result

    def __radd__(self, other: 'EvaluationMetrics' | Literal[0]) -> 'EvaluationMetrics':
        """Add two evaluation metrics."""
        return self.__add__(other)

    @abstractmethod
    def _normalize(self, pos_preds: int, neg_preds: int, ground_truth: bool) -> tuple[int, int, int, int]:
        """Normalize predictions into tp, fp, tn, fn."""
        ...

    def add(self, pos_preds: int, neg_preds: int, label: int):
        """Add pixelwise tp, fp, tn, fn values for a single sample."""
        assert pos_preds + neg_preds > 0, "pos_preds + neg_preds must be greater than 0."
        tp, fp, tn, fn = self._normalize(pos_preds, neg_preds, label)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute_metrics(self) -> dict[str, float]:
        """Compute metrics as dictionary of {metric_name: metric_value}."""
        recall = _safe_division(self.tp, self.tp + self.fn)
        precision = _safe_division(self.tp, self.tp + self.fp)
        accuracy = _safe_division(self.tp + self.tn, self.tp + self.fp + self.tn + self.fn)
        f1_score = _safe_division(2 * (precision * recall), precision + recall)

        neg_recall = _safe_division(self.tn, self.tn + self.fp)
        neg_precision = _safe_division(self.tn, self.tn + self.fn)
        neg_f1_score = _safe_division(2 * (neg_precision * neg_recall), neg_precision + neg_recall)

        metrics = {
            'f1_score': f1_score,
            'recall': recall,
            'precision': precision,
            "neg_f1_score": neg_f1_score,
            "neg_recall": neg_recall,
            "neg_precision": neg_precision,
            'accuracy': accuracy,
            'tp': self.tp,
            'fp': self.fp,
            'tn': self.tn,
            'fn': self.fn}
        return {f'{self.type}/{key}': value for key, value in metrics.items()}


def _safe_division(a: float, b: float, default: float = 0) -> float:
    if b == 0:
        logger.warning(f"Division by zero: {a} / {b}. Returning {default}.")
        return default
    return a / b


class AreaNormalizedEvaluationMetrics(EvaluationMetrics):
    """Evaluation metrics normalized by the size of the sample."""
    type = "area_normalized"

    def _normalize(self, pos_preds: int, neg_preds: int, ground_truth: bool) -> tuple[int, int, int, int]:
        """Normalize predictions into tp, fp, tn, fn."""
        total_size = pos_preds + neg_preds
        if ground_truth:
            # tp, fp, tn, fn.
            return pos_preds / total_size, 0, 0, neg_preds / total_size
        else:
            # tp, fp, tn, fn
            return 0, pos_preds / total_size, neg_preds / total_size, 0


class PixelWiseEvaluationMetrics(EvaluationMetrics):
    """Evaluation metrics pixelwise, no normalization."""
    type = "pixelwise"

    def _normalize(self, pos_preds: int, neg_preds: int, label: bool) -> tuple[int, int, int, int]:
        if label:
            # tp, fp, tn, fn
            return pos_preds, 0, 0, neg_preds
        else:
            # tp, fp, tn, fn
            return 0, pos_preds, neg_preds, 0


class SamplePointEvaluationMetrics(EvaluationMetrics):
    """Sample a random point from the predictions and use it to evaluate the metrics."""
    type = "sample_point"

    def _normalize(self, pos_preds: int, neg_preds: int, ground_truth: bool) -> tuple[int, int, int, int]:
        """Normalize predictions into tp, fp, tn, fn."""
        random_point = random.randint(1, pos_preds + neg_preds)
        is_pos_pred = random_point <= pos_preds

        if ground_truth:
            # tp, fp, tn, fn
            return int(is_pos_pred), 0, 0, int(not is_pos_pred)
        else:
            # tp, fp, tn, fn
            return 0, int(is_pos_pred), int(not is_pos_pred), 0


class ThresholdEvaluationMetrics(EvaluationMetrics, ABC):
    """Evaluation metrics normalized by the threshold. If sample is predicted above threshold, it is considered positive, otherwise negative."""
    threshold: float

    def _normalize(self, pos_preds: int, neg_preds: int, ground_truth: bool) -> tuple[int, int, int, int]:
        """Normalize predictions into tp, fp, tn, fn."""
        total_size = pos_preds + neg_preds
        is_pos_pred = (pos_preds / total_size) > self.threshold
        if ground_truth:
            # tp, fp, tn, fn
            return int(is_pos_pred), 0, 0, int(not is_pos_pred)
        else:
            # tp, fp, tn, fn
            return 0, int(is_pos_pred), int(not is_pos_pred), 0


class Threshold50EvaluationMetrics(ThresholdEvaluationMetrics):
    """Evaluation metrics normalized by the threshold. If sample is predicted above 50%, it is considered positive, otherwise negative."""
    type = "threshold_50"
    threshold = 0.5


class Threshold10EvaluationMetrics(ThresholdEvaluationMetrics):
    """Evaluation metrics normalized by the threshold. If sample is predicted above 10%, it is considered positive, otherwise negative."""
    type = "threshold_10"
    threshold = 0.1


class Threshold90EvaluationMetrics(ThresholdEvaluationMetrics):
    """Evaluation metrics normalized by the threshold. If sample is predicted above 90%, it is considered positive, otherwise negative."""
    type = "threshold_90"
    threshold = 0.9


ALL_METRICS = [AreaNormalizedEvaluationMetrics,
               PixelWiseEvaluationMetrics,
               SamplePointEvaluationMetrics,
               Threshold10EvaluationMetrics,
               Threshold50EvaluationMetrics,
               Threshold90EvaluationMetrics]
"""All evaluation metrics."""
ALL_METRICS_TYPES = [metric.type for metric in ALL_METRICS]
"""All evaluation metrics types."""

METRICS_REGISTRY: dict[str, type[EvaluationMetrics]] = {metric.type: metric for metric in ALL_METRICS}
"""Metrics registry."""


class MultipleEvaluationMetrics(EvaluationMetrics):
    """Evaluation metrics for multiple evaluation metrics."""
    type = "multiple"

    def __init__(self, metrics_types: list[str]):
        """Initialize the multiple evaluation metrics."""
        self.metrics_types = metrics_types
        for metric_type in metrics_types:
            if metric_type not in METRICS_REGISTRY:
                raise ValueError(f"Metric type {metric_type} not found in the metrics registry.")
        self.metrics = [METRICS_REGISTRY[metric_type]() for metric_type in metrics_types]

    def __add__(self, other: 'MultipleEvaluationMetrics' | Literal[0]) -> 'MultipleEvaluationMetrics':
        """Add two multiple evaluation metrics."""
        if not (isinstance(other, MultipleEvaluationMetrics) or other == 0):
            raise ValueError("One can only add MultipleEvaluationMetrics to MultipleEvaluationMetrics")

        if other == 0:
            return copy.deepcopy(self)

        if self.metrics_types != other.metrics_types:
            raise ValueError("One can only add MultipleEvaluationMetrics with the same metrics types.")

        metrics = self.__class__(self.metrics_types)
        metrics.metrics = [metric + other.metrics[i] for i, metric in enumerate(self.metrics)]
        return metrics

    def __radd__(self, other: 'MultipleEvaluationMetrics' | Literal[0]) -> 'MultipleEvaluationMetrics':
        """Add two multiple evaluation metrics."""
        return self.__add__(other)

    def _normalize(self, pos_preds: int, neg_preds: int, ground_truth: bool) -> tuple[int, int, int, int]:
        raise NotImplementedError("_normalize on MultipleEvaluationMetrics is not intended to be used directly.")

    def add(self, pos_preds: int, neg_preds: int, label: int):
        """Add pixelwise tp, fp, tn, fn values for a single sample."""
        for metric in self.metrics:
            metric.add(pos_preds, neg_preds, label)

    def compute_metrics(self) -> dict[str, float]:
        """Compute metrics as a dictionary of {metric_type}/{metric_name}: metric_value."""
        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict.update(metric.compute_metrics())
        return metrics_dict


def load_metrics(metrics_types: str | list[str] | Literal['all']) -> EvaluationMetrics:
    f"""Load metrics from the registry.

    For a single metric type, the results of the compute_metrics will be metric_name: metric_value.
    For multiple metrics, the results of the compute_metrics will be metric_type/metric_name: metric_value.

    Args:
        metrics_type: The type of metrics to load. Can be a single metric type or a list of metric types.
            If multiple metrics are provided, a MultipleEvaluationMetrics instance will be returned.
            If 'all' is provided, all metrics will be loaded.

    Returns:
        EvaluationMetrics: The loaded metrics.
    """
    if metrics_types == 'all':
        return MultipleEvaluationMetrics(metrics_types=list(METRICS_REGISTRY.keys()))
    elif isinstance(metrics_types, str):
        return METRICS_REGISTRY[metrics_types]()
    elif 'all' in metrics_types:
        return MultipleEvaluationMetrics(metrics_types=list(METRICS_REGISTRY.keys()))
    else:
        return MultipleEvaluationMetrics(metrics_types=metrics_types)
