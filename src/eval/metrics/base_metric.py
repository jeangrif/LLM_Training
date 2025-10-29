from abc import ABC, abstractmethod

class BaseMetric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, model=None):
        self.model = model

    @abstractmethod
    def compute(self, row, group=None):
        """Compute the metric for a given row or group."""
        pass
