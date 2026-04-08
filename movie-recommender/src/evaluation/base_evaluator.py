"""Abstract base evaluator."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """All evaluators accept predictions and labels and return a metric dict."""

    @abstractmethod
    def evaluate(
        self,
        predictions: list[list[int]],
        labels: list[set[int]],
    ) -> dict[str, float]:
        """Compute metrics.

        Parameters
        ----------
        predictions:
            Outer list = one query (user); inner list = ranked movie IDs
            (best first, up to k items).
        labels:
            Outer list = one query; inner set = relevant movie IDs for
            that query (rating >= relevance_threshold).

        Returns
        -------
        dict[str, float]
            Metric name → value.  All values must be scalar floats.
        """
