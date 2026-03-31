"""Base configuration dataclass with YAML loading, validation, and MLflow export."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BaseConfig(ABC):
    """Abstract base for all config dataclasses.

    Subclasses must implement ``validate()``.  ``__post_init__`` calls it
    automatically so invalid configs are caught at construction time.
    """

    def __post_init__(self) -> None:
        self.validate()

    @abstractmethod
    def validate(self) -> None:
        """Raise ``AssertionError`` with a descriptive message on invalid state."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseConfig":
        """Load a YAML file, instantiate *cls* from its contents, and return it.

        The YAML file must contain a flat mapping whose keys correspond to the
        fields of *cls*.  Raises ``FileNotFoundError`` if *path* does not exist
        and ``AssertionError`` / ``TypeError`` on invalid values.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        return cls(**raw)  # type: ignore[return-value]

    def to_mlflow_params(self) -> dict[str, str]:
        """Return a flat ``{str: str}`` dict of all fields for MLflow logging."""
        flat: dict[str, str] = {}
        for key, value in asdict(self).items():
            flat[key] = str(value)
        return flat

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the config as a plain Python dict."""
        return copy.deepcopy(asdict(self))
