"""Master experiment configuration — wraps all sub-configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.config.base_config import BaseConfig
from src.config.data_config import DataConfig
from src.config.eval_config import EvalConfig
from src.config.feature_config import FeatureConfig
from src.config.model_config import LGBMConfig, XGBConfig
from src.config.training_config import TrainingConfig


def _build_model_config(raw: dict[str, Any]) -> XGBConfig | LGBMConfig:
    """Route model section to the correct config class based on ``model_type``."""
    model_type = raw.get("model_type", "xgboost")
    if model_type == "lightgbm":
        return LGBMConfig(**raw)
    return XGBConfig(**raw)


@dataclass
class ExperimentConfig(BaseConfig):
    """Top-level config that aggregates all sub-configs.

    YAML layout expected::

        data:
          train_end_year: 2016
          ...
        feature:
          genre_vector_dim: 18
          ...
        model:
          model_type: xgboost
          ...
        training:
          experiment_name: movie_recommender
          ...
        eval:
          k: 10
          ...
    """

    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: XGBConfig | LGBMConfig = field(default_factory=XGBConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------
    # Override __post_init__ so that validate() is still called once all
    # sub-config fields are populated (the default factory creates them
    # before we can override them from YAML).
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Delegate to each sub-config — they each call their own validate()."""
        self.data.validate()
        self.feature.validate()
        self.model.validate()
        self.training.validate()
        self.eval.validate()

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":  # type: ignore[override]
        """Load a single nested YAML file and return a fully validated config."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        data_cfg = DataConfig(**(raw.get("data", {})))
        feature_cfg = FeatureConfig(**(raw.get("feature", {})))
        model_cfg = _build_model_config(raw.get("model", {}))
        training_cfg = TrainingConfig(**(raw.get("training", {})))
        eval_cfg = EvalConfig(**(raw.get("eval", {})))

        # Bypass default-factory construction so we can pass our instances.
        obj = cls.__new__(cls)
        object.__setattr__(obj, "data", data_cfg)
        object.__setattr__(obj, "feature", feature_cfg)
        object.__setattr__(obj, "model", model_cfg)
        object.__setattr__(obj, "training", training_cfg)
        object.__setattr__(obj, "eval", eval_cfg)
        obj.__post_init__()
        return obj

    # ------------------------------------------------------------------
    # MLflow export
    # ------------------------------------------------------------------
    def to_mlflow_params(self) -> dict[str, str]:
        """Return all sub-config params as strings with dot-prefixed keys.

        Example: ``{"data.train_end_year": "2016", "model.learning_rate": "0.05"}``
        """
        result: dict[str, str] = {}
        for prefix, cfg in (
            ("data", self.data),
            ("feature", self.feature),
            ("model", self.model),
            ("training", self.training),
            ("eval", self.eval),
        ):
            for k, v in cfg.to_mlflow_params().items():
                result[f"{prefix}.{k}"] = v
        return result
