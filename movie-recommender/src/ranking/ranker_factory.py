"""Factory that instantiates the correct ranker from an ExperimentConfig."""

from __future__ import annotations

from src.ranking.base_ranker import BaseRanker


class RankerFactory:
    """Create a ranker from an ``ExperimentConfig``.

    Supported model types (``config.model.model_type``):
    - ``"xgboost"`` → :class:`~src.ranking.xgb_ranker.XGBRanker`
    - ``"lightgbm"`` → :class:`~src.ranking.lgbm_ranker.LGBMRanker`
    """

    @staticmethod
    def create(config) -> BaseRanker:
        """Instantiate and return the ranker specified in *config*.

        Parameters
        ----------
        config:
            An ``ExperimentConfig`` (or any object with a ``.model``
            attribute that has ``model_type``, and the relevant
            hyper-parameter fields).

        Raises
        ------
        ValueError
            If ``config.model.model_type`` is not recognised.
        """
        model_cfg = config.model
        model_type: str = model_cfg.model_type.lower()

        if model_type == "xgboost":
            from src.ranking.xgb_ranker import XGBRanker

            return XGBRanker(
                n_estimators=model_cfg.n_estimators,
                max_depth=model_cfg.max_depth,
                learning_rate=model_cfg.learning_rate,
                subsample=model_cfg.subsample,
                colsample_bytree=model_cfg.colsample_bytree,
                early_stopping=getattr(model_cfg, "early_stopping", 50),
                random_state=config.data.random_seed,
                device=getattr(model_cfg, "device", "cpu"),
            )

        if model_type == "lightgbm":
            from src.ranking.lgbm_ranker import LGBMRanker

            return LGBMRanker(
                n_estimators=model_cfg.n_estimators,
                max_depth=getattr(model_cfg, "max_depth", -1),  # LGBMConfig has no max_depth; -1 = unlimited
                learning_rate=model_cfg.learning_rate,
                num_leaves=getattr(model_cfg, "num_leaves", 63),
                subsample=model_cfg.subsample,
                colsample_bytree=model_cfg.colsample_bytree,
                early_stopping=getattr(model_cfg, "early_stopping", 50),
                random_state=config.data.random_seed,
                device=getattr(model_cfg, "device", "cpu"),
            )

        raise ValueError(
            f"RankerFactory: unknown model_type {model_type!r}. "
            "Supported: 'xgboost', 'lightgbm'."
        )
