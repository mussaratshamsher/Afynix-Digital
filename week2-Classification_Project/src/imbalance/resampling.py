from __future__ import annotations

from dataclasses import dataclass

from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class ImbalanceStrategy:
    name: str  # 'class_weight' or 'smote'


def get_model_with_imbalance(model: BaseEstimator, strategy: ImbalanceStrategy) -> BaseEstimator:
    # This helper exists so train code can be explicit.
    return model

