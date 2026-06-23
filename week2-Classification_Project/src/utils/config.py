from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42
    test_size: float = 0.2
    target_col: str = "Churn"
    positive_label: str = "Yes"

    # Models
    logistic_max_iter: int = 1000

    # Threshold tuning
    threshold_metric: str = "f1"  # f1 | precision | recall
    default_risk_tiers: tuple[str, str, str] = ("Low", "Medium", "High")

