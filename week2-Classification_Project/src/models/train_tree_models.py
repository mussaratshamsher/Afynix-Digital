from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.models.pipelines import make_preprocessor


def build_decision_tree_pipeline(X, imbalance: str, random_state: int = 42) -> Pipeline:
    preprocessor = make_preprocessor(X, model_type="tree")
    if imbalance == "class_weight":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=random_state)
    else:
        clf = DecisionTreeClassifier(random_state=random_state)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])


def build_random_forest_pipeline(X, imbalance: str, random_state: int = 42) -> Pipeline:
    preprocessor = make_preprocessor(X, model_type="tree")

    if imbalance == "smote":
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
        )
        smote = SMOTE(random_state=random_state)
        return ImbPipeline(steps=[("preprocessor", preprocessor), ("smote", smote), ("model", clf)])

    if imbalance == "class_weight":
        clf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=random_state)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])

