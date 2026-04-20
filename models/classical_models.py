from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from utils.config import SUPPORTED_CLASSICAL_MODELS


def build_classical_models(
    random_state: int,
    n_jobs: int,
) -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=400,
                        solver="lbfgs",
                        multi_class="multinomial",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearSVC(
                        random_state=random_state,
                        dual="auto",
                        max_iter=5000,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
        ),
    }
