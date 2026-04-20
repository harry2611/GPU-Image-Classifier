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
                (
                    "scaler",
                    # Image pixels are already normalized to [0, 1]. Centering helps the
                    # linear solvers, while skipping variance scaling avoids exploding
                    # rare low-variance pixels.
                    StandardScaler(with_std=False),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="saga",
                        random_state=random_state,
                        tol=5e-3,
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_std=False)),
                (
                    "model",
                    LinearSVC(
                        random_state=random_state,
                        dual=False,
                        max_iter=8000,
                        tol=1e-3,
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
