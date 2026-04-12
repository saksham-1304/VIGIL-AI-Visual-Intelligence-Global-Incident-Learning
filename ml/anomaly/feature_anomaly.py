from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


def train_isolation_forest(
    features: np.ndarray,
    n_estimators: int = 300,
    contamination: float = 0.03,
    random_state: int = 42,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(features)
    return model


def anomaly_scores(model: IsolationForest, features: np.ndarray) -> np.ndarray:
    decision = model.decision_function(features)
    score = 1 / (1 + np.exp(3 * decision))
    return score


def save_iforest(model: IsolationForest, path: str) -> None:
    joblib.dump(model, path)


def load_iforest(path: str) -> IsolationForest:
    return joblib.load(path)
