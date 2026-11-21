"""Definición de los pipelines de scikit-learn y baselines."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def _preprocesador(escenario: str, escalar: bool) -> ColumnTransformer:
    numericas = [c for c in config.FEATURES[escenario] if c not in config.CATEGORICAS]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAS),
        ("num", StandardScaler() if escalar else "passthrough", numericas),
    ])


def pipeline_ridge(escenario: str, alpha: float = 1.0) -> Pipeline:
    return Pipeline([("prep", _preprocesador(escenario, escalar=True)),
                     ("modelo", Ridge(alpha=alpha))])


def pipeline_rf(escenario: str, params: dict | None = None) -> Pipeline:
    p = {**config.PARAMS_RF_DEFECTO, **(params or {})}
    rf = RandomForestRegressor(random_state=config.SEMILLA, n_jobs=-1, **p)
    return Pipeline([("prep", _preprocesador(escenario, escalar=False)), ("modelo", rf)])


class BaselineEstacional:
    """Baseline sin aprendizaje: promedio (en log) por fuente y mes del año en el set de entrenamiento."""

    def fit(self, X: pd.DataFrame, y_log):
        d = X[["fuente", "mes_num"]].assign(y=np.asarray(y_log))
        self.tabla_ = d.groupby(["fuente", "mes_num"])["y"].mean()
        self.por_fuente_ = d.groupby("fuente")["y"].mean()
        self.global_ = float(d["y"].mean())
        return self

    def predict(self, X: pd.DataFrame):
        idx = pd.MultiIndex.from_frame(X[["fuente", "mes_num"]])
        pred = self.tabla_.reindex(idx).to_numpy()
        fallback = X["fuente"].map(self.por_fuente_).fillna(self.global_).to_numpy()
        return np.where(np.isnan(pred), fallback, pred)


def a_gwh(pred_log) -> np.ndarray:
    """Vuelve de log1p a GWh (nunca negativo)."""
    return np.clip(np.expm1(np.asarray(pred_log)), 0, None)
