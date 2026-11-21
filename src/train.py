"""Entrena y guarda los modelos que usa la API.

Uso (desde la raíz del proyecto):
    python -m src.train

Si existe models/best_params.json (lo genera el notebook 02 con Optuna) se usan esos
hiperparámetros; si no, se usan los valores por defecto de config.PARAMS_RF_DEFECTO.

Para cada escenario:
  1. Entrena con datos anteriores a config.TEST_INICIO y mide en el período posterior (métricas honestas).
  2. Re-entrena con TODOS los datos y guarda ese modelo para producción.
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config
from .evaluacion import metricas
from .modelos import BaselineEstacional, a_gwh, pipeline_rf
from .preprocessing import preparar_datasets, split_temporal

AVISO_USO = (
    "Modelo educativo. No conoce la potencia instalada ni el clima de cada central, "
    "fue entrenado solo con centrales que existen (sesgo de supervivencia) y no debe "
    "usarse para decisiones de inversión o despacho sin validación de especialistas."
)


def cargar_params() -> dict:
    if config.RUTA_MEJORES_PARAMS.exists():
        return json.loads(config.RUTA_MEJORES_PARAMS.read_text(encoding="utf-8"))
    return {"estatico": dict(config.PARAMS_RF_DEFECTO), "historial": dict(config.PARAMS_RF_DEFECTO)}


def _cuantiles_residuo(te: pd.DataFrame, pred_log: np.ndarray) -> dict:
    """Cuantiles 10/90 del error (en log) por fuente: dan un rango aproximado en la API."""
    res = (te["y_log"].to_numpy() - pred_log)
    out = {"_global": [float(np.quantile(res, 0.10)), float(np.quantile(res, 0.90))]}
    for fuente, idx in te.groupby("fuente").indices.items():
        if len(idx) >= 30:
            out[fuente] = [float(np.quantile(res[idx], 0.10)), float(np.quantile(res[idx], 0.90))]
    return out


def calcular_cuantiles_validacion(df: pd.DataFrame, escenario: str, params: dict) -> dict:
    """Calibra el rango de incertidumbre con un año de validación ANTERIOR al test (evita circularidad)."""
    cols = config.FEATURES[escenario]
    previo = df[df["mes"] < pd.Timestamp(config.TEST_INICIO)]
    tr, va = split_temporal(previo, config.VALIDACION_INICIO)
    modelo = pipeline_rf(escenario, params).fit(tr[cols], tr["y_log"])
    return _cuantiles_residuo(va, modelo.predict(va[cols]))


def entrenar_escenario(df: pd.DataFrame, escenario: str, params: dict) -> dict:
    cols = config.FEATURES[escenario]
    tr, te = split_temporal(df)

    modelo = pipeline_rf(escenario, params).fit(tr[cols], tr["y_log"])
    pred_log = modelo.predict(te[cols])
    base = BaselineEstacional().fit(tr, tr["y_log"])

    info = {
        "params": params,
        "filas_train": int(len(tr)),
        "filas_test": int(len(te)),
        "metricas_test": metricas(te["gwh"], a_gwh(pred_log)),
        "metricas_baseline": metricas(te["gwh"], a_gwh(base.predict(te))),
        "cuantiles_residuo_log": calcular_cuantiles_validacion(df, escenario, params),
    }
    if escenario == "historial":
        info["metricas_persistencia"] = metricas(te["gwh"], np.expm1(te["log_lag_1m"]))

    final = pipeline_rf(escenario, params).fit(df[cols], df["y_log"])
    config.MODELOS_DIR.mkdir(exist_ok=True)
    joblib.dump(final, config.RUTA_MODELO[escenario], compress=3)
    return info


def main() -> dict:
    datasets = preparar_datasets()
    params = cargar_params()
    base = datasets["estatico"]

    metadata = {
        "entrenado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn": sklearn.__version__,
        "test_inicio": config.TEST_INICIO,
        "datos_desde": str(base["mes"].min().date()),
        "datos_hasta": str(base["mes"].max().date()),
        "fuentes_excluidas": list(config.FUENTES_EXCLUIDAS),
        "categorias_validas": {c: sorted(base[c].unique().tolist()) for c in config.CATEGORICAS},
        "combinaciones_vistas": sorted(
            {"|".join(t) for t in base[["fuente", "region", "provincia"]].drop_duplicates().itertuples(index=False)}),
        "antiguedad_max_vista": int(base["antiguedad_meses"].max()),
        "aviso_uso": AVISO_USO,
        "escenarios": {},
    }
    for escenario in ("estatico", "historial"):
        print(f"Entrenando escenario '{escenario}'...")
        metadata["escenarios"][escenario] = entrenar_escenario(
            datasets[escenario], escenario, params.get(escenario, config.PARAMS_RF_DEFECTO))

    config.RUTA_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Listo. Modelos y metadata guardados en", config.MODELOS_DIR)
    return metadata


if __name__ == "__main__":
    main()
