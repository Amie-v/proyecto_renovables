"""Métricas y análisis de error por subgrupo."""
import numpy as np
import pandas as pd


def metricas(y_real_gwh, y_pred_gwh) -> dict:
    """Métricas en GWh y en escala log. WAPE = suma de errores absolutos / suma de lo real."""
    y_real = np.asarray(y_real_gwh, dtype=float)
    y_pred = np.asarray(y_pred_gwh, dtype=float)
    err = y_pred - y_real
    log_real, log_pred = np.log1p(y_real), np.log1p(y_pred)
    ss_res = np.sum((log_real - log_pred) ** 2)
    ss_tot = np.sum((log_real - log_real.mean()) ** 2)
    return {
        "n": int(len(y_real)),
        "mae_gwh": float(np.mean(np.abs(err))),
        "rmse_gwh": float(np.sqrt(np.mean(err ** 2))),
        "wape": float(np.sum(np.abs(err)) / max(np.sum(y_real), 1e-9)),
        "mae_log": float(np.mean(np.abs(log_real - log_pred))),
        "r2_log": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def error_por_grupo(df: pd.DataFrame, col_grupo: str, col_real="gwh", col_pred="pred") -> pd.DataFrame:
    """Tabla de error por subgrupo (región, fuente, ...). Incluye nº de máquinas y sesgo."""
    filas = []
    for grupo, sub in df.groupby(col_grupo):
        m = metricas(sub[col_real], sub[col_pred])
        m["maquinas"] = sub["maquina"].nunique()
        m["sesgo_gwh"] = float((sub[col_pred] - sub[col_real]).mean())
        m[col_grupo] = grupo
        filas.append(m)
    cols = [col_grupo, "n", "maquinas", "mae_gwh", "wape", "mae_log", "r2_log", "sesgo_gwh"]
    return pd.DataFrame(filas)[cols].sort_values("n", ascending=False).reset_index(drop=True)
