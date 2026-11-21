"""Carga de datos limpios y construcción de variables para el modelo."""
import numpy as np
import pandas as pd

from . import config


def cargar_datos(ruta=None, excluir_fuentes=config.FUENTES_EXCLUIDAS) -> pd.DataFrame:
    """Lee el CSV limpio generado por el notebook 01 y excluye las fuentes no modelables."""
    ruta = ruta or config.RUTA_DATOS_LIMPIOS
    df = pd.read_csv(ruta, parse_dates=["mes", "primer_mes_gen"])
    df = df[~df["fuente"].isin(excluir_fuentes)].dropna(subset=["primer_mes_gen"])
    return df.sort_values(["maquina", "mes"]).reset_index(drop=True)


def codificar_mes(mes_num):
    """Codificación cíclica: diciembre y enero quedan 'cerca'."""
    angulo = 2 * np.pi * (np.asarray(mes_num) - 1) / 12
    return np.sin(angulo), np.cos(angulo)


def agregar_features_estaticas(df: pd.DataFrame) -> pd.DataFrame:
    """Variables que se conocen sin historial de la central."""
    df = df.copy()
    df["mes_num"] = df["mes"].dt.month
    df["mes_sin"], df["mes_cos"] = codificar_mes(df["mes_num"])
    df["antiguedad_meses"] = ((df["mes"].dt.year - df["primer_mes_gen"].dt.year) * 12
                              + (df["mes"].dt.month - df["primer_mes_gen"].dt.month))
    df["y_log"] = np.log1p(df["gwh"])
    return df


def agregar_rezagos(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los 3 meses previos de cada máquina. Descarta filas sin 3 meses de historial."""
    df = df.copy()
    base = df[["maquina", "mes", "gwh"]]
    for k in (1, 2, 3):
        rez = base.assign(mes=base["mes"] + pd.DateOffset(months=k)).rename(columns={"gwh": f"lag_{k}m"})
        df = df.merge(rez, on=["maquina", "mes"], how="left")
    cols = ["lag_1m", "lag_2m", "lag_3m"]
    df = df.dropna(subset=cols).copy()
    df["log_lag_1m"] = np.log1p(df["lag_1m"])
    df["log_media_3m"] = np.log1p(df[cols].mean(axis=1))
    return df.reset_index(drop=True)


def preparar_datasets(ruta=None) -> dict:
    """Devuelve {'estatico': df, 'historial': df} listos para entrenar."""
    base = agregar_features_estaticas(cargar_datos(ruta))
    return {"estatico": base, "historial": agregar_rezagos(base)}


def split_temporal(df: pd.DataFrame, test_inicio=config.TEST_INICIO):
    """Entrena con lo anterior a `test_inicio`, testea desde esa fecha (sin mezclar el futuro)."""
    corte = pd.Timestamp(test_inicio)
    return df[df["mes"] < corte].copy(), df[df["mes"] >= corte].copy()


def construir_fila(entrada: dict, con_historial: bool) -> pd.DataFrame:
    """Convierte una solicitud de la API en la fila de features que espera el modelo.

    entrada: fuente, region, provincia, programa, mes (1-12), antiguedad_meses y,
    si con_historial, gwh_ultimos_3_meses = [mes-1, mes-2, mes-3].
    """
    seno, coseno = codificar_mes(entrada["mes"])
    fila = {
        "fuente": entrada["fuente"].upper().strip(),
        "region": entrada["region"].upper().strip(),
        "provincia": entrada["provincia"].upper().strip(),
        "programa": entrada["programa"].upper().strip(),
        "mes_sin": float(seno),
        "mes_cos": float(coseno),
        "antiguedad_meses": float(entrada["antiguedad_meses"]),
    }
    if con_historial:
        ultimos = list(entrada["gwh_ultimos_3_meses"])
        fila["log_lag_1m"] = float(np.log1p(ultimos[0]))
        fila["log_media_3m"] = float(np.log1p(np.mean(ultimos)))
    return pd.DataFrame([fila])[config.FEATURES["historial" if con_historial else "estatico"]]
