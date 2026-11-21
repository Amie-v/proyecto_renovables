"""Tests de la lógica de predicción (no requieren levantar FastAPI).

Ejecutar desde la raíz del proyecto:  pytest -q
Requieren que existan los modelos: python -m src.train
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluacion import metricas
from src.modelos import a_gwh
from src.predictor import EntradaInvalida, ModeloNoDisponible, Predictor
from src.preprocessing import agregar_rezagos, codificar_mes, construir_fila

ENTRADA = {"fuente": "EOLICO", "region": "PATAGONIA", "provincia": "CHUBUT",
           "programa": "RENOVAR", "mes": 10, "antiguedad_meses": 24}
ENTRADA_HIST = {**ENTRADA, "gwh_ultimos_3_meses": [14.2, 13.1, 15.8]}


def _predictor() -> Predictor:
    try:
        return Predictor()
    except ModeloNoDisponible:
        pytest.skip("Modelos no entrenados: ejecutá `python -m src.train`")


# ---- funciones puras -----------------------------------------------------------------
def test_codificacion_ciclica_enero_y_diciembre_estan_cerca():
    s1, c1 = codificar_mes(1)
    s12, c12 = codificar_mes(12)
    s6, c6 = codificar_mes(6)
    dist_ene_dic = np.hypot(s1 - s12, c1 - c12)
    dist_ene_jun = np.hypot(s1 - s6, c1 - c6)
    assert dist_ene_dic < dist_ene_jun


def test_construir_fila_estatica_tiene_las_columnas_del_modelo():
    fila = construir_fila(ENTRADA, con_historial=False)
    assert list(fila.columns) == config.FEATURES["estatico"]
    assert fila.shape == (1, len(config.FEATURES["estatico"]))


def test_construir_fila_historial_usa_log1p_de_los_rezagos():
    fila = construir_fila(ENTRADA_HIST, con_historial=True)
    assert list(fila.columns) == config.FEATURES["historial"]
    assert fila["log_lag_1m"].iloc[0] == pytest.approx(np.log1p(14.2))
    assert fila["log_media_3m"].iloc[0] == pytest.approx(np.log1p(np.mean([14.2, 13.1, 15.8])))


def test_rezagos_no_usan_informacion_del_futuro():
    meses = pd.date_range("2024-01-01", periods=6, freq="MS")
    df = pd.DataFrame({"maquina": "X", "mes": meses, "gwh": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    out = agregar_rezagos(df)
    # La primera fila con 3 meses previos es abril (gwh=4): lag_1m=3, lag_2m=2, lag_3m=1
    assert out["mes"].iloc[0] == pd.Timestamp("2024-04-01")
    assert out["lag_1m"].iloc[0] == 3.0 and out["lag_3m"].iloc[0] == 1.0
    assert len(out) == 3


def test_metricas_prediccion_perfecta():
    y = np.array([1.0, 2.0, 5.0])
    m = metricas(y, y)
    assert m["mae_gwh"] == 0 and m["wape"] == 0 and m["r2_log"] == pytest.approx(1.0)


def test_a_gwh_nunca_es_negativo():
    assert (a_gwh(np.array([-5.0, 0.0, 1.0])) >= 0).all()


# ---- predictor con los modelos entrenados -----------------------------------------------
def test_prediccion_estatica_es_positiva_y_el_rango_esta_ordenado():
    r = _predictor().predecir(ENTRADA, "estatico")
    bajo, alto = r["rango_gwh"]
    assert r["prediccion_gwh"] >= 0
    assert 0 <= bajo <= alto


def test_prediccion_con_historial_sigue_a_los_rezagos():
    p = _predictor()
    alto = p.predecir({**ENTRADA, "gwh_ultimos_3_meses": [40.0, 40.0, 40.0]}, "historial")["prediccion_gwh"]
    bajo = p.predecir({**ENTRADA, "gwh_ultimos_3_meses": [2.0, 2.0, 2.0]}, "historial")["prediccion_gwh"]
    assert alto > bajo


def test_categoria_invalida_lanza_error():
    with pytest.raises(EntradaInvalida):
        _predictor().predecir({**ENTRADA, "fuente": "NUCLEAR"}, "estatico")


def test_combinacion_inedita_genera_advertencia():
    # Hay eólica y hay Cuyo, pero no eólica en Cuyo/Mendoza (ver EDA): se debe avisar de la extrapolación.
    r = _predictor().predecir({**ENTRADA, "region": "CUYO", "provincia": "MENDOZA"}, "estatico")
    assert any("combinación" in a for a in r["advertencias"])


def test_es_deterministico():
    p = _predictor()
    assert p.predecir(ENTRADA, "estatico")["prediccion_gwh"] == p.predecir(ENTRADA, "estatico")["prediccion_gwh"]
