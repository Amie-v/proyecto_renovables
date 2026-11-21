"""Tests de los endpoints de FastAPI.

Requieren: pip install fastapi httpx pytest   y los modelos entrenados (python -m src.train).
Ejecutar desde la raíz del proyecto:  pytest -q
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402

client = TestClient(app)

PAYLOAD = {"fuente": "EOLICO", "region": "PATAGONIA", "provincia": "CHUBUT",
           "programa": "RENOVAR", "mes": 10, "antiguedad_meses": 24}
PAYLOAD_HIST = {**PAYLOAD, "gwh_ultimos_3_meses": [14.2, 13.1, 15.8]}


def _modelo_disponible() -> bool:
    return client.get("/metadata").status_code == 200


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"estado": "ok"}


def test_predict_ok():
    if not _modelo_disponible():
        pytest.skip("Modelos no entrenados")
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["prediccion_gwh"] >= 0
    assert cuerpo["escenario"] == "estatico"
    assert len(cuerpo["rango_gwh"]) == 2


def test_predict_historial_ok():
    if not _modelo_disponible():
        pytest.skip("Modelos no entrenados")
    r = client.post("/predict/historial", json=PAYLOAD_HIST)
    assert r.status_code == 200 and r.json()["escenario"] == "historial"


def test_mes_fuera_de_rango_devuelve_422():
    if not _modelo_disponible():
        pytest.skip("Modelos no entrenados")  # sin modelo la dependencia responde 503 antes que la validación
    r = client.post("/predict", json={**PAYLOAD, "mes": 13})
    assert r.status_code == 422


def test_faltan_campos_devuelve_422():
    if not _modelo_disponible():
        pytest.skip("Modelos no entrenados")  # sin modelo la dependencia responde 503 antes que la validación
    r = client.post("/predict", json={"fuente": "EOLICO"})
    assert r.status_code == 422


def test_historial_requiere_tres_valores():
    if not _modelo_disponible():
        pytest.skip("Modelos no entrenados")  # sin modelo la dependencia responde 503 antes que la validación
    r = client.post("/predict/historial", json={**PAYLOAD, "gwh_ultimos_3_meses": [1.0, 2.0]})
    assert r.status_code == 422


def test_historial_rechaza_negativos():
    if not _modelo_disponible():
        pytest.skip("Modelos no entrenados")  # sin modelo la dependencia responde 503 antes que la validación
    r = client.post("/predict/historial", json={**PAYLOAD, "gwh_ultimos_3_meses": [1.0, -2.0, 3.0]})
    assert r.status_code == 422


def test_categoria_desconocida_devuelve_422():
    if not _modelo_disponible():
        pytest.skip("Modelos no entrenados")
    r = client.post("/predict", json={**PAYLOAD, "fuente": "NUCLEAR"})
    assert r.status_code == 422
    assert "no es válido" in r.json()["detail"]


def test_metadata_incluye_aviso_de_uso():
    if not _modelo_disponible():
        pytest.skip("Modelos no entrenados")
    assert "aviso_uso" in client.get("/metadata").json()
