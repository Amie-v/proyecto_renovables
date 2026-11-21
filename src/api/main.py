"""API REST para predecir generación renovable mensual.

Uso (desde la raíz del proyecto):
    uvicorn src.api.main:app --reload
Documentación interactiva: http://127.0.0.1:8000/docs
"""
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from ..predictor import EntradaInvalida, ModeloNoDisponible, Predictor
from .schemas import Prediccion, SolicitudEstatica, SolicitudHistorial

app = FastAPI(
    title="API de generación renovable (Argentina)",
    description="Predice GWh mensuales por central a partir de datos abiertos del sector eléctrico. "
                "Proyecto educativo: leé /metadata para conocer sus límites.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def _cargar_predictor() -> Predictor:
    return Predictor()


def get_predictor() -> Predictor:
    try:
        return _cargar_predictor()
    except ModeloNoDisponible as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
def health():
    """Chequeo de vida del servicio."""
    return {"estado": "ok"}


@app.get("/metadata")
def metadata(p: Predictor = Depends(get_predictor)):
    """Información del modelo: métricas de test, fechas, límites y aviso de uso."""
    m = p.metadata
    return {
        "entrenado_utc": m["entrenado_utc"],
        "datos": {"desde": m["datos_desde"], "hasta": m["datos_hasta"], "test_desde": m["test_inicio"]},
        "fuentes_excluidas": m["fuentes_excluidas"],
        "metricas_test": {k: {"modelo": v["metricas_test"], "baseline": v["metricas_baseline"]}
                          for k, v in m["escenarios"].items()},
        "aviso_uso": m["aviso_uso"],
    }


@app.get("/categorias")
def categorias(p: Predictor = Depends(get_predictor)):
    """Valores válidos para fuente, región, provincia y programa."""
    return p.metadata["categorias_validas"]


def _predecir(p: Predictor, solicitud, escenario: str) -> dict:
    try:
        return p.predecir(solicitud.model_dump(), escenario)
    except EntradaInvalida as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict", response_model=Prediccion)
def predict(solicitud: SolicitudEstatica, p: Predictor = Depends(get_predictor)):
    """Predicción para una central sin historial (usa solo fuente, ubicación, programa, mes y antigüedad)."""
    return _predecir(p, solicitud, "estatico")


@app.post("/predict/historial", response_model=Prediccion)
def predict_historial(solicitud: SolicitudHistorial, p: Predictor = Depends(get_predictor)):
    """Predicción para una central con historial: agrega los GWh de los 3 meses previos."""
    return _predecir(p, solicitud, "historial")
