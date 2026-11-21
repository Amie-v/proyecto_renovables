"""Esquemas (payloads JSON) de la API, validados con Pydantic."""
from pydantic import BaseModel, Field, field_validator


class SolicitudEstatica(BaseModel):
    """Predicción para una central SIN historial propio (por ejemplo, una central nueva)."""
    fuente: str = Field(description="EOLICO, SOLAR, HIDRO <=50MW, BIOGAS o BIOMASA")
    region: str = Field(description="Región del MEM, por ejemplo PATAGONIA")
    provincia: str = Field(description="Provincia, por ejemplo CHUBUT")
    programa: str = Field(description="Programa: RENOVAR, MATER, RESTO, etc.")
    mes: int = Field(ge=1, le=12, description="Mes del año a predecir (1-12)")
    antiguedad_meses: int = Field(ge=0, le=400, description="Meses desde que la central empezó a generar")

    model_config = {"json_schema_extra": {"examples": [{
        "fuente": "EOLICO", "region": "PATAGONIA", "provincia": "CHUBUT",
        "programa": "RENOVAR", "mes": 10, "antiguedad_meses": 24}]}}


class SolicitudHistorial(SolicitudEstatica):
    """Predicción para una central CON historial: se suman sus 3 meses previos."""
    gwh_ultimos_3_meses: list[float] = Field(
        min_length=3, max_length=3,
        description="GWh generados en [mes-1, mes-2, mes-3], en ese orden")

    @field_validator("gwh_ultimos_3_meses")
    @classmethod
    def _no_negativos(cls, v):
        if any(x < 0 for x in v):
            raise ValueError("Los GWh no pueden ser negativos")
        return v

    model_config = {"json_schema_extra": {"examples": [{
        "fuente": "EOLICO", "region": "PATAGONIA", "provincia": "CHUBUT",
        "programa": "RENOVAR", "mes": 10, "antiguedad_meses": 24,
        "gwh_ultimos_3_meses": [14.2, 13.1, 15.8]}]}}


class Prediccion(BaseModel):
    prediccion_gwh: float = Field(description="Generación mensual esperada [GWh]")
    rango_gwh: list[float] = Field(description="Rango aproximado (percentiles 10-90 del error histórico)")
    escenario: str
    advertencias: list[str]
    entrenado_utc: str
