"""Lógica de predicción independiente de FastAPI (se puede probar sin levantar el servidor)."""
import json

import joblib
import numpy as np

from . import config
from .modelos import a_gwh
from .preprocessing import construir_fila

ESCENARIOS = ("estatico", "historial")


class ModeloNoDisponible(RuntimeError):
    """Faltan los archivos del modelo: hay que correr `python -m src.train`."""


class EntradaInvalida(ValueError):
    """La solicitud tiene valores que el modelo no conoce."""


class Predictor:
    def __init__(self, modelos_dir=None):
        modelos_dir = modelos_dir or config.MODELOS_DIR
        ruta_meta = modelos_dir / config.RUTA_METADATA.name
        if not ruta_meta.exists():
            raise ModeloNoDisponible("No encuentro models/metadata.json. Ejecutá: python -m src.train")
        self.metadata = json.loads(ruta_meta.read_text(encoding="utf-8"))
        self.modelos = {}
        for esc in ESCENARIOS:
            ruta = modelos_dir / config.RUTA_MODELO[esc].name
            if not ruta.exists():
                raise ModeloNoDisponible(f"Falta {ruta.name}. Ejecutá: python -m src.train")
            self.modelos[esc] = joblib.load(ruta)
        self._vistas = set(self.metadata["combinaciones_vistas"])

    # ---- validación ----------------------------------------------------------------
    def validar(self, entrada: dict) -> None:
        """Lanza EntradaInvalida si una categoría no existe en el entrenamiento."""
        errores = []
        for campo, validos in self.metadata["categorias_validas"].items():
            valor = str(entrada[campo]).upper().strip()
            if valor not in validos:
                errores.append(f"{campo}='{entrada[campo]}' no es válido. Opciones: {validos}")
        if errores:
            raise EntradaInvalida(" | ".join(errores))

    def advertencias(self, entrada: dict) -> list[str]:
        avisos = []
        clave = "|".join(str(entrada[c]).upper().strip() for c in ("fuente", "region", "provincia"))
        if clave not in self._vistas:
            avisos.append("La combinación fuente/región/provincia no aparece en los datos de entrenamiento: "
                          "la predicción es una extrapolación poco confiable.")
        if entrada["antiguedad_meses"] > self.metadata["antiguedad_max_vista"]:
            avisos.append("La antigüedad supera la máxima vista en el entrenamiento.")
        return avisos

    # ---- predicción ----------------------------------------------------------------
    def predecir(self, entrada: dict, escenario: str) -> dict:
        if escenario not in ESCENARIOS:
            raise EntradaInvalida(f"Escenario desconocido: {escenario}")
        self.validar(entrada)
        fila = construir_fila(entrada, con_historial=(escenario == "historial"))
        pred_log = float(self.modelos[escenario].predict(fila)[0])

        cuantiles = self.metadata["escenarios"][escenario]["cuantiles_residuo_log"]
        q10, q90 = cuantiles.get(str(entrada["fuente"]).upper().strip(), cuantiles["_global"])
        rango = [float(a_gwh(pred_log + q10)), float(a_gwh(pred_log + q90))]

        return {
            "prediccion_gwh": round(float(a_gwh(pred_log)), 4),
            "rango_gwh": [round(rango[0], 4), round(rango[1], 4)],
            "escenario": escenario,
            "advertencias": self.advertencias(entrada),
            "entrenado_utc": self.metadata["entrenado_utc"],
        }
