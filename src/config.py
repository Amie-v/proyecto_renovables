"""Configuración central del proyecto: rutas, columnas y parámetros de validación."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATOS_DIR = RAIZ / "data"
MODELOS_DIR = RAIZ / "models"

RUTA_DATOS_LIMPIOS = DATOS_DIR / "renovables_limpio.csv"
RUTA_MEJORES_PARAMS = MODELOS_DIR / "best_params.json"
RUTA_METADATA = MODELOS_DIR / "metadata.json"
RUTA_MODELO = {
    "estatico": MODELOS_DIR / "modelo_estatico.joblib",
    "historial": MODELOS_DIR / "modelo_historial.joblib",
}

# Fuentes que se excluyen del modelo (ver hallazgo 7 del EDA: un único episodio en 2012).
FUENTES_EXCLUIDAS = ("BIODIESEL",)

# Variables del modelo
CATEGORICAS = ["fuente", "region", "provincia", "programa"]
NUM_ESTATICAS = ["mes_sin", "mes_cos", "antiguedad_meses"]
NUM_HISTORIAL = ["log_lag_1m", "log_media_3m"]
FEATURES = {
    "estatico": CATEGORICAS + NUM_ESTATICAS,
    "historial": CATEGORICAS + NUM_ESTATICAS + NUM_HISTORIAL,
}

# Validación temporal: se entrena con todo lo anterior a TEST_INICIO y se testea desde ahí.
TEST_INICIO = "2025-01-01"
# Año de validación (anterior al test) con el que se calibra el rango de incertidumbre de la API.
VALIDACION_INICIO = "2024-01-01"
# Folds de Optuna: ventanas expansivas (entrena con lo anterior a cada año y valida en ese año).
ANIOS_VALIDACION_OPTUNA = [2022, 2023, 2024]

SEMILLA = 42

# Parámetros por defecto del Random Forest (se reemplazan por best_params.json si existe).
PARAMS_RF_DEFECTO = {
    "n_estimators": 150,
    "max_depth": 12,        # acotado a propósito: con 20 el modelo pesaba >100 MB sin mejorar el error
    "min_samples_leaf": 5,
    "max_features": 0.5,
}
