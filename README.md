<<<<<<< HEAD
# Predicción de generación renovable en Argentina

Proyecto final de Data Science (Programa Energía Digital 2026). Analiza la generación mensual de centrales renovables del Mercado Eléctrico Mayorista (2011–2026), entrena modelos de Machine Learning y los expone mediante una API con FastAPI.

> **Pregunta:** ¿se puede predecir la generación mensual (GWh) de una central renovable a partir de su fuente, ubicación, programa de licitación y estacionalidad? ¿Qué tan bien funciona según la región?
>
> **Respuesta corta:** para una central **nueva** (sin historial) el modelo mejora al baseline, pero con un error todavía alto (~52 % WAPE). Para una central **existente**, casi todo el valor está en lo que generó el mes anterior: el modelo mejora apenas ~15 % a una persistencia trivial.

## Contenido

```
├── README.md
├── requirements.txt
├── data/                      # Excel original + CSV limpios (los genera el notebook 01)
├── notebooks/
│   ├── 01_eda.ipynb           # Análisis exploratorio y limpieza (pre-entrega 2)
│   ├── 02_modelos.ipynb       # Baselines, Ridge, Random Forest + Optuna, error por subgrupo
│   └── 03_clustering.ipynb    # K-means, DBSCAN, PCA/SVD: perfiles de centrales
├── src/
│   ├── config.py              # rutas, variables, partición temporal
│   ├── preprocessing.py       # carga, features, rezagos, split temporal
│   ├── modelos.py             # pipelines scikit-learn y baseline
│   ├── evaluacion.py          # métricas y error por subgrupo
│   ├── train.py               # entrena y serializa los modelos
│   ├── predictor.py           # lógica de predicción (independiente de FastAPI)
│   └── api/
│       ├── main.py            # endpoints FastAPI
│       └── schemas.py         # payloads JSON (Pydantic)
├── tests/                     # test_predictor.py, test_api.py
├── models/                    # modelos .joblib + metadata.json (se regeneran)
└── docs/ia_copiloto.md        # evidencia de uso de IA
```

## 1. Origen del dataset

- **Archivo:** `Energía_Renovables_-_Base_de_Datos_2026-09.xlsx`, hoja **"Base de Datos"**.
- **Organismo / URL de descarga / licencia:** *(completar con los datos del portal donde lo descargaste)*.
- **Qué contiene:** generación mensual en GWh por máquina de cada central renovable, de enero 2011 a septiembre 2026. Variables: central, máquina, fuente (eólica, solar, hidro ≤50 MW, biogás, biomasa, biodiésel), región, provincia, mes, GWh y programa de licitación (`Nueva Generación`).
- **Tamaño:** 23.332 filas originales; tras la limpieza quedan 22.798 filas y 289 máquinas (22.611 filas al excluir biodiésel para el modelo).
- **Qué NO contiene:** potencia instalada ni variables climáticas. Es la principal limitación del proyecto.
- Las otras tres hojas del Excel son resúmenes con fórmulas calculados desde la hoja principal; no se usan.

## 2. Hallazgos del EDA (`notebooks/01_eda.ipynb`)

| Hallazgo | Decisión |
|---|---|
| 189 filas de *Demanda MEM* mezcladas con la generación (inflaban el máximo de ~168 a ~14.000 GWh) | Separadas como serie de contexto |
| 5 pares máquina-mes duplicados (uno exacto, el resto con valores distintos) | Se elimina el exacto y se conserva el mayor (supuesto no verificable con la fuente) |
| ~10 % de ceros: 340 son de pre-operación y ~1.950 operativos | Se excluyen los primeros; se conservan los segundos |
| `AÑO` era una fórmula de Excel; nombres con espacios y mayúsculas mezcladas; programa `RENOVAR 202` posiblemente truncado | Normalizado; `RENOVAR 202` queda como categoría aparte |
| El parque pasó de ~50 máquinas (2011) a ~280 (2026); la participación sobre la demanda sube de ~1-2 % a ~19 % | Panel desbalanceado: validación temporal |
| La eólica aporta ~65 % de lo generado; Patagonia + Buenos Aires concentran ~56 % | Se mide el error por fuente y región |
| Región y fuente casi se confunden (Patagonia ≈ 96 % eólica, Litoral ≈ 99 % biogás) | Cuidado al interpretar importancias |
| Biodiésel: casi todo viene de 2 máquinas durante 12 meses de 2012 | Excluido del modelo |
| Estacionalidad: hidro (índice ~0,5 en junio-julio) y solar muy estacionales; eólica casi plana a nivel mensual | `mes` se codifica cíclicamente |
| La correlación con el mes anterior es 0,96 global, pero ~0,64 dentro de cada máquina | La correlación alta mide sobre todo *tamaño* de la central |

## 3. Modelo (especificaciones técnicas)

| Aspecto | Detalle |
|---|---|
| Objetivo | `log1p(GWh)`; las métricas se reportan en GWh |
| Escenario **estático** (central nueva) | `fuente, región, provincia, programa, mes (sin/cos), antigüedad_meses` |
| Escenario **con historial** | lo anterior + `log1p(GWh mes-1)` + `log1p(media de los 3 meses previos)` |
| Modelos | Baseline (media por fuente×mes), persistencia (solo con historial), Ridge, Random Forest |
| Preprocesamiento | One-hot (categorías) + estandarización (solo Ridge) en un `Pipeline` de scikit-learn |
| Validación | **Temporal**: entrena con < 2025, testea 2025-01 a 2026-09. Folds de Optuna: ventanas expansivas validando en 2022, 2023 y 2024 |
| Hiperparámetros | Optuna (TPE, 25 pruebas por escenario): `n_estimators` 100-250, `max_depth` 6-14, `min_samples_leaf` 2-20, `max_features` 0,3-1,0. Se guardan en `models/best_params.json` |
| Métrica principal | **WAPE** = Σ\|error\| / Σ real (más estable que MAPE con valores cercanos a cero) |
| Semilla | 42 |

**Resultados en test** *(valores de referencia con hiperparámetros por defecto; reemplazalos con los de tu corrida de Optuna)*:

| Escenario | Modelo | MAE (GWh) | WAPE |
|---|---|---|---|
| Estático | Baseline fuente×mes | 6,26 | 71,3 % |
| Estático | Random Forest | 4,58 | 52,2 % |
| Con historial | Baseline fuente×mes | 6,30 | 70,9 % |
| Con historial | Persistencia (mes anterior) | 1,37 | 15,4 % |
| Con historial | Random Forest | 1,18 | 13,2 % |

Observaciones: con historial, el R² (escala log) de 0,97 es engañoso porque la persistencia ya logra 0,96; lo relevante es la ganancia sobre ella. Detalle por subgrupo, calibración del rango e importancia de variables: ver `02_modelos.ipynb`.

**Clustering** (`03_clustering.ipynb`): K-means (k = 3, silhouette ≈ 0,33) separa un grupo de perfil plano (eólica y biogás), uno muy estacional (solar e hidro) y uno pequeño de 4 centrales de biomasa del Noroeste con generación concentrada de junio a octubre. La fuente no determina el grupo.

## 4. Diseño de la API (FastAPI)

La lógica de predicción vive en `src/predictor.py` (se prueba sin servidor); `src/api/main.py` solo expone los endpoints.

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Chequeo de vida |
| GET | `/metadata` | Fechas, métricas de test, fuentes excluidas y aviso de uso |
| GET | `/categorias` | Valores válidos de fuente, región, provincia y programa |
| POST | `/predict` | Central **sin historial** (modelo estático) |
| POST | `/predict/historial` | Central **con historial** (agrega 3 meses previos) |

**Ejemplo** — `POST /predict`

```json
{ "fuente": "EOLICO", "region": "PATAGONIA", "provincia": "CHUBUT",
  "programa": "RENOVAR", "mes": 10, "antiguedad_meses": 24 }
```

Respuesta:

```json
{ "prediccion_gwh": 14.9941,
  "rango_gwh": [7.2108, 43.2239],
  "escenario": "estatico",
  "advertencias": [],
  "entrenado_utc": "2026-11-20T06:38:27+00:00" }
```

`POST /predict/historial` agrega `"gwh_ultimos_3_meses": [14.2, 13.1, 15.8]` (meses −1, −2, −3) y devuelve un rango mucho más angosto (≈ 11,7–18,3 GWh con ese ejemplo).

**Decisiones de diseño**
- `rango_gwh` son los percentiles 10-90 del error de validación (2024), no un intervalo estadístico formal.
- `advertencias` avisa cuando la combinación fuente/región/provincia nunca apareció en el entrenamiento (extrapolación).
- **422** si falta un campo, el mes no está entre 1 y 12, hay GWh negativos o una categoría no existe. **503** si todavía no se entrenaron los modelos.
- El modelo se carga una sola vez y se reutiliza entre solicitudes.

## 5. Sesgos, límites y uso responsable

- **Sin potencia ni clima.** El modelo no puede separar "central grande" de "buen recurso". En centrales nuevas el error es alto y el rango de la API, muy ancho.
- **Las centrales nuevas son el punto débil.** En test, el WAPE estático pasa de ~49 % (máquinas vistas) a ~73 % (máquinas nuevas).
- **Subestimación sistemática** en eólica y en Buenos Aires/Patagonia (sesgo negativo). Hipótesis, no verificada: los parques recientes son más grandes que los históricos.
- **Desempeño desigual por región.** Con el modelo estático el WAPE va de ~19 % (Comahue) a ~89 % (Cuyo). Litoral (6 máquinas) y Gran Bs.As. (8) tienen pocos datos: sus métricas son inestables.
- **Sesgo de supervivencia geográfico.** Solo hay centrales que se construyeron, en zonas donde ya se decidió invertir. El modelo **no sirve para decir dónde conviene instalar**.
- **Confusión región-fuente.** La importancia de variables describe lo que el modelo *usa*, no lo que *causa* la generación.
- **Cambio de distribución.** El parque creció casi seis veces; tecnologías, regulaciones o precios nuevos pueden degradar el modelo.
- **Supuestos de limpieza** (duplicados, ceros, `RENOVAR 202`) documentados y no verificables con la fuente.
- **Uso indebido a evitar:** decisiones de inversión, despacho o regulatorias sin validación de especialistas. Es un proyecto educativo.
- **Datos personales:** el dataset no contiene datos de personas; los riesgos son técnicos y de mal uso de las predicciones.

## 6. Uso de IA copiloto

Se usó un asistente de IA para explorar el archivo, proponer el diseño y generar código. El registro de qué se pidió, qué se aceptó y qué se corrigió está en [`docs/ia_copiloto.md`](docs/ia_copiloto.md).

## 7. Cómo reproducir

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Copiá el Excel a data/ y ejecutá notebooks/01_eda.ipynb  (genera data/renovables_limpio.csv)
# 2) Ejecutá notebooks/02_modelos.ipynb                          (genera models/best_params.json)
python -m src.train                                              # entrena y guarda los modelos
pytest -q                                                        # tests
uvicorn src.api.main:app --reload                                # API → http://127.0.0.1:8000/docs
```

Probar con `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"fuente":"EOLICO","region":"PATAGONIA","provincia":"CHUBUT","programa":"RENOVAR","mes":10,"antiguedad_meses":24}'
```

## 8. Próximos pasos

- Sumar **viento y radiación** por provincia (por ejemplo, NASA POWER) y la **potencia instalada**.
- Pronóstico a varios meses y modelos de series temporales por central.
- *(Opcional)* Un RAG con LangChain sobre el diccionario de variables.
=======
# proyecto_renovables
>>>>>>> 120190e39fdfa09a97e729d0077b979ca59fae73
