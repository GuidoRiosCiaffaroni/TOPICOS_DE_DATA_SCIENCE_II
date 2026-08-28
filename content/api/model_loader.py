
"""
Carga centralizada del modelo de producción.

El modelo se carga una sola vez durante la importación
de este módulo.

Posteriormente api/app.py podrá utilizar:

    from api.model_loader import model
"""

from pathlib import Path

import hashlib
import json

import joblib
import numpy as np


# ========================================================
# 1. RAÍZ DEL PROYECTO
# ========================================================
#
# model_loader.py está ubicado en:
#
#       proyecto/api/model_loader.py
#
# parents[1] corresponde a:
#
#       proyecto/
#
# Esto evita depender del directorio de trabajo desde el
# cual se ejecute uvicorn.
#
# ========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ========================================================
# 2. RUTAS DE LOS ARTEFACTOS
# ========================================================

MODEL_PATH = (
    PROJECT_ROOT
    /
    "models"
    /
    "model.joblib"
)


METRICS_PATH = (
    PROJECT_ROOT
    /
    "models"
    /
    "train_metrics.json"
)


# ========================================================
# 3. VALIDAR EXISTENCIA
# ========================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "No se encontró el modelo empaquetado: "
        f"{MODEL_PATH}"
    )


if not METRICS_PATH.exists():

    raise FileNotFoundError(
        "No se encontró train_metrics.json: "
        f"{METRICS_PATH}"
    )


# ========================================================
# 4. CARGAR METADATOS
# ========================================================

with open(
    METRICS_PATH,
    "r",
    encoding="utf-8"
) as archivo:

    metadata = json.load(
        archivo
    )


# ========================================================
# 5. IMPORTAR TRANSFORMER PERSONALIZADO
# ========================================================
#
# model.joblib contiene una referencia a:
#
# src.transformers.FeatureEngineeringProduccion
#
# La importación explícita garantiza que el módulo esté
# disponible cuando joblib reconstruya el Pipeline.
#
# ========================================================

from src.transformers import (
    FeatureEngineeringProduccion
)


# ========================================================
# 6. FUNCIÓN SHA-256
# ========================================================

def calcular_sha256(
    ruta
):
    """
    Calcula SHA-256 de un archivo.
    """

    hash_sha256 = hashlib.sha256()


    with open(
        ruta,
        "rb"
    ) as archivo:

        for bloque in iter(
            lambda:
            archivo.read(
                1024 * 1024
            ),
            b""
        ):

            hash_sha256.update(
                bloque
            )


    return hash_sha256.hexdigest()


# ========================================================
# 7. VALIDAR INTEGRIDAD DEL ARTEFACTO
# ========================================================

hash_actual = calcular_sha256(
    MODEL_PATH
)


hash_registrado = (
    metadata
    .get(
        "artifact",
        {}
    )
    .get(
        "model_sha256"
    )
)


if (
    hash_registrado
    is not None
    and
    hash_actual
    !=
    hash_registrado
):

    raise RuntimeError(
        "La huella SHA-256 de model.joblib "
        "no coincide con train_metrics.json. "
        "El artefacto puede haber sido modificado."
    )


# ========================================================
# 8. CARGAR MODELO
# ========================================================
#
# Operación central solicitada en 7.2:
#
# model = joblib.load("models/model.joblib")
#
# ========================================================

model = joblib.load(
    MODEL_PATH
)


# ========================================================
# 9. VALIDAR INTERFAZ DEL MODELO
# ========================================================

if not hasattr(
    model,
    "predict"
):

    raise TypeError(
        "El artefacto cargado no implementa predict()."
    )


if not hasattr(
    model,
    "predict_proba"
):

    raise TypeError(
        "El artefacto cargado no implementa "
        "predict_proba()."
    )


# ========================================================
# 10. VALIDAR QUE SEA PIPELINE
# ========================================================

if not hasattr(
    model,
    "named_steps"
):

    raise TypeError(
        "El artefacto esperado debe ser "
        "un sklearn Pipeline."
    )


pasos_requeridos = {
    "preprocessor",
    "model"
}


pasos_disponibles = set(
    model.named_steps.keys()
)


if not pasos_requeridos.issubset(
    pasos_disponibles
):

    raise TypeError(
        "El Pipeline cargado no posee "
        "los pasos esperados."
    )


# ========================================================
# 11. INFORMACIÓN DEL ESTIMADOR FINAL
# ========================================================

estimator = (
    model
    .named_steps[
        "model"
    ]
)


MODEL_CLASS = (
    estimator
    .__class__
    .__name__
)


MODEL_VERSION = (
    metadata
    .get(
        "artifact",
        {}
    )
    .get(
        "model_version",
        "unknown"
    )
)


SELECTED_MODEL = (
    metadata
    .get(
        "model",
        {}
    )
    .get(
        "selected_model",
        MODEL_CLASS
    )
)


# ========================================================
# 12. FEATURES ESPERADAS
# ========================================================

INPUT_FEATURES = (
    metadata
    .get(
        "features",
        {}
    )
    .get(
        "input",
        []
    )
)


N_INPUT_FEATURES = len(
    INPUT_FEATURES
)


if N_INPUT_FEATURES != 15:

    raise RuntimeError(
        "El modelo empaquetado debería registrar "
        "15 features de entrada."
    )


# ========================================================
# 13. CLASES
# ========================================================

CLASSES = [
    str(clase)
    for clase
    in estimator.classes_
]


N_CLASSES = len(
    CLASSES
)


if N_CLASSES != 10:

    raise RuntimeError(
        "Se esperaban 10 clases en el modelo."
    )


# ========================================================
# 14. ESTADO DE CARGA
# ========================================================

MODEL_LOADED = True
