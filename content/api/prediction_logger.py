
"""
Registro persistente de inferencias.

Cada predicción se almacena en:

    monitoring/predictions.csv

El módulo también permite incorporar posteriormente
el target real utilizando inference_id.
"""

from pathlib import Path
from datetime import datetime, timezone

import csv
import json
import uuid

import pandas as pd

from api.model_loader import (
    INPUT_FEATURES,
    CLASSES,
    MODEL_VERSION,
    SELECTED_MODEL
)


# ========================================================
# 1. RAÍZ DEL PROYECTO
# ========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ========================================================
# 2. DIRECTORIO DE MONITOREO
# ========================================================

MONITORING_DIR = (
    PROJECT_ROOT
    /
    "monitoring"
)


MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ========================================================
# 3. ARCHIVO DE PREDICCIONES
# ========================================================

PREDICTIONS_PATH = (
    MONITORING_DIR
    /
    "predictions.csv"
)


# ========================================================
# 4. COLUMNAS DEL REGISTRO
# ========================================================
#
# Las 15 features se conservan como columnas individuales.
#
# Esto permitirá en Fase 4 calcular directamente:
#
#       PSI
#       KS
#
# para cada feature.
#
# ========================================================

COLUMNAS_REGISTRO = [

    "inference_id",

    "timestamp_utc",

    "selected_model",

    "model_version",

    *INPUT_FEATURES,

    "inputs_json",

    "predicted_class",

    "confidence",

    "probabilities_json",

    "target_real",

    "target_available"
]


# ========================================================
# 5. NORMALIZAR TARGET
# ========================================================

def _validar_target(
    target_real
):
    """
    Valida el target cuando está disponible.

    None significa que todavía no existe etiqueta real.
    """

    if target_real is None:

        return None


    target_real = str(
        target_real
    )


    if target_real not in CLASSES:

        raise ValueError(
            "El target real no pertenece a las "
            "clases conocidas por el modelo."
        )


    return target_real


# ========================================================
# 6. REGISTRAR PREDICCIÓN
# ========================================================

def registrar_prediccion(
    inputs,
    predicted_class,
    confidence,
    probabilities,
    target_real=None,
    inference_id=None
):
    """
    Registra una inferencia en predictions.csv.

    Parameters
    ----------
    inputs : dict
        Diccionario con las 15 features originales.

    predicted_class : str
        Clase predicha por el modelo.

    confidence : float
        Probabilidad máxima de la predicción.

    probabilities : dict
        Probabilidades por clase.

    target_real : str or None
        Clase verdadera cuando esté disponible.

    inference_id : str or None
        Identificador opcional. Si no se entrega,
        se genera automáticamente mediante UUID.

    Returns
    -------
    dict
        Información básica del registro creado.
    """


    # ====================================================
    # 6.1 VALIDAR INPUTS
    # ====================================================

    features_recibidas = set(
        inputs.keys()
    )


    features_esperadas = set(
        INPUT_FEATURES
    )


    faltantes = (
        features_esperadas
        -
        features_recibidas
    )


    extras = (
        features_recibidas
        -
        features_esperadas
    )


    if faltantes:

        raise ValueError(
            "Faltan features para registrar la "
            f"inferencia: {sorted(faltantes)}"
        )


    if extras:

        raise ValueError(
            "Se recibieron features no esperadas: "
            f"{sorted(extras)}"
        )


    # ====================================================
    # 6.2 VALIDAR PREDICCIÓN
    # ====================================================

    predicted_class = str(
        predicted_class
    )


    if predicted_class not in CLASSES:

        raise ValueError(
            "La clase predicha no pertenece a "
            "las clases del modelo."
        )


    # ====================================================
    # 6.3 VALIDAR SCORE
    # ====================================================

    confidence = float(
        confidence
    )


    if not (
        0.0
        <=
        confidence
        <=
        1.0
    ):

        raise ValueError(
            "confidence debe pertenecer al "
            "intervalo [0, 1]."
        )


    # ====================================================
    # 6.4 VALIDAR TARGET REAL
    # ====================================================

    target_real = _validar_target(
        target_real
    )


    # ====================================================
    # 6.5 IDENTIFICADOR ÚNICO
    # ====================================================

    if inference_id is None:

        inference_id = str(
            uuid.uuid4()
        )


    # ====================================================
    # 6.6 TIMESTAMP UTC
    # ====================================================

    timestamp_utc = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


    # ====================================================
    # 6.7 NORMALIZAR INPUTS
    # ====================================================

    inputs_ordenados = {
        feature: float(
            inputs[
                feature
            ]
        )
        for feature
        in INPUT_FEATURES
    }


    # ====================================================
    # 6.8 NORMALIZAR PROBABILIDADES
    # ====================================================

    probabilidades_ordenadas = {
        str(clase): float(
            probabilities[
                str(clase)
            ]
        )
        for clase
        in CLASSES
    }


    # ====================================================
    # 6.9 CONSTRUIR REGISTRO
    # ====================================================

    registro = {

        "inference_id":
            inference_id,

        "timestamp_utc":
            timestamp_utc,

        "selected_model":
            SELECTED_MODEL,

        "model_version":
            MODEL_VERSION
    }


    # ----------------------------------------------------
    # Agregar cada feature como columna individual
    # ----------------------------------------------------

    registro.update(
        inputs_ordenados
    )


    # ----------------------------------------------------
    # Información estructurada
    # ----------------------------------------------------

    registro.update(
        {

            "inputs_json":
                json.dumps(
                    inputs_ordenados,
                    ensure_ascii=False,
                    sort_keys=True
                ),

            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "probabilities_json":
                json.dumps(
                    probabilidades_ordenadas,
                    ensure_ascii=False,
                    sort_keys=True
                ),

            "target_real":
                (
                    target_real
                    if target_real is not None
                    else ""
                ),

            "target_available":
                (
                    target_real
                    is not None
                )
        }
    )


    # ====================================================
    # 6.10 ESCRIBIR CSV
    # ====================================================

    archivo_nuevo = (
        not PREDICTIONS_PATH.exists()
        or
        PREDICTIONS_PATH.stat().st_size == 0
    )


    with open(
        PREDICTIONS_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as archivo:

        writer = csv.DictWriter(
            archivo,
            fieldnames=COLUMNAS_REGISTRO
        )


        if archivo_nuevo:

            writer.writeheader()


        writer.writerow(
            registro
        )


    # ====================================================
    # 6.11 RESULTADO
    # ====================================================

    return {

        "inference_id":
            inference_id,

        "timestamp_utc":
            timestamp_utc,

        "path":
            str(
                PREDICTIONS_PATH
            )
    }


# ========================================================
# 7. INCORPORAR TARGET REAL POSTERIORMENTE
# ========================================================

def actualizar_target_real(
    inference_id,
    target_real
):
    """
    Incorpora la etiqueta verdadera a una inferencia
    previamente almacenada.

    Importante:
    el target se incorpora DESPUÉS de generar la
    predicción.
    """


    # ====================================================
    # 7.1 VALIDAR TARGET
    # ====================================================

    target_real = _validar_target(
        target_real
    )


    if target_real is None:

        raise ValueError(
            "Debe proporcionarse un target real."
        )


    # ====================================================
    # 7.2 VERIFICAR ARCHIVO
    # ====================================================

    if not PREDICTIONS_PATH.exists():

        raise FileNotFoundError(
            "No existe predictions.csv."
        )


    # ====================================================
    # 7.3 CARGAR REGISTROS
    # ====================================================

    df = pd.read_csv(
        PREDICTIONS_PATH,
        dtype={
            "inference_id": "string",
            "target_real": "string"
        }
    )


    # ====================================================
    # 7.3.1 NORMALIZAR TIPOS DE COLUMNAS
    # ====================================================
    #
    # Si target_real contiene inicialmente valores vacíos,
    # Pandas puede inferir float64. Se fuerza StringDtype
    # antes de insertar posteriormente una clase de texto.
    # ====================================================

    if "target_real" not in df.columns:

        df[
            "target_real"
        ] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string"
        )

    else:

        df[
            "target_real"
        ] = (
            df[
                "target_real"
            ]
            .astype(
                "string"
            )
        )


    if "target_available" not in df.columns:

        df[
            "target_available"
        ] = False


    if df.empty:

        raise ValueError(
            "predictions.csv no contiene registros."
        )


    # ====================================================
    # 7.4 LOCALIZAR INFERENCIA
    # ====================================================

    mascara = (
        df[
            "inference_id"
        ]
        .astype(str)
        .eq(
            str(
                inference_id
            )
        )
    )


    if not mascara.any():

        raise KeyError(
            "No se encontró inference_id: "
            f"{inference_id}"
        )


    # ====================================================
    # 7.5 ACTUALIZAR TARGET
    # ====================================================

    df.loc[
        mascara,
        "target_real"
    ] = target_real


    df.loc[
        mascara,
        "target_available"
    ] = True


    # ====================================================
    # 7.6 GUARDAR
    # ====================================================

    df.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8"
    )


    return {

        "inference_id":
            str(
                inference_id
            ),

        "target_real":
            target_real,

        "updated":
            True
    }
