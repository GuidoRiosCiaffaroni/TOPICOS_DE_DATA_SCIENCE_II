
# ========================================================
# API DE INFERENCIA — PLANTVILLAGE
# FASE 3 — 7.4 CON REGISTRO PERSISTENTE
# ========================================================

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException,
    status
)

from api.schemas import (
    PredictionRequest
)

from api.model_loader import (
    model,
    INPUT_FEATURES,
    N_INPUT_FEATURES,
    CLASSES,
    N_CLASSES,
    MODEL_VERSION,
    SELECTED_MODEL
)

from api.prediction_logger import (
    registrar_prediccion
)


# ========================================================
# 1. CONFIGURACIÓN DE FASTAPI
# ========================================================

app = FastAPI(
    title=(
        "PlantVillage Tomato "
        "Classification API"
    ),
    description=(
        "Servicio local de inferencia para "
        "clasificación multiclase de enfermedades "
        "foliares del tomate con registro persistente."
    ),
    version="1.1.0"
)


# ========================================================
# 2. FUNCIÓN AUXILIAR — ESTADO DEL MODELO
# ========================================================

def modelo_disponible() -> bool:
    """
    Comprueba que el artefacto cargado posea
    las operaciones necesarias para inferencia.
    """

    return (
        model is not None
        and hasattr(
            model,
            "predict"
        )
        and hasattr(
            model,
            "predict_proba"
        )
    )


# ========================================================
# 3. ENDPOINT GET /health
# ========================================================

@app.get(
    "/health",
    tags=["System"],
    summary=(
        "Verificar disponibilidad del servicio"
    )
)
def health():
    """
    Verifica que el modelo y el contrato de entrada
    estén disponibles para realizar inferencias.
    """

    model_loaded = (
        modelo_disponible()
    )

    schema_ok = (
        N_INPUT_FEATURES
        ==
        15
    )

    classes_ok = (
        N_CLASSES
        ==
        10
    )

    ready = (
        model_loaded
        and schema_ok
        and classes_ok
    )

    return {
        "status": (
            "ok"
            if ready
            else "degraded"
        ),

        "model_loaded": (
            model_loaded
        ),

        "ready_for_prediction": (
            ready
        ),

        "selected_model": (
            SELECTED_MODEL
        ),

        "model_version": (
            MODEL_VERSION
        ),

        "n_features": int(
            N_INPUT_FEATURES
        ),

        "n_classes": int(
            N_CLASSES
        ),

        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    }


# ========================================================
# 4. ENDPOINT POST /predict
# ========================================================

@app.post(
    "/predict",
    tags=["Prediction"],
    summary=(
        "Generar y registrar una predicción"
    )
)
def predict(
    request: PredictionRequest
):
    """
    Recibe las 15 features visuales originales,
    ejecuta el Pipeline empaquetado, registra la
    inferencia y devuelve la predicción.
    """

    # ====================================================
    # 4.1 VERIFICAR MODELO
    # ====================================================

    if not modelo_disponible():

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "El modelo no se encuentra "
                "disponible."
            )
        )


    # ====================================================
    # 4.2 PYDANTIC -> DICCIONARIO
    # ====================================================
    #
    # model_dump() corresponde a Pydantic v2.
    # Se mantiene compatibilidad con dict() si el entorno
    # utilizara una versión anterior.
    # ====================================================

    datos = (
        request.model_dump()
        if hasattr(
            request,
            "model_dump"
        )
        else request.dict()
    )


    # ====================================================
    # 4.3 DICCIONARIO -> DATAFRAME
    # ====================================================

    X_new = pd.DataFrame(
        [
            datos
        ],
        columns=INPUT_FEATURES
    )


    # ====================================================
    # 4.4 CONTROL DEL CONTRATO
    # ====================================================

    if list(
        X_new.columns
    ) != list(
        INPUT_FEATURES
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Las features no coinciden con "
                "el contrato del modelo."
            )
        )


    # ====================================================
    # 4.5 INFERENCIA
    # ====================================================

    try:

        prediccion = (
            model
            .predict(
                X_new
            )[0]
        )

        probabilidades = (
            model
            .predict_proba(
                X_new
            )[0]
        )

    except Exception as error:

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Error durante la inferencia: "
                f"{str(error)}"
            )
        )


    # ====================================================
    # 4.6 VALIDAR VECTOR DE PROBABILIDADES
    # ====================================================

    probabilidades = np.asarray(
        probabilidades,
        dtype=float
    )

    if (
        probabilidades.ndim
        !=
        1
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "El modelo devolvió un vector de "
                "probabilidades con dimensión inválida."
            )
        )

    if (
        len(
            probabilidades
        )
        !=
        N_CLASSES
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "El número de probabilidades no "
                "coincide con el número de clases."
            )
        )


    # ====================================================
    # 4.7 CLASE DE MÁXIMA PROBABILIDAD
    # ====================================================

    indice_max = int(
        np.argmax(
            probabilidades
        )
    )

    clase_max = str(
        CLASSES[
            indice_max
        ]
    )

    confianza = float(
        probabilidades[
            indice_max
        ]
    )


    # ====================================================
    # 4.8 CONTROL DE CONSISTENCIA
    # ====================================================

    if (
        str(
            prediccion
        )
        !=
        clase_max
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Inconsistencia entre predict() "
                "y predict_proba()."
            )
        )


    # ====================================================
    # 4.9 PROBABILIDADES POR CLASE
    # ====================================================

    probabilidades_por_clase = {
        str(clase): float(probabilidad)
        for clase, probabilidad
        in zip(
            CLASSES,
            probabilidades
        )
    }


    # ====================================================
    # 4.10 REGISTRAR INFERENCIA
    # ====================================================
    #
    # El target real todavía no está disponible.
    # Se incorporará posteriormente mediante inference_id.
    # ====================================================

    try:

        registro = registrar_prediccion(
            inputs=datos,
            predicted_class=clase_max,
            confidence=confianza,
            probabilities=probabilidades_por_clase,
            target_real=None
        )

    except Exception as error:

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "La predicción fue calculada, pero "
                "no fue posible registrar la inferencia: "
                f"{str(error)}"
            )
        )


    # ====================================================
    # 4.11 RESPUESTA
    # ====================================================

    return {
        "status": "success",

        "inference_id": (
            registro[
                "inference_id"
            ]
        ),

        "logged": True,

        "predicted_class": (
            clase_max
        ),

        "confidence": (
            confianza
        ),

        "probabilities": (
            probabilidades_por_clase
        ),

        "selected_model": (
            SELECTED_MODEL
        ),

        "model_version": (
            MODEL_VERSION
        ),

        "n_features": int(
            N_INPUT_FEATURES
        ),

        "timestamp_utc": (
            registro[
                "timestamp_utc"
            ]
        )
    }
