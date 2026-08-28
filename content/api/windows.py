
"""
Endpoints de inspección de ventanas de producción.

Permite consultar:

    /windows
    /windows/W0
    /windows/W1
    /windows/W2
    /windows/W3
    /windows/W4
"""

from pathlib import Path

import json
import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
    Query
)


# ========================================================
# 1. ROUTER
# ========================================================

router = APIRouter(
    tags=[
        "Production Windows"
    ]
)


# ========================================================
# 2. RUTAS
# ========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


WINDOWS_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "production"
    /
    "windows"
)


METADATA_PATH = (
    WINDOWS_DIR
    /
    "metadata.json"
)


# ========================================================
# 3. CARGAR METADATOS
# ========================================================

def cargar_metadata():

    if not METADATA_PATH.exists():

        raise HTTPException(
            status_code=503,
            detail=(
                "No existen metadatos de ventanas. "
                "Ejecute primero la sección 8.1."
            )
        )


    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as archivo:

        return json.load(
            archivo
        )


# ========================================================
# 4. CARGAR UNA VENTANA
# ========================================================

def obtener_ventana(
    nombre,
    limit
):

    nombre = nombre.upper()


    if nombre not in {
        "W0",
        "W1",
        "W2",
        "W3",
        "W4"
    }:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Ventana desconocida: {nombre}"
            )
        )


    path = (
        WINDOWS_DIR
        /
        f"{nombre}.csv"
    )


    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                f"No existe el archivo {path.name}"
            )
        )


    df = pd.read_csv(
        path
    )


    metadata = (
        cargar_metadata()
        .get(
            nombre,
            {}
        )
    )


    return {

        "window":
            nombre,

        "interpretation":
            metadata.get(
                "interpretacion"
            ),

        "status":
            metadata.get(
                "estado"
            ),

        "n_observations":
            int(
                len(
                    df
                )
            ),

        "n_columns":
            int(
                df.shape[1]
            ),

        "columns":
            list(
                df.columns
            ),

        "showing":
            int(
                min(
                    limit,
                    len(
                        df
                    )
                )
            ),

        "data":
            df
            .head(
                limit
            )
            .where(
                pd.notna(
                    df.head(
                        limit
                    )
                ),
                None
            )
            .to_dict(
                orient="records"
            )
    }


# ========================================================
# 5. RESUMEN GENERAL
# ========================================================

@router.get(
    "/windows",
    summary="Resumen de las ventanas"
)
def windows_summary():

    return {
        "windows":
            cargar_metadata()
    }


# ========================================================
# 6. W0
# ========================================================

@router.get(
    "/windows/W0",
    summary="W0 — Referencia"
)
def window_w0(
    limit: int = Query(
        10,
        ge=1,
        le=100
    )
):

    return obtener_ventana(
        "W0",
        limit
    )


# ========================================================
# 7. W1
# ========================================================

@router.get(
    "/windows/W1",
    summary="W1 — Producción estable"
)
def window_w1(
    limit: int = Query(
        10,
        ge=1,
        le=100
    )
):

    return obtener_ventana(
        "W1",
        limit
    )


# ========================================================
# 8. W2
# ========================================================

@router.get(
    "/windows/W2",
    summary="W2 — Drift leve"
)
def window_w2(
    limit: int = Query(
        10,
        ge=1,
        le=100
    )
):

    return obtener_ventana(
        "W2",
        limit
    )


# ========================================================
# 9. W3
# ========================================================

@router.get(
    "/windows/W3",
    summary="W3 — Drift moderado"
)
def window_w3(
    limit: int = Query(
        10,
        ge=1,
        le=100
    )
):

    return obtener_ventana(
        "W3",
        limit
    )


# ========================================================
# 10. W4
# ========================================================

@router.get(
    "/windows/W4",
    summary="W4 — Drift significativo"
)
def window_w4(
    limit: int = Query(
        10,
        ge=1,
        le=100
    )
):

    return obtener_ventana(
        "W4",
        limit
    )
