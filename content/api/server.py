
"""
Servidor principal del proyecto MLOps.
"""

from api.app import app

from api.windows import (
    router as windows_router
)


# Evitar duplicar el router si el módulo se recarga.
#
# No se utiliza route.path porque algunas versiones de
# FastAPI incorporan objetos internos (_IncludedRouter)
# que no exponen ese atributo.
#
# OpenAPI contiene las rutas públicas efectivas.
rutas_actuales = set(
    app
    .openapi()
    .get(
        "paths",
        {}
    )
    .keys()
)


if "/windows" not in rutas_actuales:

    app.include_router(
        windows_router
    )


    # Invalidar caché OpenAPI después de agregar rutas.
    app.openapi_schema = None
