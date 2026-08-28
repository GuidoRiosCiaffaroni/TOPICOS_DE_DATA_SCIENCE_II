"""
Política de actuación ante data drift y deterioro de performance.

El módulo transforma los resultados del monitoreo en una
recomendación operacional auditable.

No reentrena, reemplaza ni elimina automáticamente modelos.
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ORDEN_VENTANAS = {
    "W0": 0,
    "W1": 1,
    "W2": 2,
    "W3": 3,
    "W4": 4
}


NIVEL_ESTADO_DRIFT = {
    "Referencia": 0,
    "Normal": 0,
    "No": 0,
    "Warning": 1,
    "Drift": 2,
    "Sí": 2,
    "Si": 2,
    "Drift significativo": 2
}


# Umbrales de caída relativa de F1 macro.
PERFORMANCE_WARNING = 0.05
PERFORMANCE_CRITICAL = 0.15

EPSILON = 1e-12


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def nivel_drift(
    estado: str
) -> int:
    """
    Convierte una etiqueta de drift en nivel 0, 1 o 2.
    """

    estado = str(
        estado
    ).strip()

    if estado not in NIVEL_ESTADO_DRIFT:

        raise ValueError(
            f"Estado de drift desconocido: {estado}"
        )

    return NIVEL_ESTADO_DRIFT[
        estado
    ]


def evaluar_performance(
    performance_actual: Optional[float],
    performance_referencia: Optional[float]
) -> Dict[str, Any]:
    """
    Evalúa el deterioro de F1 macro respecto de la referencia.
    """

    if (
        performance_actual is None
        or
        performance_referencia is None
        or
        pd.isna(performance_actual)
        or
        pd.isna(performance_referencia)
    ):

        return {
            "nivel_performance": None,
            "estado_performance": "No disponible",
            "delta_f1": None,
            "caida_absoluta": None,
            "caida_relativa": None,
            "caida_porcentual": None
        }

    performance_actual = float(
        performance_actual
    )

    performance_referencia = float(
        performance_referencia
    )

    delta_f1 = (
        performance_actual
        -
        performance_referencia
    )

    caida_absoluta = max(
        0.0,
        performance_referencia
        -
        performance_actual
    )

    caida_relativa = (
        caida_absoluta
        /
        max(
            abs(performance_referencia),
            EPSILON
        )
    )

    if caida_relativa < PERFORMANCE_WARNING:

        nivel = 0
        estado = "Estable"

    elif caida_relativa < PERFORMANCE_CRITICAL:

        nivel = 1
        estado = "Deterioro leve"

    else:

        nivel = 2
        estado = "Deterioro crítico"

    return {
        "nivel_performance": int(nivel),
        "estado_performance": estado,
        "delta_f1": float(delta_f1),
        "caida_absoluta": float(caida_absoluta),
        "caida_relativa": float(caida_relativa),
        "caida_porcentual": float(
            100 * caida_relativa
        )
    }


def detectar_persistencia(
    estados_previos: list,
    k: int = 2
) -> bool:
    """
    Determina si las últimas k ventanas poseen al menos Warning.
    """

    if k < 1:

        raise ValueError(
            "k debe ser mayor o igual que 1."
        )

    if len(estados_previos) < k:

        return False

    ultimos_estados = (
        estados_previos[-k:]
    )

    niveles = [
        nivel_drift(
            estado
        )
        for estado
        in ultimos_estados
    ]

    return all(
        nivel >= 1
        for nivel
        in niveles
    )


def obtener_accion(
    nivel_drift_actual: int,
    nivel_performance: Optional[int],
    persistencia: bool,
    flag_retraining: bool
) -> Dict[str, Any]:
    """
    Aplica la matriz de decisión operacional.
    """

    # --------------------------------------------------------
    # Performance no disponible
    # --------------------------------------------------------

    if nivel_performance is None:

        if nivel_drift_actual == 0:

            return {
                "codigo_accion": "MONITOR",
                "prioridad": "Baja",
                "accion": (
                    "Continuar monitoreando y esperar "
                    "la disponibilidad del target real."
                ),
                "retraining_recomendado": False,
                "rollback_recomendado": False
            }

        if nivel_drift_actual == 1:

            return {
                "codigo_accion": "REVIEW_DATA",
                "prioridad": "Media",
                "accion": (
                    "Realizar revisión manual, verificar "
                    "calidad de datos y reforzar monitoreo."
                ),
                "retraining_recomendado": False,
                "rollback_recomendado": False
            }

        return {
            "codigo_accion": "EVALUATE_RETRAINING",
            "prioridad": "Alta",
            "accion": (
                "Validar el drift significativo y evaluar "
                "reentrenamiento. No promover un modelo "
                "sin disponer de target real."
            ),
            "retraining_recomendado": True,
            "rollback_recomendado": False
        }

    # --------------------------------------------------------
    # Normal + performance estable
    # --------------------------------------------------------

    if (
        nivel_drift_actual == 0
        and
        nivel_performance == 0
    ):

        return {
            "codigo_accion": "MONITOR",
            "prioridad": "Baja",
            "accion": (
                "Continuar monitoreando. No se requiere "
                "intervención."
            ),
            "retraining_recomendado": False,
            "rollback_recomendado": False
        }

    # --------------------------------------------------------
    # Normal + deterioro
    # --------------------------------------------------------

    if (
        nivel_drift_actual == 0
        and
        nivel_performance >= 1
    ):

        return {
            "codigo_accion": "INVESTIGATE_CONCEPT_DRIFT",
            "prioridad": (
                "Alta"
                if nivel_performance == 2
                else "Media"
            ),
            "accion": (
                "Investigar concept drift, calidad del target, "
                "errores de etiquetado y posibles cambios en "
                "la relación entre features y objetivo."
            ),
            "retraining_recomendado": (
                nivel_performance == 2
            ),
            "rollback_recomendado": False
        }

    # --------------------------------------------------------
    # Warning + performance estable
    # --------------------------------------------------------

    if (
        nivel_drift_actual == 1
        and
        nivel_performance == 0
    ):

        return {
            "codigo_accion": (
                "PERSISTENT_REVIEW"
                if persistencia
                else "REINFORCED_MONITORING"
            ),
            "prioridad": (
                "Media"
                if persistencia
                else "Baja"
            ),
            "accion": (
                "Realizar revisión manual y seguimiento "
                "reforzado. Verificar si la señal persiste "
                "en las ventanas siguientes."
            ),
            "retraining_recomendado": False,
            "rollback_recomendado": False
        }

    # --------------------------------------------------------
    # Warning + deterioro
    # --------------------------------------------------------

    if (
        nivel_drift_actual == 1
        and
        nivel_performance >= 1
    ):

        return {
            "codigo_accion": "PREPARE_RETRAINING",
            "prioridad": (
                "Alta"
                if nivel_performance == 2
                else "Media"
            ),
            "accion": (
                "Priorizar el diagnóstico, validar nuevos "
                "datos y preparar una evaluación de "
                "reentrenamiento semiautomático."
            ),
            "retraining_recomendado": (
                nivel_performance == 2
                or
                persistencia
            ),
            "rollback_recomendado": False
        }

    # --------------------------------------------------------
    # Drift significativo + performance estable
    # --------------------------------------------------------

    if (
        nivel_drift_actual == 2
        and
        nivel_performance == 0
    ):

        return {
            "codigo_accion": "VALIDATE_SIGNIFICANT_DRIFT",
            "prioridad": "Media",
            "accion": (
                "Validar el drift significativo, revisar "
                "calidad de datos y evaluar intervención. "
                "No existe evidencia suficiente para rollback."
            ),
            "retraining_recomendado": bool(
                flag_retraining
                or
                persistencia
            ),
            "rollback_recomendado": False
        }

    # --------------------------------------------------------
    # Drift significativo + deterioro leve
    # --------------------------------------------------------

    if (
        nivel_drift_actual == 2
        and
        nivel_performance == 1
    ):

        return {
            "codigo_accion": "EVALUATE_RETRAINING",
            "prioridad": "Alta",
            "accion": (
                "Evaluar reentrenamiento con datos validados "
                "y comparar el candidato contra el modelo "
                "campeón antes de promoverlo."
            ),
            "retraining_recomendado": True,
            "rollback_recomendado": False
        }

    # --------------------------------------------------------
    # Drift significativo + deterioro crítico
    # --------------------------------------------------------

    return {
        "codigo_accion": "EVALUATE_RETRAIN_OR_ROLLBACK",
        "prioridad": "Crítica",
        "accion": (
            "Evaluar reentrenamiento o rollback con prioridad "
            "crítica. Mantener el modelo campeón hasta validar "
            "un candidato o confirmar una versión estable."
        ),
        "retraining_recomendado": True,
        "rollback_recomendado": True
    }


def generar_decision_id(
    ventana: str,
    estado_drift: str,
    codigo_accion: str
) -> str:
    """
    Genera un identificador reproducible para la decisión.
    """

    contenido = (
        f"{ventana}|"
        f"{estado_drift}|"
        f"{codigo_accion}"
    )

    return hashlib.sha256(
        contenido.encode("utf-8")
    ).hexdigest()


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def evaluar_politica_actuacion(
    summary_path: Path = Path(
        "monitoring/window_summary.csv"
    ),
    alerts_path: Path = Path(
        "monitoring/alerts.csv"
    ),
    flag_path: Path = Path(
        "monitoring/retrain_required.flag"
    )
) -> pd.DataFrame:
    """
    Evalúa la política para las ventanas de producción.

    Returns
    -------
    pd.DataFrame
        Una fila por ventana con estado, performance y acción.
    """

    summary_path = Path(
        summary_path
    )

    alerts_path = Path(
        alerts_path
    )

    flag_path = Path(
        flag_path
    )

    if not summary_path.exists():

        raise FileNotFoundError(
            f"No se encontró: {summary_path}"
        )

    df_summary = pd.read_csv(
        summary_path
    )

    columnas_requeridas = {
        "Ventana",
        "Estado_general",
        "Performance"
    }

    faltantes = (
        columnas_requeridas
        -
        set(df_summary.columns)
    )

    if faltantes:

        raise KeyError(
            "window_summary.csv no cumple el contrato. "
            f"Faltan: {sorted(faltantes)}"
        )

    # --------------------------------------------------------
    # Filtrar y ordenar ventanas
    # --------------------------------------------------------

    df_produccion = (
        df_summary[
            df_summary["Ventana"]
            .astype(str)
            .isin(ORDEN_VENTANAS)
        ]
        .copy()
    )

    if df_produccion.empty:

        raise ValueError(
            "No existen ventanas reconocidas."
        )

    df_produccion["_orden"] = (
        df_produccion["Ventana"]
        .astype(str)
        .map(ORDEN_VENTANAS)
    )

    df_produccion = (
        df_produccion
        .sort_values("_orden")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Performance de referencia
    # --------------------------------------------------------

    fila_w0 = df_produccion[
        df_produccion["Ventana"]
        .astype(str)
        .eq("W0")
    ]

    if not fila_w0.empty:

        performance_referencia = float(
            fila_w0["Performance"].iloc[0]
        )

    else:

        # Si W0 no está en window_summary, se usa la primera
        # ventana disponible como referencia operacional.
        performance_referencia = float(
            df_produccion["Performance"].iloc[0]
        )

    # --------------------------------------------------------
    # Alertas por ventana
    # --------------------------------------------------------

    if alerts_path.exists():

        df_alertas = pd.read_csv(
            alerts_path
        )

    else:

        df_alertas = pd.DataFrame()

    # --------------------------------------------------------
    # Cargar flag
    # --------------------------------------------------------

    solicitud_flag = {}

    if flag_path.exists():

        try:

            solicitud_flag = json.loads(
                flag_path.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError:

            solicitud_flag = {}

    ventana_flag = str(
        solicitud_flag.get(
            "ventana",
            ""
        )
    )

    flag_required = bool(
        solicitud_flag.get(
            "required",
            False
        )
    )

    # --------------------------------------------------------
    # Evaluar cada ventana
    # --------------------------------------------------------

    decisiones = []
    estados_acumulados = []

    for _, fila in df_produccion.iterrows():

        ventana = str(
            fila["Ventana"]
        )

        estado_drift = str(
            fila["Estado_general"]
        )

        performance_actual = float(
            fila["Performance"]
        )

        nivel_drift_actual = nivel_drift(
            estado_drift
        )

        estados_acumulados.append(
            estado_drift
        )

        persistencia = detectar_persistencia(
            estados_previos=estados_acumulados,
            k=2
        )

        resultado_performance = evaluar_performance(
            performance_actual=performance_actual,
            performance_referencia=(
                performance_referencia
            )
        )

        flag_ventana = (
            flag_required
            and
            ventana == ventana_flag
        )

        resultado_accion = obtener_accion(
            nivel_drift_actual=(
                nivel_drift_actual
            ),
            nivel_performance=(
                resultado_performance[
                    "nivel_performance"
                ]
            ),
            persistencia=persistencia,
            flag_retraining=flag_ventana
        )

        # ----------------------------------------------------
        # Número de alertas
        # ----------------------------------------------------

        if (
            not df_alertas.empty
            and
            "ventana" in df_alertas.columns
        ):

            alertas_ventana = (
                df_alertas[
                    df_alertas["ventana"]
                    .astype(str)
                    .eq(ventana)
                ]
            )

            n_alertas = int(
                len(alertas_ventana)
            )

            if "severidad" in alertas_ventana.columns:

                n_significativas = int(
                    alertas_ventana[
                        "severidad"
                    ]
                    .astype(str)
                    .eq(
                        "Drift significativo"
                    )
                    .sum()
                )

            else:

                n_significativas = 0

        else:

            n_alertas = 0
            n_significativas = 0

        decision_id = generar_decision_id(
            ventana=ventana,
            estado_drift=estado_drift,
            codigo_accion=(
                resultado_accion[
                    "codigo_accion"
                ]
            )
        )

        decisiones.append(
            {
                "timestamp_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "decision_id":
                    decision_id,

                "ventana":
                    ventana,

                "estado_drift":
                    estado_drift,

                "nivel_drift":
                    int(
                        nivel_drift_actual
                    ),

                "performance_f1_macro":
                    performance_actual,

                "performance_referencia":
                    performance_referencia,

                **resultado_performance,

                "n_alertas":
                    n_alertas,

                "n_alertas_significativas":
                    n_significativas,

                "persistencia_2_ventanas":
                    bool(
                        persistencia
                    ),

                "retrain_flag_activo":
                    bool(
                        flag_ventana
                    ),

                **resultado_accion,

                "revision_humana_requerida":
                    bool(
                        nivel_drift_actual >= 1
                        or
                        (
                            resultado_performance[
                                "nivel_performance"
                            ]
                            is not None
                            and
                            resultado_performance[
                                "nivel_performance"
                            ]
                            >= 1
                        )
                    )
            }
        )

    return pd.DataFrame(
        decisiones
    )
