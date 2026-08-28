"""
Módulo para materializar una solicitud semiautomática de
reentrenamiento cuando el monitoreo detecta drift significativo.

Flujo:
    drift.py
        -> monitoring/alerts.csv
        -> monitoring/retrain_required.flag
        -> src/retrain.py

El módulo NO entrena ni reemplaza automáticamente el modelo.
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import hashlib
import json
import os
import tempfile

import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ORDEN_VENTANAS = {
    "W1": 1,
    "W2": 2,
    "W3": 3,
    "W4": 4
}

SEVERIDAD_TRIGGER = "Drift significativo"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def _guardar_json_atomico(
    contenido: Dict[str, Any],
    ruta_destino: Path
) -> None:
    """
    Guarda un documento JSON mediante reemplazo atómico.

    Primero se escribe un archivo temporal y luego se reemplaza
    el destino. Esto reduce el riesgo de dejar un flag incompleto
    si la ejecución se interrumpe.
    """

    ruta_destino = Path(ruta_destino)

    ruta_destino.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    descriptor, ruta_temporal = tempfile.mkstemp(
        prefix="retrain_request_",
        suffix=".tmp",
        dir=str(ruta_destino.parent)
    )

    try:

        with os.fdopen(
            descriptor,
            mode="w",
            encoding="utf-8"
        ) as archivo:

            json.dump(
                contenido,
                archivo,
                ensure_ascii=False,
                indent=4,
                sort_keys=True
            )

        os.replace(
            ruta_temporal,
            ruta_destino
        )

    except Exception:

        ruta_temporal_path = Path(
            ruta_temporal
        )

        if ruta_temporal_path.exists():
            ruta_temporal_path.unlink()

        raise


def _identificar_ultima_ventana(
    df_alertas: pd.DataFrame
) -> str:
    """
    Identifica la ventana de producción más reciente.
    """

    ventanas = (
        df_alertas["ventana"]
        .astype(str)
        .map(ORDEN_VENTANAS)
    )

    mascara_valida = ventanas.notna()

    if not mascara_valida.any():

        raise ValueError(
            "alerts.csv no contiene ventanas válidas W1-W4."
        )

    orden_maximo = int(
        ventanas.loc[mascara_valida].max()
    )

    for ventana, orden in ORDEN_VENTANAS.items():

        if orden == orden_maximo:
            return ventana

    raise RuntimeError(
        "No fue posible identificar la última ventana."
    )


def _construir_request_id(
    ventana: str,
    alert_keys: list
) -> str:
    """
    Genera un identificador reproducible para la solicitud.
    """

    contenido = (
        str(ventana)
        +
        "|"
        +
        "|".join(
            sorted(
                map(str, alert_keys)
            )
        )
    )

    return hashlib.sha256(
        contenido.encode("utf-8")
    ).hexdigest()


def _valor_float_o_none(
    valor: Any
) -> Optional[float]:
    """
    Convierte un valor a float o devuelve None.
    """

    if pd.isna(valor):
        return None

    return float(valor)


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def materializar_gatillo_retraining(
    alerts_path: Path = Path("monitoring/alerts.csv"),
    flag_path: Path = Path(
        "monitoring/retrain_required.flag"
    ),
    model_path: Path = Path("models/model.joblib"),
    metrics_path: Path = Path(
        "models/train_metrics.json"
    ),
    ventana: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evalúa las alertas y crea retrain_required.flag cuando la
    última ventana presenta drift significativo.

    Parameters
    ----------
    alerts_path:
        Ruta del registro persistente de alertas.

    flag_path:
        Ruta donde se materializará el gatillo.

    model_path:
        Ruta del modelo campeón actual.

    metrics_path:
        Ruta de los metadatos del entrenamiento.

    ventana:
        Ventana que se desea evaluar. Si es None, se utiliza la
        última ventana disponible entre W1-W4.

    Returns
    -------
    dict
        Resultado de la evaluación y creación del gatillo.
    """

    alerts_path = Path(alerts_path)
    flag_path = Path(flag_path)
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    # --------------------------------------------------------
    # 1. Verificar alerts.csv
    # --------------------------------------------------------

    if not alerts_path.exists():

        raise FileNotFoundError(
            f"No se encontró el registro de alertas: "
            f"{alerts_path}"
        )

    df_alertas = pd.read_csv(
        alerts_path
    )

    if df_alertas.empty:

        return {
            "required": False,
            "status": "not_required",
            "flag_created": False,
            "reason": (
                "El registro de alertas está vacío."
            )
        }

    # --------------------------------------------------------
    # 2. Validar columnas necesarias
    # --------------------------------------------------------

    columnas_requeridas = {
        "ventana",
        "feature",
        "metrica",
        "valor",
        "umbral",
        "severidad",
        "accion_recomendada",
        "performance_f1_macro",
        "alert_key"
    }

    faltantes = (
        columnas_requeridas
        -
        set(df_alertas.columns)
    )

    if faltantes:

        raise KeyError(
            "alerts.csv no cumple el contrato requerido. "
            f"Faltan columnas: {sorted(faltantes)}"
        )

    # --------------------------------------------------------
    # 3. Seleccionar ventana
    # --------------------------------------------------------

    if ventana is None:

        ventana_evaluada = (
            _identificar_ultima_ventana(
                df_alertas
            )
        )

    else:

        ventana_evaluada = str(
            ventana
        ).strip().upper()

        if ventana_evaluada not in ORDEN_VENTANAS:

            raise ValueError(
                "La ventana debe pertenecer a "
                "{W1, W2, W3, W4}."
            )

    df_ventana = (
        df_alertas[
            df_alertas["ventana"]
            .astype(str)
            .str.upper()
            .eq(ventana_evaluada)
        ]
        .copy()
    )

    if df_ventana.empty:

        return {
            "required": False,
            "status": "not_required",
            "flag_created": False,
            "ventana": ventana_evaluada,
            "reason": (
                "La ventana seleccionada no contiene alertas."
            )
        }

    # --------------------------------------------------------
    # 4. Seleccionar alertas significativas
    # --------------------------------------------------------

    df_significativas = (
        df_ventana[
            df_ventana["severidad"]
            .astype(str)
            .str.strip()
            .eq(SEVERIDAD_TRIGGER)
        ]
        .copy()
    )

    if df_significativas.empty:

        return {
            "required": False,
            "status": "not_required",
            "flag_created": False,
            "ventana": ventana_evaluada,
            "n_alertas_ventana": int(
                len(df_ventana)
            ),
            "n_alertas_significativas": 0,
            "reason": (
                "La ventana contiene alertas, pero ninguna "
                "alcanza Drift significativo."
            )
        }

    # --------------------------------------------------------
    # 5. Obtener performance
    # --------------------------------------------------------

    performance_disponible = (
        df_ventana["performance_f1_macro"]
        .dropna()
    )

    if performance_disponible.empty:

        performance_actual = None

    else:

        performance_actual = float(
            performance_disponible.iloc[-1]
        )

    # --------------------------------------------------------
    # 6. Construir trazabilidad
    # --------------------------------------------------------

    alert_keys = (
        df_significativas["alert_key"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    request_id = _construir_request_id(
        ventana=ventana_evaluada,
        alert_keys=alert_keys
    )

    features_afectadas = (
        df_significativas["feature"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    metricas_activadoras = (
        df_significativas["metrica"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    detalle_alertas = []

    for _, fila in df_significativas.iterrows():

        detalle_alertas.append(
            {
                "feature":
                    str(fila["feature"]),

                "metrica":
                    str(fila["metrica"]),

                "valor":
                    _valor_float_o_none(
                        fila["valor"]
                    ),

                "umbral":
                    _valor_float_o_none(
                        fila["umbral"]
                    ),

                "alert_key":
                    str(fila["alert_key"])
            }
        )

    # --------------------------------------------------------
    # 7. Construir el flag
    # --------------------------------------------------------

    solicitud = {
        "schema_version":
            "1.0",

        "request_id":
            request_id,

        "required":
            True,

        "status":
            "pending_review",

        "mode":
            "semi-automatic",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "ventana":
            ventana_evaluada,

        "estado":
            SEVERIDAD_TRIGGER,

        "prioridad":
            (
                "Alta"
                if performance_actual is not None
                else "Por revisar"
            ),

        "performance_f1_macro":
            performance_actual,

        "n_alertas_ventana":
            int(
                len(df_ventana)
            ),

        "n_alertas_significativas":
            int(
                len(df_significativas)
            ),

        "features_afectadas":
            features_afectadas,

        "metricas_activadoras":
            metricas_activadoras,

        "alert_keys":
            alert_keys,

        "detalle_alertas":
            detalle_alertas,

        "model_path":
            str(model_path),

        "train_metrics_path":
            str(metrics_path),

        "model_exists":
            bool(model_path.exists()),

        "train_metrics_exists":
            bool(metrics_path.exists()),

        "review_required":
            True,

        "approved":
            False,

        "action_required":
            (
                "Revisar calidad de datos, disponibilidad "
                "del target, deterioro de performance y "
                "viabilidad del reentrenamiento antes de "
                "modificar el modelo campeón."
            ),

        "next_command":
            "python src/retrain.py --check",

        "source_alerts":
            str(alerts_path)
    }

    # --------------------------------------------------------
    # 8. Idempotencia
    # --------------------------------------------------------

    if flag_path.exists():

        try:

            solicitud_existente = json.loads(
                flag_path.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError:

            solicitud_existente = {}

        if (
            solicitud_existente.get("request_id")
            ==
            request_id
        ):

            return {
                "required": True,
                "status": solicitud_existente.get(
                    "status",
                    "pending_review"
                ),
                "flag_created": False,
                "idempotent": True,
                "flag_path": str(flag_path),
                "request": solicitud_existente,
                "reason": (
                    "La misma solicitud ya se encuentra "
                    "materializada."
                )
            }

    # --------------------------------------------------------
    # 9. Persistir flag
    # --------------------------------------------------------

    _guardar_json_atomico(
        contenido=solicitud,
        ruta_destino=flag_path
    )

    return {
        "required": True,
        "status": "pending_review",
        "flag_created": True,
        "idempotent": False,
        "flag_path": str(flag_path),
        "request": solicitud,
        "reason": (
            "Drift significativo detectado. "
            "Se creó una solicitud semiautomática "
            "de evaluación de reentrenamiento."
        )
    }
