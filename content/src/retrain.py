"""
Consumidor semiautomático de retrain_required.flag.

Uso:
    python src/retrain.py --check
    python src/retrain.py --approve

La opción --approve aprueba la solicitud para iniciar una fase
posterior de reentrenamiento. No reemplaza automáticamente el
modelo campeón.
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict

import argparse
import json
import os
import tempfile

import pandas as pd


# ============================================================
# RUTAS PREDETERMINADAS
# ============================================================

DEFAULT_FLAG_PATH = Path(
    "monitoring/retrain_required.flag"
)

DEFAULT_ALERTS_PATH = Path(
    "monitoring/alerts.csv"
)

DEFAULT_HISTORY_PATH = Path(
    "monitoring/retraining_requests.csv"
)


# ============================================================
# CONTRATO DEL FLAG
# ============================================================

REQUIRED_FIELDS = {
    "schema_version",
    "request_id",
    "required",
    "status",
    "mode",
    "created_at_utc",
    "ventana",
    "estado",
    "prioridad",
    "n_alertas_significativas",
    "features_afectadas",
    "metricas_activadoras",
    "alert_keys",
    "model_path",
    "train_metrics_path",
    "review_required",
    "approved",
    "action_required"
}


# ============================================================
# FUNCIONES
# ============================================================

def cargar_flag(
    flag_path: Path
) -> Dict[str, Any]:
    """
    Carga y valida sintácticamente el archivo flag.
    """

    flag_path = Path(
        flag_path
    )

    if not flag_path.exists():

        raise FileNotFoundError(
            f"No existe el gatillo: {flag_path}. "
            "No hay una solicitud activa de reentrenamiento."
        )

    try:

        solicitud = json.loads(
            flag_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            "retrain_required.flag no contiene JSON válido."
        ) from error

    return solicitud


def validar_contrato(
    solicitud: Dict[str, Any]
) -> None:
    """
    Comprueba el contrato mínimo de la solicitud.
    """

    faltantes = (
        REQUIRED_FIELDS
        -
        set(solicitud.keys())
    )

    if faltantes:

        raise KeyError(
            "El flag no cumple el contrato. "
            f"Faltan campos: {sorted(faltantes)}"
        )

    if solicitud["required"] is not True:

        raise ValueError(
            "El flag existe, pero required no es True."
        )

    if (
        int(
            solicitud["n_alertas_significativas"]
        )
        <=
        0
    ):

        raise ValueError(
            "El gatillo no contiene alertas significativas."
        )

    if not solicitud["request_id"]:

        raise ValueError(
            "La solicitud no contiene request_id."
        )


def verificar_dependencias(
    solicitud: Dict[str, Any],
    alerts_path: Path
) -> Dict[str, Any]:
    """
    Verifica artefactos y trazabilidad de alertas.
    """

    model_path = Path(
        solicitud["model_path"]
    )

    metrics_path = Path(
        solicitud["train_metrics_path"]
    )

    alerts_path = Path(
        alerts_path
    )

    if not alerts_path.exists():

        raise FileNotFoundError(
            f"No se encontró alerts.csv: {alerts_path}"
        )

    df_alertas = pd.read_csv(
        alerts_path
    )

    if "alert_key" not in df_alertas.columns:

        raise KeyError(
            "alerts.csv no contiene alert_key."
        )

    claves_esperadas = set(
        map(
            str,
            solicitud["alert_keys"]
        )
    )

    claves_observadas = set(
        df_alertas["alert_key"]
        .dropna()
        .astype(str)
    )

    claves_no_encontradas = (
        claves_esperadas
        -
        claves_observadas
    )

    resultado = {
        "model_exists":
            model_path.exists(),

        "train_metrics_exists":
            metrics_path.exists(),

        "alerts_exists":
            alerts_path.exists(),

        "all_alert_keys_found":
            len(claves_no_encontradas) == 0,

        "missing_alert_keys":
            sorted(
                claves_no_encontradas
            )
    }

    if not resultado["model_exists"]:

        raise FileNotFoundError(
            f"No se encontró el modelo campeón: {model_path}"
        )

    if not resultado["train_metrics_exists"]:

        raise FileNotFoundError(
            "No se encontraron los metadatos: "
            f"{metrics_path}"
        )

    if not resultado["all_alert_keys_found"]:

        raise ValueError(
            "No fue posible verificar todas las alertas "
            f"del gatillo: {claves_no_encontradas}"
        )

    return resultado


def guardar_json_atomico(
    contenido: Dict[str, Any],
    ruta: Path
) -> None:
    """
    Guarda el flag actualizado mediante reemplazo atómico.
    """

    ruta = Path(
        ruta
    )

    descriptor, ruta_temporal = tempfile.mkstemp(
        prefix="retrain_update_",
        suffix=".tmp",
        dir=str(ruta.parent)
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
            ruta
        )

    except Exception:

        temporal = Path(
            ruta_temporal
        )

        if temporal.exists():
            temporal.unlink()

        raise


def registrar_solicitud(
    solicitud: Dict[str, Any],
    history_path: Path
) -> None:
    """
    Registra el estado de la solicitud sin duplicados lógicos.
    """

    history_path = Path(
        history_path
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    registro = {
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "request_id":
            solicitud["request_id"],

        "ventana":
            solicitud["ventana"],

        "estado_drift":
            solicitud["estado"],

        "status":
            solicitud["status"],

        "prioridad":
            solicitud["prioridad"],

        "performance_f1_macro":
            solicitud.get(
                "performance_f1_macro"
            ),

        "n_alertas_significativas":
            solicitud["n_alertas_significativas"],

        "approved":
            solicitud["approved"]
    }

    if history_path.exists():

        df_historial = pd.read_csv(
            history_path
        )

    else:

        df_historial = pd.DataFrame()

    df_nuevo = pd.DataFrame(
        [registro]
    )

    df_total = pd.concat(
        [
            df_historial,
            df_nuevo
        ],
        ignore_index=True
    )

    # La combinación request_id + status representa un estado
    # lógico único de la solicitud.
    df_total = (
        df_total
        .drop_duplicates(
            subset=[
                "request_id",
                "status"
            ],
            keep="last"
        )
        .sort_values(
            "timestamp_utc"
        )
        .reset_index(
            drop=True
        )
    )

    df_total.to_csv(
        history_path,
        index=False,
        encoding="utf-8"
    )


def mostrar_resumen(
    solicitud: Dict[str, Any],
    verificacion: Dict[str, Any]
) -> None:
    """
    Muestra la evidencia principal del gatillo.
    """

    print("=" * 90)
    print("SOLICITUD DE EVALUACIÓN DE REENTRENAMIENTO")
    print("=" * 90)

    print(
        f"Request ID              : "
        f"{solicitud['request_id']}"
    )

    print(
        f"Ventana                 : "
        f"{solicitud['ventana']}"
    )

    print(
        f"Estado de drift         : "
        f"{solicitud['estado']}"
    )

    print(
        f"Estado de solicitud     : "
        f"{solicitud['status']}"
    )

    print(
        f"Prioridad               : "
        f"{solicitud['prioridad']}"
    )

    print(
        f"F1 macro observado      : "
        f"{solicitud.get('performance_f1_macro')}"
    )

    print(
        f"Alertas significativas  : "
        f"{solicitud['n_alertas_significativas']}"
    )

    print(
        f"Features afectadas      : "
        f"{len(solicitud['features_afectadas'])}"
    )

    print(
        f"Modelo campeón existe   : "
        f"{verificacion['model_exists']}"
    )

    print(
        f"Metadatos disponibles   : "
        f"{verificacion['train_metrics_exists']}"
    )

    print(
        f"Alertas verificadas     : "
        f"{verificacion['all_alert_keys_found']}"
    )

    print(
        "\nAcción requerida:"
    )

    print(
        solicitud["action_required"]
    )

    print("=" * 90)


def main() -> None:
    """
    Punto de entrada del script.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Valida o aprueba una solicitud semiautomática "
            "de reentrenamiento."
        )
    )

    grupo = parser.add_mutually_exclusive_group(
        required=True
    )

    grupo.add_argument(
        "--check",
        action="store_true",
        help=(
            "Valida la solicitud sin aprobarla."
        )
    )

    grupo.add_argument(
        "--approve",
        action="store_true",
        help=(
            "Aprueba la solicitud para una etapa posterior "
            "de reentrenamiento."
        )
    )

    parser.add_argument(
        "--flag-path",
        type=Path,
        default=DEFAULT_FLAG_PATH
    )

    parser.add_argument(
        "--alerts-path",
        type=Path,
        default=DEFAULT_ALERTS_PATH
    )

    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH
    )

    args = parser.parse_args()

    solicitud = cargar_flag(
        args.flag_path
    )

    validar_contrato(
        solicitud
    )

    verificacion = verificar_dependencias(
        solicitud=solicitud,
        alerts_path=args.alerts_path
    )

    if args.approve:

        solicitud["status"] = (
            "approved_for_retraining"
        )

        solicitud["approved"] = True

        solicitud["approved_at_utc"] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        solicitud["next_step"] = (
            "Construir un modelo candidato con datos "
            "validados y compararlo contra el campeón. "
            "No promover automáticamente."
        )

        guardar_json_atomico(
            contenido=solicitud,
            ruta=args.flag_path
        )

        print(
            "\n✅ Solicitud aprobada para iniciar "
            "la evaluación de reentrenamiento."
        )

    else:

        print(
            "\n✅ Solicitud validada."
        )

        print(
            "No se modificó el modelo ni el estado del flag."
        )

    registrar_solicitud(
        solicitud=solicitud,
        history_path=args.history_path
    )

    mostrar_resumen(
        solicitud=solicitud,
        verificacion=verificacion
    )


if __name__ == "__main__":
    main()
