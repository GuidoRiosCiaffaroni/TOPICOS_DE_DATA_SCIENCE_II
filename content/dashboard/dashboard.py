
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================

st.set_page_config(
    page_title="MLOps Data Drift Monitor",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CONSTANTES
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


MONITORING_DIR = (
    PROJECT_ROOT
    /
    "monitoring"
)


DRIFT_PATH = (
    MONITORING_DIR
    /
    "drift_metrics.csv"
)


SUMMARY_PATH = (
    MONITORING_DIR
    /
    "window_summary.csv"
)


ORDEN_VENTANAS = [
    "W0",
    "W1",
    "W2",
    "W3",
    "W4"
]


SEVERIDAD = {

    "Referencia":
        0,

    "Normal":
        0,

    "No":
        0,

    "Warning":
        1,

    "Drift":
        2,

    "Sí":
        2,

    "Drift significativo":
        2
}


# ============================================================
# FUNCIONES
# ============================================================

def cargar_datos():
    """
    Carga los productos consolidados de la Fase 4B.
    """

    if not DRIFT_PATH.exists():

        st.error(
            f"No existe: {DRIFT_PATH}"
        )

        st.stop()


    if not SUMMARY_PATH.exists():

        st.error(
            f"No existe: {SUMMARY_PATH}"
        )

        st.stop()


    drift = pd.read_csv(
        DRIFT_PATH
    )


    summary = pd.read_csv(
        SUMMARY_PATH
    )


    return (
        drift,
        summary
    )


def ordenar_ventanas(
    dataframe
):
    """
    Ordena W0-W4 de manera operacional.
    """

    resultado = (
        dataframe
        .copy()
    )


    resultado[
        "_orden"
    ] = (
        resultado[
            "Ventana"
        ]
        .map(
            {
                ventana: i
                for i, ventana
                in enumerate(
                    ORDEN_VENTANAS
                )
            }
        )
    )


    resultado = (

        resultado

        .sort_values(
            "_orden"
        )

        .drop(
            columns="_orden"
        )

        .reset_index(
            drop=True
        )
    )


    return resultado


def accion_recomendada(
    estado
):
    """
    Política informativa del dashboard.

    La ejecución automática del plan de acción
    corresponde a la Fase 5.
    """

    estado = str(
        estado
    )


    if estado in {
        "Referencia",
        "Normal",
        "No"
    }:

        return (
            "Continuar monitoreando. "
            "No se requiere intervención inmediata."
        )


    if estado == "Warning":

        return (
            "Realizar revisión manual, inspeccionar las "
            "variables que generaron advertencia y aumentar "
            "la frecuencia de seguimiento."
        )


    return (
        "Evaluar reentrenamiento, rollback o intervención "
        "semi-automática. Confirmar también si existe "
        "deterioro de performance antes de actuar."
    )


def mostrar_estado(
    estado
):
    """
    Muestra el estado usando los componentes nativos
    de Streamlit.
    """

    if estado in {
        "Referencia",
        "Normal",
        "No"
    }:

        st.success(
            f"Estado actual: {estado}"
        )


    elif estado == "Warning":

        st.warning(
            "Estado actual: WARNING"
        )


    else:

        st.error(
            f"Estado actual: {estado}"
        )


# ============================================================
# CARGA
# ============================================================

df_drift, df_summary = cargar_datos()


df_drift[
    "Ventana"
] = (
    df_drift[
        "Ventana"
    ]
    .astype(str)
)


df_summary[
    "Ventana"
] = (
    df_summary[
        "Ventana"
    ]
    .astype(str)
)


df_summary = ordenar_ventanas(
    df_summary
)


# ============================================================
# CABECERA
# ============================================================

st.title(
    "📊 MLOps Local — Monitoreo de Data Drift"
)


st.caption(
    "PlantVillage · clasificación multiclase · "
    "ventanas de producción simuladas"
)


st.info(
    "W0–W4 representan ventanas experimentales de producción. "
    "No corresponden a fechas reales de captura del dataset."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Controles"
)


ventanas_disponibles = [

    ventana
    for ventana
    in ORDEN_VENTANAS
    if ventana
    in set(
        df_summary[
            "Ventana"
        ]
    )
]


ventana_seleccionada = (
    st.sidebar.selectbox(
        "Ventana",
        ventanas_disponibles,
        index=(
            len(
                ventanas_disponibles
            )
            -
            1
        )
    )
)


features_disponibles = sorted(
    [
        feature
        for feature
        in df_drift[
            "Feature"
        ]
        .dropna()
        .unique()

        if str(
            feature
        ).upper()
        !=
        "TARGET"
    ]
)


feature_seleccionada = (
    st.sidebar.selectbox(
        "Feature",
        features_disponibles
    )
)


if st.sidebar.button(
    "Recargar datos"
):

    st.rerun()


# ============================================================
# ESTADO GENERAL ACTUAL
# ============================================================

produccion = (
    df_summary[
        df_summary[
            "Ventana"
        ]
        .isin(
            [
                "W1",
                "W2",
                "W3",
                "W4"
            ]
        )
    ]
)


if not produccion.empty:

    fila_actual = (
        ordenar_ventanas(
            produccion
        )
        .iloc[-1]
    )

else:

    fila_actual = (
        df_summary
        .iloc[-1]
    )


estado_actual = str(
    fila_actual[
        "Estado_general"
    ]
)


performance_actual = float(
    fila_actual[
        "Performance"
    ]
)


ventana_actual = str(
    fila_actual[
        "Ventana"
    ]
)


# ============================================================
# KPI SUPERIORES
# ============================================================

st.subheader(
    "1. Estado general del sistema"
)


mostrar_estado(
    estado_actual
)


col1, col2, col3, col4 = st.columns(
    4
)


col1.metric(
    "Ventana actual",
    ventana_actual
)


col2.metric(
    "F1 macro actual",
    f"{performance_actual:.4f}"
)


# ------------------------------------------------------------
# Alertas de la ventana actual
# ------------------------------------------------------------

alertas_actuales = (
    df_drift[
        (
            df_drift[
                "Ventana"
            ]
            ==
            ventana_actual
        )
        &
        (
            df_drift[
                "Drift"
            ]
            !=
            "No"
        )
    ]
)


col3.metric(
    "Elementos con alerta",
    int(
        len(
            alertas_actuales
        )
    )
)


if (
    "Drift significativo"
    in
    df_summary.columns
):

    n_drift_actual = int(
        fila_actual.get(
            "Drift significativo",
            0
        )
        if pd.notna(
            fila_actual.get(
                "Drift significativo",
                np.nan
            )
        )
        else 0
    )

else:

    n_drift_actual = int(
        (
            alertas_actuales[
                "Drift"
            ]
            ==
            "Sí"
        )
        .sum()
    )


col4.metric(
    "Drift significativo",
    n_drift_actual
)


# ============================================================
# PERFORMANCE
# ============================================================

st.divider()


st.subheader(
    "2. Performance del modelo por ventana"
)


performance_plot = (

    df_summary[
        [
            "Ventana",
            "Performance"
        ]
    ]

    .dropna()

    .copy()
)


performance_plot = ordenar_ventanas(
    performance_plot
)


performance_chart = (

    performance_plot

    .set_index(
        "Ventana"
    )[
        [
            "Performance"
        ]
    ]
)


st.line_chart(
    performance_chart
)


st.caption(
    "Performance = F1 macro. W0 se muestra como referencia; "
    "W1 representa el baseline operacional estable."
)


with st.expander(
    "Ver tabla de performance"
):

    st.dataframe(
        performance_plot,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DRIFT POR FEATURE
# ============================================================

st.divider()


st.subheader(
    "3. Métricas de drift por feature"
)


df_feature = (

    df_drift[
        df_drift[
            "Feature"
        ]
        ==
        feature_seleccionada
    ]

    .copy()
)


df_feature = ordenar_ventanas(
    df_feature
)


st.markdown(
    f"### `{feature_seleccionada}`"
)


col_psi, col_ks = st.columns(
    2
)


with col_psi:

    st.markdown(
        "**Population Stability Index (PSI)**"
    )


    psi_plot = (

        df_feature[
            [
                "Ventana",
                "PSI"
            ]
        ]

        .dropna()

        .set_index(
            "Ventana"
        )
    )


    st.line_chart(
        psi_plot
    )


    st.caption(
        "Referencia operacional: "
        "PSI < 0.10 normal; 0.10–0.20 warning; "
        "PSI ≥ 0.20 drift significativo."
    )


with col_ks:

    st.markdown(
        "**Kolmogorov-Smirnov / métrica secundaria**"
    )


    ks_plot = (

        df_feature[
            [
                "Ventana",
                "KS/Chi2"
            ]
        ]

        .dropna()

        .set_index(
            "Ventana"
        )
    )


    st.line_chart(
        ks_plot
    )


with st.expander(
    "Detalle técnico de la feature"
):

    columnas_detalle = [

        columna
        for columna
        in [
            "Ventana",
            "Feature",
            "PSI",
            "KS/Chi2",
            "p_value_fdr",
            "Severidad",
            "Drift",
            "Performance"
        ]
        if columna
        in df_feature.columns
    ]


    st.dataframe(
        df_feature[
            columnas_detalle
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EVOLUCIÓN DEL ESTADO
# ============================================================

st.divider()


st.subheader(
    "4. Evolución por ventanas de producción"
)


df_estado = (
    df_summary[
        [
            "Ventana",
            "Estado_general"
        ]
    ]
    .copy()
)


df_estado[
    "Severidad"
] = (
    df_estado[
        "Estado_general"
    ]
    .map(
        SEVERIDAD
    )
    .fillna(
        0
    )
)


df_estado = ordenar_ventanas(
    df_estado
)


st.line_chart(
    df_estado
    .set_index(
        "Ventana"
    )[
        [
            "Severidad"
        ]
    ]
)


st.caption(
    "Escala: 0 = Normal/Referencia · "
    "1 = Warning · 2 = Drift significativo. "
    "El eje representa ventanas simuladas, no tiempo calendario."
)


st.dataframe(
    df_estado,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# VARIABLES QUE GENERARON ALERTA
# ============================================================

st.divider()


st.subheader(
    "5. Variables que generaron alerta"
)


df_alertas = (

    df_drift[
        df_drift[
            "Drift"
        ]
        .isin(
            [
                "Warning",
                "Sí"
            ]
        )
    ]

    .copy()
)


df_alertas = ordenar_ventanas(
    df_alertas
)


if df_alertas.empty:

    st.success(
        "No existen elementos con alerta."
    )


else:

    columnas_alerta = [

        columna
        for columna
        in [
            "Ventana",
            "Feature",
            "PSI",
            "KS/Chi2",
            "Drift",
            "Performance"
        ]
        if columna
        in df_alertas.columns
    ]


    st.dataframe(
        df_alertas[
            columnas_alerta
        ],
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Alertas de ventana seleccionada
    # --------------------------------------------------------

    alertas_ventana = (

        df_alertas[
            df_alertas[
                "Ventana"
            ]
            ==
            ventana_seleccionada
        ]
    )


    st.markdown(
        f"**Alertas en {ventana_seleccionada}:** "
        f"{len(alertas_ventana)}"
    )


# ============================================================
# TARGET DRIFT
# ============================================================

df_target = (

    df_drift[
        df_drift[
            "Feature"
        ]
        .astype(str)
        .str.upper()
        ==
        "TARGET"
    ]

    .copy()
)


if not df_target.empty:

    st.divider()


    st.subheader(
        "6. Drift del target"
    )


    df_target = ordenar_ventanas(
        df_target
    )


    columnas_target = [

        columna
        for columna
        in [
            "Ventana",
            "KS/Chi2",
            "Cramers_V",
            "TVD",
            "Drift",
            "Performance"
        ]
        if columna
        in df_target.columns
    ]


    st.dataframe(
        df_target[
            columnas_target
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ACCIONES RECOMENDADAS
# ============================================================

st.divider()


st.subheader(
    "7. Resumen de acciones recomendadas"
)


accion = accion_recomendada(
    estado_actual
)


if estado_actual in {
    "Normal",
    "Referencia",
    "No"
}:

    st.success(
        accion
    )


elif estado_actual == "Warning":

    st.warning(
        accion
    )


else:

    st.error(
        accion
    )


st.markdown(
    """
**Criterio de actuación**

- **Normal:** continuar monitoreando.
- **Warning:** revisión manual y seguimiento reforzado.
- **Drift significativo:** evaluar reentrenamiento, rollback o intervención semi-automática.
"""
)


st.info(
    "Una alerta de drift no ejecuta automáticamente un "
    "reentrenamiento. La decisión debe considerar también "
    "la evolución de F1 macro y la persistencia del cambio."
)


# ============================================================
# TRAZABILIDAD
# ============================================================

st.divider()


st.caption(
    "Fuente: monitoring/drift_metrics.csv + "
    "monitoring/window_summary.csv"
)
