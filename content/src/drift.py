
"""
Utilidades para monitoreo de Data Drift.

Implementación inicial:
    - Population Stability Index (PSI)

Los bins se construyen exclusivamente utilizando
la distribución de referencia.
"""

import numpy as np
import pandas as pd


# ========================================================
# 1. CREAR BINS DESDE REFERENCIA
# ========================================================

def crear_bins_referencia(
    referencia,
    n_bins=10
):
    """
    Construye bins mediante cuantiles de la referencia.

    Parameters
    ----------
    referencia : array-like
        Valores de la distribución de referencia.

    n_bins : int, default=10
        Número objetivo de intervalos.

    Returns
    -------
    np.ndarray
        Bordes de los bins.

    Notas
    -----
    Los extremos se reemplazan por -inf y +inf para
    capturar valores de producción fuera del rango
    observado en referencia.
    """

    referencia = np.asarray(
        referencia,
        dtype=float
    )


    referencia = referencia[
        np.isfinite(
            referencia
        )
    ]


    if referencia.size == 0:

        raise ValueError(
            "La referencia no contiene valores "
            "numéricos finitos."
        )


    if n_bins < 2:

        raise ValueError(
            "n_bins debe ser al menos 2."
        )


    # ----------------------------------------------------
    # Caso especial: variable constante
    # ----------------------------------------------------

    if np.allclose(
        referencia,
        referencia[0]
    ):

        valor = float(
            referencia[0]
        )


        return np.array(
            [
                -np.inf,
                valor,
                np.inf
            ],
            dtype=float
        )


    # ----------------------------------------------------
    # Cuantiles de referencia
    # ----------------------------------------------------

    cuantiles = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )


    bordes = np.quantile(
        referencia,
        cuantiles
    )


    # ----------------------------------------------------
    # Eliminar duplicados
    # ----------------------------------------------------

    bordes = np.unique(
        bordes
    )


    if len(
        bordes
    ) < 2:

        raise ValueError(
            "No fue posible construir bins válidos."
        )


    # ----------------------------------------------------
    # Capturar valores fuera del rango de referencia
    # ----------------------------------------------------

    bordes = bordes.astype(
        float
    )


    bordes[0] = -np.inf
    bordes[-1] = np.inf


    return bordes


# ========================================================
# 2. CALCULAR PSI
# ========================================================

def calcular_psi(
    referencia,
    produccion,
    n_bins=10,
    epsilon=1e-6,
    devolver_detalle=False
):
    """
    Calcula Population Stability Index (PSI).

    PSI =
        sum(
            (A_i - E_i)
            *
            ln(A_i / E_i)
        )

    donde:

        E_i = proporción de referencia
        A_i = proporción de producción

    Parameters
    ----------
    referencia : array-like
        Distribución esperada.

    produccion : array-like
        Distribución observada.

    n_bins : int
        Número objetivo de bins.

    epsilon : float
        Constante para evitar proporciones cero.

    devolver_detalle : bool
        Si True, retorna además una tabla por bin.

    Returns
    -------
    float
        PSI total.

    pd.DataFrame, optional
        Detalle de contribución por bin.
    """

    referencia = np.asarray(
        referencia,
        dtype=float
    )


    produccion = np.asarray(
        produccion,
        dtype=float
    )


    referencia = referencia[
        np.isfinite(
            referencia
        )
    ]


    produccion = produccion[
        np.isfinite(
            produccion
        )
    ]


    if referencia.size == 0:

        raise ValueError(
            "La referencia está vacía."
        )


    if produccion.size == 0:

        raise ValueError(
            "La producción está vacía."
        )


    if epsilon <= 0:

        raise ValueError(
            "epsilon debe ser positivo."
        )


    # ----------------------------------------------------
    # Bins definidos SOLO con referencia
    # ----------------------------------------------------

    bins = crear_bins_referencia(
        referencia=referencia,
        n_bins=n_bins
    )


    # ----------------------------------------------------
    # Frecuencias absolutas
    # ----------------------------------------------------

    conteo_ref, _ = np.histogram(
        referencia,
        bins=bins
    )


    conteo_prod, _ = np.histogram(
        produccion,
        bins=bins
    )


    # ----------------------------------------------------
    # Proporciones
    # ----------------------------------------------------

    prop_ref = (
        conteo_ref
        /
        conteo_ref.sum()
    )


    prop_prod = (
        conteo_prod
        /
        conteo_prod.sum()
    )


    # ----------------------------------------------------
    # Estabilización numérica
    # ----------------------------------------------------

    prop_ref_segura = np.clip(
        prop_ref,
        epsilon,
        None
    )


    prop_prod_segura = np.clip(
        prop_prod,
        epsilon,
        None
    )


    # Renormalizar después del clipping.
    prop_ref_segura = (
        prop_ref_segura
        /
        prop_ref_segura.sum()
    )


    prop_prod_segura = (
        prop_prod_segura
        /
        prop_prod_segura.sum()
    )


    # ----------------------------------------------------
    # Contribución PSI por bin
    # ----------------------------------------------------

    contribucion = (

        (
            prop_prod_segura
            -
            prop_ref_segura
        )

        *

        np.log(
            prop_prod_segura
            /
            prop_ref_segura
        )
    )


    psi_total = float(
        np.sum(
            contribucion
        )
    )


    # ----------------------------------------------------
    # Retorno simple
    # ----------------------------------------------------

    if not devolver_detalle:

        return psi_total


    # ----------------------------------------------------
    # Tabla auditable
    # ----------------------------------------------------

    detalle = pd.DataFrame(
        {
            "bin":
                np.arange(
                    1,
                    len(
                        conteo_ref
                    )
                    +
                    1
                ),

            "limite_inferior":
                bins[:-1],

            "limite_superior":
                bins[1:],

            "n_referencia":
                conteo_ref,

            "n_produccion":
                conteo_prod,

            "E_i":
                prop_ref_segura,

            "A_i":
                prop_prod_segura,

            "contribucion_psi":
                contribucion
        }
    )


    return (
        psi_total,
        detalle
    )


# ========================================================
# 3. CLASIFICACIÓN OPERACIONAL DEL PSI
# ========================================================

def clasificar_psi(
    psi,
    umbral_warning=0.10,
    umbral_drift=0.20
):
    """
    Clasifica PSI según los umbrales operacionales
    definidos para este proyecto.

    No representa una ley estadística universal.
    """

    psi = float(
        psi
    )


    if psi < umbral_warning:

        return "Normal"


    if psi < umbral_drift:

        return "Warning"


    return "Drift significativo"


# ============================================================
# KOLMOGOROV-SMIRNOV
# ============================================================

def calcular_ks(
    referencia,
    produccion
):
    """
    Ejecuta el test Kolmogorov-Smirnov de dos muestras.

    Parameters
    ----------
    referencia : array-like
        Distribución de referencia.

    produccion : array-like
        Distribución observada en producción.

    Returns
    -------
    dict
        {
            "ks_statistic": float,
            "p_value": float
        }

    Notes
    -----
    Se utiliza un test bilateral:

        H0:
            F_ref(x) = F_prod(x)

        H1:
            F_ref(x) != F_prod(x)

    Los valores no finitos son eliminados antes del test.
    """

    import numpy as np

    from scipy.stats import (
        ks_2samp
    )


    # --------------------------------------------------------
    # Convertir a arrays
    # --------------------------------------------------------

    referencia = np.asarray(
        referencia,
        dtype=float
    )


    produccion = np.asarray(
        produccion,
        dtype=float
    )


    # --------------------------------------------------------
    # Eliminar NaN / inf
    # --------------------------------------------------------

    referencia = referencia[
        np.isfinite(
            referencia
        )
    ]


    produccion = produccion[
        np.isfinite(
            produccion
        )
    ]


    # --------------------------------------------------------
    # Verificaciones
    # --------------------------------------------------------

    if referencia.size == 0:

        raise ValueError(
            "La muestra de referencia está vacía."
        )


    if produccion.size == 0:

        raise ValueError(
            "La muestra de producción está vacía."
        )


    # --------------------------------------------------------
    # Test KS de dos muestras
    # --------------------------------------------------------

    resultado = ks_2samp(

        referencia,

        produccion,

        alternative="two-sided",

        method="auto"
    )


    return {

        "ks_statistic":
            float(
                resultado.statistic
            ),

        "p_value":
            float(
                resultado.pvalue
            )
    }


# ============================================================
# MAGNITUD DESCRIPTIVA DEL ESTADÍSTICO KS
# ============================================================

def interpretar_magnitud_ks(
    ks_statistic
):
    """
    Clasifica descriptivamente la magnitud del estadístico KS.

    Umbrales internos del proyecto:

        D < 0.10       -> Baja
        0.10 <= D < .20 -> Moderada
        D >= 0.20      -> Alta

    Estos valores NO constituyen umbrales universales.
    """

    ks_statistic = float(
        ks_statistic
    )


    if not (
        0.0
        <=
        ks_statistic
        <=
        1.0
    ):

        raise ValueError(
            "El estadístico KS debe estar "
            "entre 0 y 1."
        )


    if ks_statistic < 0.10:

        return "Baja"


    if ks_statistic < 0.20:

        return "Moderada"


    return "Alta"


# ============================================================
# CHI-CUADRADO PARA VARIABLES CATEGÓRICAS
# ============================================================

def calcular_chi2_categorica(
    referencia,
    produccion,
    alpha=0.05
):
    """
    Compara una variable categórica entre referencia
    y producción mediante Chi-cuadrado.

    Parameters
    ----------
    referencia : array-like
        Categorías de la población de referencia.

    produccion : array-like
        Categorías observadas en producción.

    alpha : float
        Nivel de significancia.

    Returns
    -------
    dict
        Resultado con:

        - chi2_statistic
        - p_value
        - dof
        - cramers_v
        - significativo
        - n_categorias
    """

    import numpy as np
    import pandas as pd

    from scipy.stats import (
        chi2_contingency
    )


    # --------------------------------------------------------
    # Convertir a Series
    # --------------------------------------------------------

    ref = pd.Series(
        referencia
    ).astype(
        "string"
    )


    prod = pd.Series(
        produccion
    ).astype(
        "string"
    )


    # --------------------------------------------------------
    # Eliminar valores faltantes
    # --------------------------------------------------------

    ref = ref.dropna()

    prod = prod.dropna()


    if len(ref) == 0:

        raise ValueError(
            "La referencia categórica está vacía."
        )


    if len(prod) == 0:

        raise ValueError(
            "La producción categórica está vacía."
        )


    # --------------------------------------------------------
    # Unión de categorías
    # --------------------------------------------------------
    #
    # Esto permite detectar categorías nuevas o categorías
    # desaparecidas en producción.
    #
    # --------------------------------------------------------

    categorias = sorted(
        set(
            ref.unique()
        )
        |
        set(
            prod.unique()
        )
    )


    if len(categorias) < 2:

        raise ValueError(
            "Chi-cuadrado requiere al menos "
            "dos categorías."
        )


    # --------------------------------------------------------
    # Frecuencias
    # --------------------------------------------------------

    conteo_ref = (
        ref
        .value_counts()
        .reindex(
            categorias,
            fill_value=0
        )
    )


    conteo_prod = (
        prod
        .value_counts()
        .reindex(
            categorias,
            fill_value=0
        )
    )


    # --------------------------------------------------------
    # Tabla de contingencia 2 x K
    # --------------------------------------------------------

    tabla = np.vstack(
        [
            conteo_ref.to_numpy(),
            conteo_prod.to_numpy()
        ]
    )


    # --------------------------------------------------------
    # Eliminar categorías ausentes en ambas muestras
    # --------------------------------------------------------

    columnas_validas = (
        tabla.sum(
            axis=0
        )
        >
        0
    )


    tabla = tabla[
        :,
        columnas_validas
    ]


    categorias_validas = np.asarray(
        categorias
    )[
        columnas_validas
    ]


    # --------------------------------------------------------
    # Test Chi-cuadrado
    # --------------------------------------------------------

    chi2, p_value, dof, expected = (
        chi2_contingency(
            tabla,
            correction=False
        )
    )


    # --------------------------------------------------------
    # Cramér's V
    # --------------------------------------------------------

    n_total = float(
        tabla.sum()
    )


    min_dimension = min(
        tabla.shape[0] - 1,
        tabla.shape[1] - 1
    )


    if min_dimension <= 0:

        cramers_v = 0.0

    else:

        cramers_v = float(
            np.sqrt(
                chi2
                /
                (
                    n_total
                    *
                    min_dimension
                )
            )
        )


    # --------------------------------------------------------
    # Tabla auditable
    # --------------------------------------------------------

    detalle = pd.DataFrame(
        {
            "categoria":
                categorias_validas,

            "frecuencia_referencia":
                tabla[0],

            "frecuencia_produccion":
                tabla[1],

            "proporcion_referencia":
                (
                    tabla[0]
                    /
                    tabla[0].sum()
                ),

            "proporcion_produccion":
                (
                    tabla[1]
                    /
                    tabla[1].sum()
                )
        }
    )


    return {

        "chi2_statistic":
            float(
                chi2
            ),

        "p_value":
            float(
                p_value
            ),

        "dof":
            int(
                dof
            ),

        "cramers_v":
            cramers_v,

        "significativo":
            bool(
                p_value
                <
                alpha
            ),

        "n_categorias":
            int(
                len(
                    categorias_validas
                )
            ),

        "detalle":
            detalle
    }


# ============================================================
# INTERPRETACIÓN DE CRAMÉR'S V
# ============================================================

def interpretar_cramers_v(
    valor
):
    """
    Clasificación descriptiva interna del proyecto.

        V < 0.10         -> Baja
        0.10 <= V < .30 -> Moderada
        V >= 0.30        -> Alta

    No representa una regla universal.
    """

    valor = float(
        valor
    )


    if not (
        0.0
        <=
        valor
        <=
        1.0
    ):

        raise ValueError(
            "Cramér's V debe encontrarse "
            "entre 0 y 1."
        )


    if valor < 0.10:

        return "Baja"


    if valor < 0.30:

        return "Moderada"


    return "Alta"
