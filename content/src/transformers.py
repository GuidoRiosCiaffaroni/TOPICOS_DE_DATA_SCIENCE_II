
import numpy as np
import pandas as pd

from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)

from sklearn.utils.validation import (
    check_is_fitted
)


class FeatureEngineeringProduccion(
    BaseEstimator,
    TransformerMixin
):
    """
    Feature Engineering determinista utilizado
    por el Pipeline PlantVillage.

    Transformaciones
    ----------------
    h_mean:
        h_sin = sin(theta)
        h_cos = cos(theta)

    laplacian_variance:
        laplacian_log1p = log(1 + x)

    No aprende estadísticas del dataset.
    """

    def __init__(
        self,
        eliminar_h_mean=True,
        eliminar_laplacian_original=True
    ):

        self.eliminar_h_mean = (
            eliminar_h_mean
        )

        self.eliminar_laplacian_original = (
            eliminar_laplacian_original
        )


    def fit(
        self,
        X,
        y=None
    ):

        if not isinstance(
            X,
            pd.DataFrame
        ):

            raise TypeError(
                "FeatureEngineeringProduccion "
                "requiere un pandas.DataFrame."
            )


        columnas_requeridas = {
            "h_mean",
            "laplacian_variance"
        }


        faltantes = (
            columnas_requeridas
            -
            set(
                X.columns
            )
        )


        if faltantes:

            raise KeyError(
                "Faltan columnas requeridas: "
                f"{sorted(faltantes)}"
            )


        self.feature_names_in_ = np.array(
            X.columns,
            dtype=object
        )


        self.n_features_in_ = (
            X.shape[1]
        )


        return self


    def transform(
        self,
        X
    ):

        check_is_fitted(
            self,
            attributes=[
                "feature_names_in_"
            ]
        )


        if not isinstance(
            X,
            pd.DataFrame
        ):

            raise TypeError(
                "La entrada debe ser un "
                "pandas.DataFrame."
            )


        X_out = X.copy()


        # ================================================
        # HUE
        # ================================================

        h = pd.to_numeric(
            X_out[
                "h_mean"
            ],
            errors="coerce"
        )


        if h.isna().any():

            raise ValueError(
                "'h_mean' contiene valores faltantes "
                "o no numéricos."
            )


        if (
            (h < 0)
            |
            (h >= 360)
        ).any():

            raise ValueError(
                "'h_mean' debe estar en [0, 360)."
            )


        h_rad = np.deg2rad(
            h.to_numpy(
                dtype=float
            )
        )


        X_out[
            "h_sin"
        ] = np.sin(
            h_rad
        )


        X_out[
            "h_cos"
        ] = np.cos(
            h_rad
        )


        # ================================================
        # LAPLACIAN VARIANCE
        # ================================================

        laplacian = pd.to_numeric(
            X_out[
                "laplacian_variance"
            ],
            errors="coerce"
        )


        if laplacian.isna().any():

            raise ValueError(
                "'laplacian_variance' contiene "
                "valores faltantes o no numéricos."
            )


        if (
            laplacian < 0
        ).any():

            raise ValueError(
                "'laplacian_variance' no puede "
                "contener valores negativos."
            )


        X_out[
            "laplacian_log1p"
        ] = np.log1p(
            laplacian.to_numpy(
                dtype=float
            )
        )


        # ================================================
        # ELIMINAR VARIABLES ORIGINALES
        # ================================================

        columnas_eliminar = []


        if self.eliminar_h_mean:

            columnas_eliminar.append(
                "h_mean"
            )


        if self.eliminar_laplacian_original:

            columnas_eliminar.append(
                "laplacian_variance"
            )


        X_out = X_out.drop(
            columns=columnas_eliminar
        )


        return X_out


    def get_feature_names_out(
        self,
        input_features=None
    ):

        check_is_fitted(
            self,
            attributes=[
                "feature_names_in_"
            ]
        )


        if input_features is None:

            features = list(
                self.feature_names_in_
            )

        else:

            features = list(
                input_features
            )


        if (
            self.eliminar_h_mean
            and
            "h_mean" in features
        ):

            features.remove(
                "h_mean"
            )


        if (
            self.eliminar_laplacian_original
            and
            "laplacian_variance" in features
        ):

            features.remove(
                "laplacian_variance"
            )


        features.extend(
            [
                "h_sin",
                "h_cos",
                "laplacian_log1p"
            ]
        )


        return np.array(
            features,
            dtype=object
        )
