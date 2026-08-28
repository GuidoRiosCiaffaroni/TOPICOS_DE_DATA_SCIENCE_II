
import math

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator
)


class PredictionRequest(BaseModel):
    """
    Contrato de entrada para el modelo PlantVillage.

    El Pipeline espera 15 features visuales originales.

    Las transformaciones:

        h_mean
            -> h_sin
            -> h_cos

        laplacian_variance
            -> laplacian_log1p

    se ejecutan internamente en model.joblib.

    Por tanto, NO deben ser enviadas por el cliente.
    """

    # ====================================================
    # CONFIGURACIÓN PYDANTIC
    # ====================================================

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )


    # ====================================================
    # RGB — MEDIAS
    # ====================================================

    r_mean: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Media normalizada del canal rojo."
        )
    )


    g_mean: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Media normalizada del canal verde."
        )
    )


    b_mean: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Media normalizada del canal azul."
        )
    )


    # ====================================================
    # RGB — DESVIACIONES ESTÁNDAR
    # ====================================================

    r_std: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Desviación estándar normalizada "
            "del canal rojo."
        )
    )


    g_std: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Desviación estándar normalizada "
            "del canal verde."
        )
    )


    b_std: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Desviación estándar normalizada "
            "del canal azul."
        )
    )


    # ====================================================
    # HSV
    # ====================================================

    h_mean: float = Field(
        ...,
        ge=0.0,
        lt=360.0,
        description=(
            "Media circular de Hue en grados. "
            "Dominio [0, 360)."
        )
    )


    s_mean: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Saturación media normalizada."
        )
    )


    v_mean: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Value medio normalizado."
        )
    )


    # ====================================================
    # VEGETACIÓN
    # ====================================================

    excess_green: float = Field(
        ...,
        ge=-2.0,
        le=2.0,
        description=(
            "Índice Excess Green derivado "
            "de canales RGB normalizados."
        )
    )


    # ====================================================
    # BRILLO Y CONTRASTE
    # ====================================================

    brightness_mean: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Brillo medio normalizado."
        )
    )


    contrast_std: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Desviación estándar utilizada "
            "como medida de contraste."
        )
    )


    # ====================================================
    # NITIDEZ / TEXTURA
    # ====================================================

    laplacian_variance: float = Field(
        ...,
        ge=0.0,
        description=(
            "Varianza del Laplaciano. "
            "Debe ser no negativa."
        )
    )


    entropy: float = Field(
        ...,
        ge=0.0,
        le=8.0,
        description=(
            "Entropía de imagen."
        )
    )


    edge_density: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Proporción de píxeles asociados "
            "a bordes."
        )
    )


    # ====================================================
    # VALIDACIÓN GENERAL
    # ====================================================

    @field_validator("*")
    @classmethod
    def validar_valor_finito(
        cls,
        valor
    ):
        """
        Rechaza NaN, +Inf y -Inf.

        Los modelos de producción deben recibir
        valores numéricos finitos.
        """

        if not math.isfinite(
            float(valor)
        ):

            raise ValueError(
                "Todos los valores deben ser "
                "números finitos."
            )


        return valor
