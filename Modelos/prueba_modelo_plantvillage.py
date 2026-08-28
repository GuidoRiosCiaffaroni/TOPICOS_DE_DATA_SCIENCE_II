# ============================================================
# PRUEBA INTERACTIVA DEL MODELO MLOPS — PLANTVILLAGE TOMATO
# ============================================================
#
# Objetivo
# --------
# 1. Localizar y cargar:
#       models/model.joblib
#       models/train_metrics.json
#       src/transformers.py
#
# 2. Verificar la integridad SHA-256 del modelo.
# 3. Solicitar una imagen de prueba.
# 4. Extraer exactamente las 15 características visuales usadas
#    durante el entrenamiento.
# 5. Ejecutar predict() y predict_proba().
# 6. Mostrar clase predicha, confianza y Top-3 probabilidades.
#
# Compatible con:
# - Google Colab
# - Python local
#
# IMPORTANTE
# ----------
# El modelo empaquetado es un Pipeline de scikit-learn cuyo
# estimador final es RandomForestClassifier y cuyo transformer
# personalizado se encuentra en:
#
#       src.transformers.FeatureEngineeringProduccion
#
# Por ello, src/transformers.py debe estar disponible antes de
# ejecutar joblib.load().
# ============================================================


# ============================================================
# BLOQUE 1 — LIBRERÍAS
# ============================================================

import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import cv2
from PIL import Image


# ============================================================
# BLOQUE 2 — CONFIGURACIÓN
# ============================================================

NOMBRE_MODELO = "model.joblib"
NOMBRE_METADATA = "train_metrics.json"

CARACTERISTICAS_VISUALES = [
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
    "h_mean",
    "s_mean",
    "v_mean",
    "excess_green",
    "brightness_mean",
    "contrast_std",
    "laplacian_variance",
    "entropy",
    "edge_density",
]


# ============================================================
# BLOQUE 3 — LOCALIZAR AUTOMÁTICAMENTE EL PROYECTO
# ============================================================

def localizar_raiz_proyecto():
    """
    Busca una carpeta que contenga simultáneamente:
        models/model.joblib
        models/train_metrics.json
        src/transformers.py
    """

    candidatos = [
        Path.cwd(),
        Path("/content"),
    ]

    # Añadir directorios inmediatos de /content, útil en Colab
    if Path("/content").exists():
        candidatos.extend(
            p for p in Path("/content").iterdir()
            if p.is_dir()
        )

    vistos = set()

    for base in candidatos:
        try:
            base = base.resolve()
        except Exception:
            continue

        if base in vistos:
            continue

        vistos.add(base)

        model_path = base / "models" / NOMBRE_MODELO
        metadata_path = base / "models" / NOMBRE_METADATA
        transformer_path = base / "src" / "transformers.py"

        if (
            model_path.exists()
            and metadata_path.exists()
            and transformer_path.exists()
        ):
            return base

    raise FileNotFoundError(
        "\nNo fue posible localizar automáticamente el proyecto.\n"
        "Se requieren estos archivos:\n"
        "  models/model.joblib\n"
        "  models/train_metrics.json\n"
        "  src/transformers.py\n\n"
        "Ejecute este script desde la raíz del proyecto o copie "
        "la estructura completa a /content."
    )


RAIZ_PROYECTO = localizar_raiz_proyecto()

MODEL_PATH = RAIZ_PROYECTO / "models" / NOMBRE_MODELO
METADATA_PATH = RAIZ_PROYECTO / "models" / NOMBRE_METADATA
TRANSFORMER_PATH = RAIZ_PROYECTO / "src" / "transformers.py"

print("=" * 90)
print("LOCALIZACIÓN DEL PROYECTO")
print("=" * 90)
print(f"Raíz              : {RAIZ_PROYECTO}")
print(f"Modelo            : {MODEL_PATH}")
print(f"Metadatos         : {METADATA_PATH}")
print(f"Transformer       : {TRANSFORMER_PATH}")


# ============================================================
# BLOQUE 4 — PREPARAR IMPORTACIÓN DEL TRANSFORMER
# ============================================================

if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

# El import es necesario antes de joblib.load() porque el
# Pipeline serializado contiene esta clase personalizada.
from src.transformers import FeatureEngineeringProduccion  # noqa: F401


# ============================================================
# BLOQUE 5 — CARGAR METADATOS
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as archivo:
    metadata = json.load(archivo)

features_metadata = (
    metadata
    .get("features", {})
    .get("input", [])
)

clases_metadata = (
    metadata
    .get("target", {})
    .get("classes", [])
)

modelo_registrado = (
    metadata
    .get("model", {})
    .get("selected_model", "No informado")
)

version_modelo = (
    metadata
    .get("artifact", {})
    .get("model_version", "No informada")
)

hash_registrado = (
    metadata
    .get("artifact", {})
    .get("model_sha256")
)


# ============================================================
# BLOQUE 6 — VERIFICAR CONTRATO DE FEATURES
# ============================================================

if not features_metadata:
    raise ValueError(
        "train_metrics.json no contiene metadata['features']['input']."
    )

if len(features_metadata) != 15:
    raise ValueError(
        f"Se esperaban 15 features de entrada y se encontraron "
        f"{len(features_metadata)}."
    )

if set(features_metadata) != set(CARACTERISTICAS_VISUALES):
    faltantes = sorted(
        set(features_metadata) - set(CARACTERISTICAS_VISUALES)
    )
    sobrantes = sorted(
        set(CARACTERISTICAS_VISUALES) - set(features_metadata)
    )

    raise ValueError(
        "El contrato de features no coincide con el extractor.\n"
        f"Faltantes en extractor: {faltantes}\n"
        f"Sobrantes en extractor: {sobrantes}"
    )


# ============================================================
# BLOQUE 7 — VERIFICAR SHA-256 DEL MODELO
# ============================================================

def calcular_sha256(ruta_archivo):
    sha = hashlib.sha256()

    with open(ruta_archivo, "rb") as archivo:
        for bloque in iter(
            lambda: archivo.read(1024 * 1024),
            b""
        ):
            sha.update(bloque)

    return sha.hexdigest()


hash_actual = calcular_sha256(MODEL_PATH)

if hash_registrado:
    hash_valido = hash_actual == hash_registrado
else:
    hash_valido = None


# ============================================================
# BLOQUE 8 — CARGAR EL PIPELINE
# ============================================================

modelo = joblib.load(MODEL_PATH)

if not hasattr(modelo, "predict"):
    raise TypeError(
        "El artefacto cargado no implementa predict()."
    )

if not hasattr(modelo, "predict_proba"):
    raise TypeError(
        "El artefacto cargado no implementa predict_proba()."
    )

if hasattr(modelo, "named_steps") and "model" in modelo.named_steps:
    estimador_final = modelo.named_steps["model"]
else:
    estimador_final = modelo

clases_modelo = [
    str(c)
    for c in getattr(
        estimador_final,
        "classes_",
        clases_metadata
    )
]

print("\n" + "=" * 90)
print("MODELO CARGADO")
print("=" * 90)
print(f"Modelo registrado : {modelo_registrado}")
print(f"Versión           : {version_modelo}")
print(f"Tipo de artefacto : {type(modelo).__name__}")
print(f"Estimador final   : {type(estimador_final).__name__}")
print(f"Features entrada  : {len(features_metadata)}")
print(f"Número de clases  : {len(clases_modelo)}")

if hash_valido is True:
    print("SHA-256           : OK")
elif hash_valido is False:
    raise RuntimeError(
        "El SHA-256 actual de model.joblib NO coincide "
        "con el registrado en train_metrics.json."
    )
else:
    print("SHA-256           : No disponible en metadatos")

print("[OK] Modelo preparado para inferencia.")


# ============================================================
# BLOQUE 9 — EXTRACCIÓN DE LAS 15 FEATURES DESDE UNA IMAGEN
# ============================================================

def extraer_caracteristicas_visuales(ruta_imagen):
    """
    Reproduce el extractor utilizado en el proyecto.

    Retorna exactamente estas 15 variables:
    r_mean, g_mean, b_mean,
    r_std, g_std, b_std,
    h_mean, s_mean, v_mean,
    excess_green,
    brightness_mean,
    contrast_std,
    laplacian_variance,
    entropy,
    edge_density.
    """

    ruta_imagen = Path(ruta_imagen)

    if not ruta_imagen.exists():
        raise FileNotFoundError(
            f"No existe la imagen: {ruta_imagen}"
        )

    # --------------------------------------------------------
    # 9.1 Cargar imagen y convertir a RGB
    # --------------------------------------------------------

    with Image.open(ruta_imagen) as img:
        rgb_uint8 = np.asarray(
            img.convert("RGB"),
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # 9.2 RGB normalizado [0, 1]
    # --------------------------------------------------------

    rgb = rgb_uint8.astype(np.float32) / 255.0

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    r_mean = float(np.mean(r))
    g_mean = float(np.mean(g))
    b_mean = float(np.mean(b))

    r_std = float(np.std(r))
    g_std = float(np.std(g))
    b_std = float(np.std(b))

    # --------------------------------------------------------
    # 9.3 HSV
    # H de OpenCV [0,179] -> grados [0,360)
    # S y V -> [0,1]
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        rgb_uint8,
        cv2.COLOR_RGB2HSV
    )

    h_deg = (
        hsv[:, :, 0].astype(np.float32)
        * 2.0
    )

    s = (
        hsv[:, :, 1].astype(np.float32)
        / 255.0
    )

    v = (
        hsv[:, :, 2].astype(np.float32)
        / 255.0
    )

    # Hue requiere media circular.
    h_rad = np.deg2rad(h_deg)

    sin_mean = np.mean(np.sin(h_rad))
    cos_mean = np.mean(np.cos(h_rad))

    h_mean = float(
        (
            np.degrees(
                np.arctan2(
                    sin_mean,
                    cos_mean
                )
            )
            + 360.0
        )
        % 360.0
    )

    s_mean = float(np.mean(s))
    v_mean = float(np.mean(v))

    # --------------------------------------------------------
    # 9.4 Excess Green
    # --------------------------------------------------------

    excess_green = float(
        np.mean(
            2.0 * g - r - b
        )
    )

    # --------------------------------------------------------
    # 9.5 Brillo aproximado
    # --------------------------------------------------------

    luminancia = (
        0.299 * r
        + 0.587 * g
        + 0.114 * b
    )

    brightness_mean = float(
        np.mean(luminancia)
    )

    # --------------------------------------------------------
    # 9.6 Escala de grises y contraste
    # --------------------------------------------------------

    gray_uint8 = cv2.cvtColor(
        rgb_uint8,
        cv2.COLOR_RGB2GRAY
    )

    gray = (
        gray_uint8.astype(np.float32)
        / 255.0
    )

    contrast_std = float(
        np.std(gray)
    )

    # --------------------------------------------------------
    # 9.7 Varianza del Laplaciano
    # --------------------------------------------------------

    laplacian_variance = float(
        cv2.Laplacian(
            gray_uint8,
            cv2.CV_64F
        ).var()
    )

    # --------------------------------------------------------
    # 9.8 Entropía de Shannon
    # --------------------------------------------------------

    hist = np.bincount(
        gray_uint8.ravel(),
        minlength=256
    ).astype(np.float64)

    probabilidades_hist = (
        hist / hist.sum()
    )

    probabilidades_hist = probabilidades_hist[
        probabilidades_hist > 0
    ]

    entropy = float(
        -np.sum(
            probabilidades_hist
            * np.log2(probabilidades_hist)
        )
    )

    # --------------------------------------------------------
    # 9.9 Densidad de bordes Canny
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray_uint8,
        threshold1=100,
        threshold2=200
    )

    edge_density = float(
        np.mean(edges > 0)
    )

    return {
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "r_std": r_std,
        "g_std": g_std,
        "b_std": b_std,
        "h_mean": h_mean,
        "s_mean": s_mean,
        "v_mean": v_mean,
        "excess_green": excess_green,
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
        "laplacian_variance": laplacian_variance,
        "entropy": entropy,
        "edge_density": edge_density,
    }


# ============================================================
# BLOQUE 10 — SOLICITAR IMAGEN DE PRUEBA
# ============================================================

def solicitar_imagen_prueba():
    """
    En Google Colab abre el selector de archivos.
    En ejecución local solicita una ruta mediante input().
    """

    try:
        from google.colab import files

        print("\n" + "=" * 90)
        print("PRUEBA DEL MODELO")
        print("=" * 90)
        print("Seleccione una imagen de una hoja de tomate para clasificar.")

        archivos = files.upload()

        if not archivos:
            raise RuntimeError(
                "No se seleccionó ninguna imagen."
            )

        nombre = next(iter(archivos.keys()))
        ruta = Path(nombre)

        return ruta

    except ImportError:
        print("\n" + "=" * 90)
        print("PRUEBA DEL MODELO")
        print("=" * 90)

        ruta = input(
            "Ingrese la ruta de la imagen de prueba: "
        ).strip().strip('"').strip("'")

        if not ruta:
            raise ValueError(
                "No se indicó una imagen."
            )

        return Path(ruta)


# ============================================================
# BLOQUE 11 — EJECUTAR PREDICCIÓN
# ============================================================

def predecir_imagen(ruta_imagen):
    caracteristicas = (
        extraer_caracteristicas_visuales(
            ruta_imagen
        )
    )

    # El orden de columnas debe respetar exactamente
    # metadata["features"]["input"].
    X_prueba = pd.DataFrame(
        [
            {
                feature: caracteristicas[feature]
                for feature in features_metadata
            }
        ],
        columns=features_metadata
    )

    prediccion = str(
        modelo.predict(X_prueba)[0]
    )

    probabilidades = np.asarray(
        modelo.predict_proba(X_prueba)[0],
        dtype=float
    )

    if len(probabilidades) != len(clases_modelo):
        raise ValueError(
            "El número de probabilidades no coincide "
            "con el número de clases."
        )

    if not np.isclose(
        probabilidades.sum(),
        1.0,
        atol=1e-6
    ):
        raise ValueError(
            "Las probabilidades no suman 1."
        )

    indice_pred = int(
        np.argmax(probabilidades)
    )

    clase_max = clases_modelo[
        indice_pred
    ]

    if prediccion != clase_max:
        raise ValueError(
            "predict() no coincide con argmax(predict_proba())."
        )

    confianza = float(
        probabilidades[indice_pred]
    )

    df_probabilidades = (
        pd.DataFrame(
            {
                "Clase": clases_modelo,
                "Probabilidad": probabilidades,
            }
        )
        .sort_values(
            "Probabilidad",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return {
        "ruta_imagen": str(ruta_imagen),
        "features": X_prueba,
        "prediccion": prediccion,
        "confianza": confianza,
        "probabilidades": df_probabilidades,
    }


# ============================================================
# BLOQUE 12 — MOSTRAR RESULTADOS
# ============================================================

def mostrar_resultados(resultado):
    print("\n" + "=" * 90)
    print("RESULTADO DE LA PRUEBA")
    print("=" * 90)

    print(
        f"Imagen             : "
        f"{resultado['ruta_imagen']}"
    )

    print(
        f"Clase predicha     : "
        f"{resultado['prediccion']}"
    )

    print(
        f"Confianza          : "
        f"{resultado['confianza']:.4f} "
        f"({resultado['confianza'] * 100:.2f}%)"
    )

    print("\nTOP 3 DE PROBABILIDADES")
    print("-" * 90)

    top3 = (
        resultado["probabilidades"]
        .head(3)
        .copy()
    )

    for i, fila in top3.iterrows():
        print(
            f"{i + 1}. "
            f"{fila['Clase']:<55} "
            f"{fila['Probabilidad']:.4f} "
            f"({fila['Probabilidad'] * 100:.2f}%)"
        )

    print("\n" + "=" * 90)
    print("15 FEATURES EXTRAÍDAS")
    print("=" * 90)

    print(
        resultado["features"]
        .T
        .rename(columns={0: "Valor"})
        .to_string()
    )

    print("\n[OK] Prueba finalizada correctamente.")


# ============================================================
# BLOQUE 13 — OPCIONAL: COMPARAR CON CLASE REAL
# ============================================================

def solicitar_clase_real(resultado):
    """
    Permite ingresar opcionalmente la clase verdadera.
    Si se deja vacío, la evaluación se omite.
    """

    print("\n" + "=" * 90)
    print("VALIDACIÓN OPCIONAL")
    print("=" * 90)

    print(
        "Si conoce la clase real, escríbala exactamente.\n"
        "Presione ENTER para omitir esta comparación."
    )

    clase_real = input(
        "Clase real: "
    ).strip()

    if not clase_real:
        print("Comparación con target real omitida.")
        return

    acierto = (
        clase_real
        ==
        resultado["prediccion"]
    )

    print(
        f"Clase real         : {clase_real}"
    )

    print(
        f"Clase predicha     : "
        f"{resultado['prediccion']}"
    )

    print(
        "Resultado          : "
        + (
            "CORRECTO"
            if acierto
            else "INCORRECTO"
        )
    )


# ============================================================
# BLOQUE 14 — PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    ruta_prueba = (
        solicitar_imagen_prueba()
    )

    resultado_prueba = (
        predecir_imagen(
            ruta_prueba
        )
    )

    mostrar_resultados(
        resultado_prueba
    )

    # En terminal/Colab interactivo se ofrece una comparación
    # opcional con el target verdadero.
    try:
        solicitar_clase_real(
            resultado_prueba
        )
    except EOFError:
        # Permite ejecución no interactiva.
        pass
