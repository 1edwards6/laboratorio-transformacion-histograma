from __future__ import annotations

from io import BytesIO
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.interpolate import PchipInterpolator


st.set_page_config(
    page_title="Transformación del histograma",
    page_icon="📈",
    layout="wide",
)

LEVELS = np.arange(256, dtype=float)


def load_grayscale(uploaded_file) -> np.ndarray:
    """Read an uploaded image and convert it to 8-bit grayscale."""
    with Image.open(uploaded_file) as source:
        return np.asarray(source.convert("L"), dtype=np.uint8)


def get_histogram_data(image: np.ndarray):
    """Return counts, normalized histogram profile and empirical CDF."""
    hist = np.bincount(image.ravel(), minlength=256).astype(float)
    max_count = float(hist.max()) if hist.size else 0.0
    profile = hist / max_count if max_count > 0 else hist.copy()
    cdf = np.cumsum(hist) / image.size
    return hist, profile, cdf


def build_histogram_approximation(
    profile: np.ndarray,
    num_points: int,
    method: str,
    harmonics: int = 8,
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray, int]:
    """Build an approximation of the normalized histogram profile.

    The histogram is normalized by its maximum frequency. This makes its
    vertical scale independent of the image's pixel count.
    """
    num_points = int(np.clip(num_points, 2, 256))
    control_x = np.linspace(0.0, 255.0, num_points)
    control_y = np.interp(control_x, LEVELS, profile)

    if method == "PCHIP":
        interpolator = PchipInterpolator(control_x, control_y, extrapolate=False)

        def approximation(x):
            values = np.asarray(interpolator(np.clip(x, 0, 255)), dtype=float)
            return values

        effective_harmonics = 0

    elif method == "Lineal":
        def approximation(x):
            return np.interp(np.asarray(x, dtype=float), control_x, control_y)

        effective_harmonics = 0

    elif method == "Fourier":
        # Least-squares trigonometric polynomial fitted to the selected points.
        # A maximum of floor((n_points-1)/2) harmonics keeps the model
        # determined by the available control points.
        effective_harmonics = max(
            0,
            min(int(harmonics), (num_points - 1) // 2),
        )

        def design_matrix(x_values: np.ndarray) -> np.ndarray:
            x_values = np.asarray(x_values, dtype=float)
            phase = 2.0 * np.pi * x_values / 256.0
            columns = [np.ones_like(phase)]
            for k in range(1, effective_harmonics + 1):
                columns.append(np.cos(k * phase))
                columns.append(np.sin(k * phase))
            return np.column_stack(columns)

        coefficients, *_ = np.linalg.lstsq(
            design_matrix(control_x), control_y, rcond=None
        )

        def approximation(x):
            return design_matrix(np.asarray(x, dtype=float)) @ coefficients

    else:
        raise ValueError(f"Método no reconocido: {method}")

    return approximation, control_x, control_y, effective_harmonics


def transform_from_histogram(
    image: np.ndarray,
    approximation: Callable[[np.ndarray], np.ndarray],
    A: float,
    B: float,
    C: float,
    D: float,
    effect_scale: float,
):
    """Transform intensities using deformation of the histogram approximation.

    q_hat(r) is the fitted, max-normalized histogram. First define
        G(r) = C * q_hat(A*r + B) + D
    then turn the curve deformation into an intensity displacement:
        T(r) = r + effect_scale * (G(r) - q_hat(r)).

    This definition intentionally guarantees that A=1, B=0, C=1, D=0
    yields T(r)=r (up to floating-point precision and integer rounding).
    The histogram is not used directly as a classical equalization map.
    """
    base_curve = np.asarray(approximation(LEVELS), dtype=float)
    transformed_argument = np.clip(A * LEVELS + B, 0.0, 255.0)
    transformed_curve = C * np.asarray(approximation(transformed_argument), dtype=float) + D
    curve_difference = transformed_curve - base_curve

    intensity_map_float = LEVELS + effect_scale * curve_difference
    intensity_map = np.rint(np.clip(intensity_map_float, 0, 255)).astype(np.uint8)
    transformed_image = intensity_map[image]

    return (
        transformed_image,
        base_curve,
        transformed_curve,
        intensity_map_float,
        intensity_map,
        curve_difference,
    )


def calculate_metrics(image: np.ndarray) -> dict[str, float]:
    hist = np.bincount(image.ravel(), minlength=256).astype(float)
    probabilities = hist / image.size
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    p1, p99 = np.percentile(image, [1, 99])
    return {
        "Media": float(np.mean(image)),
        "Desviación estándar": float(np.std(image)),
        "Mínimo": float(np.min(image)),
        "Máximo": float(np.max(image)),
        "Rango dinámico": float(np.max(image) - np.min(image)),
        "Rango P1–P99": float(p99 - p1),
        "Entropía (bits)": entropy,
        "Píxeles en 0 (%)": float(100 * np.mean(image == 0)),
        "Píxeles en 255 (%)": float(100 * np.mean(image == 255)),
    }


def make_figure_histogram_overlay(
    hist: np.ndarray,
    approximation_curve: np.ndarray,
    title: str,
):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(LEVELS, hist, width=1.0, color="lightgray", label="Histograma original")
    ax.plot(
        LEVELS,
        approximation_curve * hist.max(),
        linewidth=2,
        label="Aproximación del perfil del histograma",
    )
    ax.set_title(title)
    ax.set_xlabel("Nivel de intensidad")
    ax.set_ylabel("Frecuencia de píxeles")
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def make_figure_function_transform(base_curve, transformed_curve, intensity_map_float):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(LEVELS, base_curve, label="Aproximación original", linewidth=2)
    axes[0].plot(LEVELS, transformed_curve, label="Función transformada", linewidth=2)
    axes[0].set_title("Transformación de la aproximación del histograma")
    axes[0].set_xlabel("Variable r")
    axes[0].set_ylabel("Perfil normalizado")
    axes[0].set_xlim(0, 255)
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(LEVELS, intensity_map_float, label="Mapa T(r)", linewidth=2)
    axes[1].plot(LEVELS, LEVELS, "--", label="Identidad T(r)=r")
    axes[1].set_title("Mapa final de intensidades")
    axes[1].set_xlabel("Intensidad de entrada r")
    axes[1].set_ylabel("Intensidad de salida")
    axes[1].set_xlim(0, 255)
    axes[1].set_ylim(-10, 265)
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.tight_layout()
    return fig


def make_figure_distributions(results: dict[str, np.ndarray]):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for name, image in results.items():
        hist, _, cdf = get_histogram_data(image)
        axes[0].plot(LEVELS, hist / image.size, label=name, linewidth=1.5)
        axes[1].plot(LEVELS, cdf, label=name, linewidth=1.7)

    axes[0].set_title("Comparación de histogramas")
    axes[0].set_xlabel("Nivel de intensidad")
    axes[0].set_ylabel("Frecuencia relativa")
    axes[0].set_xlim(0, 255)
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Comparación de CDF (solo evaluación)")
    axes[1].set_xlabel("Nivel de intensidad")
    axes[1].set_ylabel("Probabilidad acumulada")
    axes[1].set_xlim(0, 255)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


st.title("Laboratorio de transformación de intensidades a partir del histograma")
st.markdown(
    "Esta aplicación aproxima **directamente el perfil del histograma**, lo transforma "
    "mediante cuatro parámetros y convierte la diferencia entre ambas curvas en un "
    "desplazamiento de intensidades. La CDF se muestra únicamente para describir y evaluar "
    "las imágenes; no se utiliza para construir la función de transformación."
)

with st.sidebar:
    st.header("1. Aproximación del histograma")
    method = st.selectbox("Método", ["PCHIP", "Lineal", "Fourier"], index=0)
    num_points = st.slider("Número de puntos de control", 4, 256, 32, step=4)
    harmonics = 8
    if method == "Fourier":
        max_allowed = max(1, (num_points - 1) // 2)
        harmonics = st.slider(
            "Número de armónicos",
            min_value=1,
            max_value=min(40, max_allowed),
            value=min(8, max_allowed),
        )

    st.header("2. Parámetros de transformación")
    A = st.slider("A — multiplicación de la variable", -3.0, 3.0, 1.0, step=0.05)
    B = st.slider("B — desplazamiento de la variable", -255, 255, 0, step=1)
    C = st.slider("C — multiplicación de la función", -5.0, 5.0, 1.0, step=0.05)
    D = st.slider("D — desplazamiento de la función", -5.0, 5.0, 0.0, step=0.05)
    effect_scale = st.slider(
        "Escala del efecto sobre las intensidades",
        min_value=0,
        max_value=255,
        value=80,
        step=5,
        help="Convierte la diferencia entre la función transformada y la original en un desplazamiento de intensidad.",
    )
    st.caption("Parámetros neutros: A=1, B=0, C=1, D=0. Con ellos, T(r)=r.")

uploaded = st.file_uploader(
    "Carga una imagen",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
)

if uploaded is None:
    st.info("Carga una imagen para comenzar el análisis.")
    st.stop()

try:
    image_gray = load_grayscale(uploaded)
except Exception as exc:
    st.error(f"No fue posible leer la imagen: {exc}")
    st.stop()

hist, profile, cdf = get_histogram_data(image_gray)
approximation, control_x, control_y, effective_harmonics = build_histogram_approximation(
    profile,
    num_points,
    method,
    harmonics,
)

(
    proposed_image,
    base_curve,
    transformed_curve,
    intensity_map_float,
    intensity_map,
    curve_difference,
) = transform_from_histogram(
    image_gray,
    approximation,
    A,
    float(B),
    C,
    D,
    float(effect_scale),
)

he_image = cv2.equalizeHist(image_gray)
clahe_image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_gray)
results = {
    "Original": image_gray,
    "Modelo histograma": proposed_image,
    "Ecualización clásica (HE)": he_image,
    "CLAHE": clahe_image,
}

# Overview
st.subheader("Imagen original y métricas iniciales")
col_img, col_metrics = st.columns([1, 2])
with col_img:
    st.image(image_gray, caption="Imagen convertida a escala de grises", clamp=True, use_container_width=True)
with col_metrics:
    st.dataframe(
        pd.DataFrame([calculate_metrics(image_gray)], index=["Original"]).T.round(4),
        use_container_width=True,
    )

st.subheader("Histograma original y aproximación")
st.caption(
    f"El perfil del histograma se normaliza dividiendo las frecuencias por la frecuencia máxima. "
    f"Método: {method}. Puntos de control: {num_points}."
    + (f" Armónicos efectivos: {effective_harmonics}." if method == "Fourier" else "")
)
st.pyplot(make_figure_histogram_overlay(hist, base_curve, "Histograma original y curva aproximada"), use_container_width=True)

st.subheader("Función transformada y mapa de intensidades")
st.latex(r"G(r)=C\,\widehat h(Ar+B)+D,\qquad T(r)=r+\lambda\,[G(r)-\widehat h(r)]")
st.caption(
    r"La función aproximada $\widehat h$ es el histograma normalizado por su máximo. "
    r"La escala $\lambda$ controla cuánto se traduce la deformación de la curva en un desplazamiento de intensidad."
)
st.pyplot(make_figure_function_transform(base_curve, transformed_curve, intensity_map_float), use_container_width=True)

neutral_parameters = (A == 1.0 and B == 0 and C == 1.0 and D == 0.0)
if neutral_parameters:
    if np.array_equal(proposed_image, image_gray):
        st.success("Parámetros neutros verificados: la imagen propuesta es idéntica a la original.")
    else:
        st.warning("Se esperaba que los parámetros neutros conservaran la imagen. Revisa la implementación.")

monotonicity_violations = int(np.sum(np.diff(intensity_map.astype(int)) < 0))
clipped_levels = int(np.sum((intensity_map_float < 0) | (intensity_map_float > 255)))
quality_col1, quality_col2, quality_col3 = st.columns(3)
quality_col1.metric("Saltos decrecientes en T(r)", monotonicity_violations)
quality_col2.metric("Niveles recortados", clipped_levels)
quality_col3.metric("Píxeles saturados en el modelo (%)", round(100 * np.mean((proposed_image == 0) | (proposed_image == 255)), 3))

st.subheader("Comparación de histogramas y CDF")
st.pyplot(make_figure_distributions(results), use_container_width=True)

st.subheader("Comparación de imágenes")
image_columns = st.columns(4)
for column, (name, result_image) in zip(image_columns, results.items()):
    with column:
        st.image(result_image, caption=name, clamp=True, use_container_width=True)

st.subheader("Comparación de métricas")
metrics_table = pd.DataFrame({name: calculate_metrics(img) for name, img in results.items()}).T
st.dataframe(metrics_table.round(4), use_container_width=True)

csv_data = metrics_table.to_csv(index=True).encode("utf-8-sig")
st.download_button(
    "Descargar tabla de métricas (CSV)",
    data=csv_data,
    file_name="metricas_comparacion_histograma.csv",
    mime="text/csv",
)

buffer = BytesIO()
Image.fromarray(proposed_image).save(buffer, format="PNG")
st.download_button(
    "Descargar imagen del modelo propuesto (PNG)",
    data=buffer.getvalue(),
    file_name="imagen_transformada_histograma.png",
    mime="image/png",
)

st.divider()
st.caption(
    "Nota metodológica: esta es una formulación experimental para estudiar deformaciones del perfil del histograma. "
    "No es ecualización clásica. Algunas combinaciones pueden producir mapas no monótonos o recorte de intensidades; "
    "estos efectos deben registrarse e interpretarse en el análisis."
)
