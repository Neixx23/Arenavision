import io
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

st.set_page_config(page_title="ArenaVision", page_icon="🪨", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1500px;}
[data-testid="stMetric"] {background: #f5f7fa; border: 1px solid #e4e8ee; border-radius: 12px; padding: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("🪨 ArenaVision — análisis visual de arena sílice")
st.caption("Herramienta experimental de orientación. La composición química y la dosis de lixiviación deben validarse en laboratorio.")

st.warning("**No mezcle ácidos ni prepare una receta operacional con estas estimaciones.** Los vapores y residuos pueden ser peligrosos. Realice cualquier ensayo bajo un procedimiento aprobado y con EPP, ventilación y manejo de residuos.")

@st.cache_data
def resize_image(raw_bytes, max_side=1600):
    image = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def analyze_grains(rgb, min_size=35):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0]
    blur = cv2.GaussianBlur(luminance, (5, 5), 0)
    # Segmenta granos a partir de bordes y separa granos tocantes con watershed.
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
    distance = ndimage.distance_transform_edt(binary)
    local_max = peak_local_max(distance, labels=binary, footprint=np.ones((15, 15)), min_distance=7)
    markers = np.zeros(binary.shape, dtype=np.int32)
    markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
    labels = watershed(-distance, markers, mask=binary)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    chroma = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 1:]
    stats, areas, diameters = [], [], []
    for grain_id in range(1, labels.max() + 1):
        mask = labels == grain_id
        area = int(mask.sum())
        if area < min_size:
            continue
        ys, xs = np.where(mask)
        radius = np.sqrt(max(area / np.pi, 1))
        areas.append(area)
        diameters.append(2 * radius)
        pixels_rgb = rgb[mask]
        pixels_hsv = hsv[mask]
        mean_rgb = pixels_rgb.mean(axis=0)
        stats.append({
            "mean_r": mean_rgb[0], "mean_g": mean_rgb[1], "mean_b": mean_rgb[2],
            "mean_hue": pixels_hsv[:, 0].mean(), "mean_sat": pixels_hsv[:, 1].mean(),
            "chroma_a": chroma[mask, 0].mean(), "chroma_b": chroma[mask, 1].mean(),
        })

    overlay = rgb.copy()
    boundaries = (labels > 0) & (labels != ndimage.shift(labels, (0, 1), order=0))
    overlay[boundaries] = [255, 90, 90]
    return labels, overlay, np.asarray(areas), np.asarray(diameters), stats


def visible_composition(rgb, grid_rows=30, grid_cols=30):
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    luminance = lab[:, :, 0] / 255.0
    a = lab[:, :, 1] - 128
    b = lab[:, :, 2] - 128
    sat = hsv[:, :, 1] / 255.0

    orange_red = ((b > 8) & (a > 2)).mean() * 100
    dark = (luminance < 0.48).mean() * 100
    bright = ((luminance > 0.70) & (sat < 0.18)).mean() * 100
    neutral = (np.sqrt(a*a + b*b) < 12).mean() * 100
    yellow = ((b > 7) & (b > np.abs(a))).mean() * 100

    # Esta separación es visual, no una identificación química.
    impurities = np.clip(dark * 0.65 + orange_red * 0.55 + yellow * 0.30, 0, 35)
    sio2 = np.clip(bright + neutral * 0.35, 45, 99.5)
    sio2 = np.clip(sio2, impurities, 99.5)
    sio2 = 100 - min(impurities, 100 - sio2)
    rest = max(0, 100 - sio2)
    return {
        "SiO2 visual": round(float(sio2), 2),
        "Impurezas oscuras/rojizas (visual)": round(float(rest), 2),
        "Área brillante claro": round(float(bright), 1),
        "Área neutra clara": round(float(neutral), 1),
        "Área rojiza/amarilla": round(float(orange_red + yellow), 1),
        "Área oscura": round(float(dark), 1),
    }


def acid_equivalents(composition_pct, sample_kg, safety_factor=1.25):
    moles_kg = {oxide: sample_kg * composition_pct.get(oxide, 0.0) / 100 / molar_mass for oxide, molar_mass in {
        "Fe2O3": 159.69, "Al2O3": 101.96, "CaO": 56.08, "MgO": 40.30, "TiO2": 79.87, "K2O": 94.20
    }.items()}
    # kmol del reactivo puro requeridos por unidad de óxido según una ecuación simplificada.
    hcl_kmol = moles_kg.get("Fe2O3", 0) * 6 + moles_kg.get("Al2O3", 0) * 6 + moles_kg.get("CaO", 0) * 2 + moles_kg.get("MgO", 0) * 2 + moles_kg.get("TiO2", 0) * 4 + moles_kg.get("K2O", 0) * 2
    h2so4_kmol = moles_kg.get("Fe2O3", 0) * 3 + moles_kg.get("Al2O3", 0) * 3 + moles_kg.get("CaO", 0) * 1 + moles_kg.get("MgO", 0) * 1 + moles_kg.get("TiO2", 0) * 2
    # reactivo puro masa y masa comercial aproximada
    return {
        "HCl 37%": hcl_kmol * 36.46 * safety_factor / 0.37,
        "H2SO4 98%": h2so4_kmol * 98.079 * safety_factor / 0.98,
    }

with st.sidebar:
    st.header("1. Captura")
    source = st.radio("Fuente", ["Subir imágenes", "Usar cámara"], horizontal=False)
    uploads = []
    if source == "Subir imágenes":
        uploads = st.file_uploader("Imágenes representativas", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
    else:
        camera = st.camera_input("Foto de una muestra homogénea sobre fondo neutro")
        if camera:
            uploads = [camera]

    st.divider()
    st.header("2. Muestra")
    sample_kg = st.number_input("Masa seca de muestra (kg)", min_value=0.001, max_value=10000.0, value=1.0, step=0.1, format="%.3f")
    pixels_per_mm = st.number_input("Calibración visual (px/mm)", min_value=1.0, value=100.0, help="Calibre con una regla o patrón en la misma distancia focal.")
    acid_type = st.selectbox("Ácido comercial considerado", ["HCl 37%", "H2SO4 98%"])
    safety_factor = st.slider("Factor adicional sobre equilibrio teórico", 1.0, 1.8, 1.25, 0.05)

if uploads:
    raw = uploads[0].getvalue() if hasattr(uploads[0], "getvalue") else uploads[0]
    rgb = resize_image(raw)

    tab_image, tab_composition, tab_reagents, tab_report = st.tabs(["📷 Imagen y granos", "🔬 Composición", "⚗️ Reactivos", "📄 Informe"])

    with tab_image:
        labels, overlay, areas, diameters, grain_stats = analyze_grains(rgb)
        c1, c2 = st.columns(2)
        c1.image(rgb, caption="Imagen original", use_container_width=True)
        c2.image(overlay, caption="Segmentación experimental de granos", use_container_width=True)

        if len(diameters):
            diam_mm = diameters / pixels_per_mm
            area_mm2 = areas / (pixels_per_mm ** 2)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Granos detectados", f"{len(diameters):,}")
            k2.metric("D50 aproximado", f"{np.median(diam_mm):.3f} mm")
            k3.metric("D10 aproximado", f"{np.percentile(diam_mm, 10):.3f} mm")
            k4.metric("D90 aproximado", f"{np.percentile(diam_mm, 90):.3f} mm")
            fig = px.histogram(x=diam_mm * 1000, nbins=30, labels={"x": "Diámetro (µm)", "y": "Granos"}, title="Distribución de tamaño aparente")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se detectaron granos separados. Ajuste la iluminación, enfoque o calibración.")

    with tab_composition:
        visual = visible_composition(rgb)
        st.subheader("Estimación visual preliminar")
        st.dataframe(pd.DataFrame({"Indicador": visual.keys(), "Valor": visual.values()}), hide_index=True, use_container_width=True)
        st.info("La imagen solo puede estimar rasgos visibles. Para una composición real debe cargarse un ensayo de laboratorio o ingresar los valores analizados.")

        assay_mode = st.radio("Fuente de composición química", ["Solo estimación visual", "Usar ensayo de laboratorio"], horizontal=True)
        if assay_mode == "Solo estimación visual":
            silice_visual = st.slider("SiO₂ visual estimado (%)", 0.0, 100.0, min(float(visual["SiO2 visual"]), 99.9), 0.1)
            visible_impurity = 100 - silice_visual
            defaults = {
                "Fe2O3": visible_impurity * 0.35,
                "Al2O3": visible_impurity * 0.35,
                "CaO": visible_impurity * 0.10,
                "MgO": visible_impurity * 0.05,
                "TiO2": visible_impurity * 0.05,
                "K2O": visible_impurity * 0.10,
            }
        else:
            silice_visual = 100.0
            defaults = {"Fe2O3": 0.10, "Al2O3": 0.20, "CaO": 0.03, "MgO": 0.02, "TiO2": 0.03, "K2O": 0.05}

        st.markdown("##### Composición reportada / calibrada")
        composition = {"SiO2": silice_visual}
        cols = st.columns(3)
        oxide_names = ["Fe2O3", "Al2O3", "CaO", "MgO", "TiO2", "K2O"]
        for i, oxide in enumerate(oxide_names):
            composition[oxide] = cols[i % 3].number_input(oxide + " (%)", min_value=0.0, max_value=100.0, value=float(defaults[oxide]), step=0.01, format="%.3f")
        if assay_mode == "Usar ensayo de laboratorio":
            composition["SiO2"] = cols[0].number_input("SiO₂ de laboratorio (%)", min_value=0.0, max_value=100.0, value=99.5, step=0.01, format="%.3f")

        total = sum(composition.values())
        st.metric("Total registrado", f"{total:.2f} %")
        if abs(total - 100) > 2.0:
            st.warning("La suma de los óxidos no es 100%. Revise SiO₂ y los componentes registrados.")

        fig = px.pie(names=list(composition.keys()), values=list(composition.values()), hole=0.45, title="Composición ingresada")
        st.plotly_chart(fig, use_container_width=True)

    with tab_reagents:
        st.subheader("Estimación preliminar de reactivo equivalente")
        if "composition" not in locals():
            st.info("Complete primero la pestaña Composición.")
        else:
            estimates = acid_equivalents(composition, sample_kg, safety_factor)
            if acid_type == "HCl 37%":
                value = estimates["HCl 37%"]
                units = "kg de solución comercial"
            else:
                value = estimates["H2SO4 98%"]
                units = "kg de solución comercial"
            st.metric(f"{acid_type} equivalente", f"{value:.3f} {units}")
            st.caption(f"Basado en {sample_kg:.3f} kg de arena; el cálculo no incluye consumo cinético real ni pérdidas.")
            st.dataframe(pd.DataFrame([{"Reactivo": k, "kg solución estimados": round(v, 4)} for k, v in estimates.items()]), hide_index=True, use_container_width=True)
            st.error("Esta cifra es teórica y no constituye una receta de lixiviación. Debe verificarse con ensayos menores, relaciones sólido-líquido y análisis de la fase líquida.")

    with tab_report:
        if "composition" not in locals():
            st.info("Complete primero la pestaña Composición.")
        else:
            st.subheader("Resumen exportable")
            report = {
                "Fecha": datetime.now().isoformat(timespec="seconds"),
                "Fuente composición": assay_mode,
                "Masa muestra kg": sample_kg,
                **composition,
            }
            if "estimates" in locals() and acid_type in estimates:
                report[f"Estimacion {acid_type} kg"] = estimates[acid_type] if acid_type in ["HCl 37%", "H2SO4 98%"] else None
            report_df = pd.DataFrame([report]).T.reset_index()
            report_df.columns = ["Campo", "Valor"]
            st.dataframe(report_df, hide_index=True, use_container_width=True)
            csv = report_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Descargar informe CSV", data=csv, file_name="arena_vision_informe.csv", mime="text/csv")
else:
    st.info("Suba o capture una imagen para comenzar.")
    st.markdown("### Preparación recomendada")
    st.markdown("1. Mezcle bien una muestra representativa y seque la arena si hay humedad variable.\n2. Extienda una capa fina sobre papel blanco sin fondo brillante.\n3. Mantenga una altura, zoom y luz constantes.\n4. Incluya una regla o patrón conocido para calibrar mm/píxel.\n5. Compare al menos una muestra con los resultados de XRF, ICP o XRD de laboratorio.")
