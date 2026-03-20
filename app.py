import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import platform
import os

st.set_page_config(
    page_title="Reconocimiento de Imágenes",
    page_icon="📷",
    layout="wide"
)

# ---------------------------
# Estilos suaves
# ---------------------------
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.subtext {
    color: #6b7280;
    margin-bottom: 1rem;
}
.result-box {
    padding: 14px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.03);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Cargar modelo
# ---------------------------
@st.cache_resource
def cargar_modelo():
    return load_model("keras_model.h5", compile=False)

# ---------------------------
# Cargar etiquetas
# ---------------------------
def cargar_etiquetas(ruta="labels.txt"):
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            etiquetas = [line.strip() for line in f.readlines()]
        return etiquetas
    return None

# ---------------------------
# Preparar imagen al estilo Teachable Machine
# ---------------------------
def preparar_imagen(img_pil):
    size = (224, 224)

    # Convierte a RGB por seguridad
    img_pil = img_pil.convert("RGB")

    # Teachable Machine suele funcionar mejor con contain + fondo
    image = ImageOps.fit(img_pil, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image).astype(np.float32)

    # Normalización típica de Teachable Machine
    normalized_image_array = (image_array / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    return image, data

# ---------------------------
# Predicción
# ---------------------------
def predecir(modelo, data):
    prediction = modelo.predict(data, verbose=0)
    return prediction[0]

# ---------------------------
# App
# ---------------------------
st.markdown('<div class="main-title">Reconocimiento de Imágenes con Teachable Machine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtext">Usa tu modelo exportado desde Teachable Machine en formato Keras para clasificar imágenes desde cámara o archivo.</div>',
    unsafe_allow_html=True
)

st.write("Versión de Python:", platform.python_version())

modelo = cargar_modelo()
etiquetas = cargar_etiquetas()

with st.sidebar:
    st.subheader("Configuración")
    fuente = st.radio("Selecciona la fuente", ["Cámara", "Subir imagen"])

    st.markdown("---")
    st.subheader("Información")
    if etiquetas:
        st.success(f"Se cargaron {len(etiquetas)} clases desde labels.txt")
    else:
        st.warning("No se encontró labels.txt. Se mostrarán nombres genéricos.")

# Imagen de ejemplo opcional
if os.path.exists("OIG5.jpg"):
    st.image("OIG5.jpg", width=280, caption="Imagen de referencia")

archivo = None

if fuente == "Cámara":
    archivo = st.camera_input("Toma una foto")
else:
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])

if archivo is not None:
    img = Image.open(archivo)

    imagen_preparada, data = preparar_imagen(img)
    probabilidades = predecir(modelo, data)

    indice_ganador = int(np.argmax(probabilidades))
    confianza_ganadora = float(probabilidades[indice_ganador])

    if etiquetas and indice_ganador < len(etiquetas):
        clase_ganadora = etiquetas[indice_ganador]
    else:
        clase_ganadora = f"Clase {indice_ganador}"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagen original")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Imagen procesada")
        st.image(imagen_preparada, use_container_width=True)

    st.markdown("### Resultado principal")
    st.markdown(
        f"""
        <div class="result-box">
            <h3 style="margin-top:0;">Clase detectada: {clase_ganadora}</h3>
            <p style="font-size:1.1rem;">Probabilidad: <strong>{confianza_ganadora:.2%}</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Probabilidades por clase")

    resultados = []
    for i, prob in enumerate(probabilidades):
        if etiquetas and i < len(etiquetas):
            nombre_clase = etiquetas[i]
        else:
            nombre_clase = f"Clase {i}"

        # Si labels.txt viene como "0 Izquierda", limpiamos un poco opcionalmente
        resultados.append({
            "Clase": nombre_clase,
            "Probabilidad": float(prob)
        })

    resultados = sorted(resultados, key=lambda x: x["Probabilidad"], reverse=True)

    for r in resultados:
        st.write(f"**{r['Clase']}**")
        st.progress(min(max(r["Probabilidad"], 0.0), 1.0))
        st.caption(f"{r['Probabilidad']:.2%}")

    # Si quieres seguir la lógica vieja del profesor con umbral > 0.5
    st.markdown("### Interpretación rápida")
    detectadas = [r for r in resultados if r["Probabilidad"] > 0.5]

    if detectadas:
        for r in detectadas:
            st.success(f"{r['Clase']} detectada con probabilidad de {r['Probabilidad']:.2%}")
    else:
        st.info("Ninguna clase superó el umbral de 50%.")
