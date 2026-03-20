import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import platform
import os

st.set_page_config(
    page_title="Detector Arriba / Abajo",
    page_icon="☝️",
    layout="centered"
)

# ---------- estilo ----------
st.markdown("""
<style>
.main-title {
    font-size: 2.3rem;
    font-weight: 700;
    text-align: center;
}

.result-box {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.15);
    background-color: rgba(255,255,255,0.05);
    text-align: center;
    font-size: 1.4rem;
    margin-top: 10px;
}
.arriba {
    color: #22c55e;
    font-weight: bold;
}
.abajo {
    color: #ef4444;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"


# ---------- cargar modelo ----------
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No se encontró keras_model.h5")
    return load_model(MODEL_PATH, compile=False)


@st.cache_data
def cargar_etiquetas():
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("No se encontró labels.txt")
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return f.readlines()


def limpiar(texto):
    texto = texto.strip()
    if len(texto) > 2:
        return texto[2:]
    return texto


def preparar_imagen(imagen):
    size = (224, 224)

    imagen = imagen.convert("RGB")
    imagen = ImageOps.fit(imagen, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(imagen)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    return imagen, data


# ---------- UI ----------
st.markdown('<div class="main-title">Detector de dedo Arriba / Abajo</div>', unsafe_allow_html=True)

st.write("Python:", platform.python_version())

try:
    model = cargar_modelo()
    class_names = cargar_etiquetas()
except Exception as e:
    st.error(e)
    st.stop()


# ---------- cámara ----------
img_file = st.camera_input("Toma una foto")

if img_file is not None:

    imagen_original = Image.open(img_file)

    imagen_procesada, data = preparar_imagen(imagen_original)

    prediction = model.predict(data, verbose=0)

    index = np.argmax(prediction)

    clase = limpiar(class_names[index])

    confianza = float(prediction[0][index])

    col1, col2 = st.columns(2)

    with col1:
        st.image(imagen_original, caption="Imagen", use_container_width=True)

    with col2:
        st.image(imagen_procesada, caption="Procesada", use_container_width=True)

    # ---------- resultado grande ----------
    if clase.lower() == "arriba":

        st.markdown(
            f"""
            <div class="result-box">
                ☝️ Detectado: <span class="arriba">ARRIBA</span><br>
                Confianza: {confianza:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )

    elif clase.lower() == "abajo":

        st.markdown(
            f"""
            <div class="result-box">
                👇 Detectado: <span class="abajo">ABAJO</span><br>
                Confianza: {confianza:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------- barras ----------
    st.subheader("Probabilidades")

    for i, prob in enumerate(prediction[0]):

        nombre = limpiar(class_names[i])

        st.write(nombre)

        st.progress(float(prob))

        st.caption(f"{float(prob):.2%}")
