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

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró {MODEL_PATH}")
    return load_model(MODEL_PATH, compile=False)

def cargar_etiquetas():
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"No se encontró {LABELS_PATH}")
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def limpiar_nombre_clase(texto):
    texto = texto.strip()
    if len(texto) > 2 and texto[0].isdigit() and texto[1] == " ":
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

st.title("Detector Arriba / Abajo")
st.write("Versión de Python:", platform.python_version())

try:
    model = cargar_modelo()
    class_names = cargar_etiquetas()
    clases_limpias = [limpiar_nombre_clase(c) for c in class_names]
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

st.subheader("Diagnóstico")
st.write("Ruta del modelo:", os.path.abspath(MODEL_PATH))
st.write("Ruta de labels:", os.path.abspath(LABELS_PATH))
st.write("Etiquetas cargadas:", clases_limpias)
st.write("Salida del modelo:", model.output_shape)

img_file = st.camera_input("Toma una foto")

if img_file is not None:
    imagen_original = Image.open(img_file)
    imagen_procesada, data = preparar_imagen(imagen_original)

    prediction = model.predict(data, verbose=0)[0]

    # Protección por si el modelo y el labels no coinciden
    n_modelo = len(prediction)
    n_labels = len(clases_limpias)

    if n_modelo != n_labels:
        st.error(f"El modelo tiene {n_modelo} salidas, pero labels.txt tiene {n_labels} etiquetas.")
        st.stop()

    index = int(np.argmax(prediction))
    clase = clases_limpias[index]
    confianza = float(prediction[index])

    col1, col2 = st.columns(2)
    with col1:
        st.image(imagen_original, caption="Imagen capturada", use_container_width=True)
    with col2:
        st.image(imagen_procesada, caption="Procesada", use_container_width=True)

    st.subheader("Resultado principal")
    st.write(f"Clase detectada: **{clase}**")
    st.write(f"Confianza: **{confianza:.2%}**")

    st.subheader("Probabilidades")
    for i, prob in enumerate(prediction):
        st.write(clases_limpias[i])
        st.progress(float(prob))
        st.caption(f"{float(prob):.2%}")
