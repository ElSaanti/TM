import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import platform
import os
import io

# Evita notación científica en numpy
np.set_printoptions(suppress=True)

st.set_page_config(
    page_title="Clasificador de Imágenes",
    page_icon="📷",
    layout="wide"
)

# Mejora visual ligera
st.markdown("""
<style>
    .title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        margin-bottom: 1rem;
    }
    .box {
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        background-color: rgba(255,255,255,0.03);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    return load_model("keras_Model.h5", compile=False)

@st.cache_data
def cargar_etiquetas():
    with open("labels.txt", "r", encoding="utf-8") as f:
        return f.readlines()

def preparar_imagen(imagen):
    size = (224, 224)
    imagen = imagen.convert("RGB")
    imagen = ImageOps.fit(imagen, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(imagen)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    return imagen, data

def limpiar_nombre_clase(texto):
    texto = texto.strip()
    if len(texto) > 2 and texto[0].isdigit() and texto[1] == " ":
        return texto[2:]
    return texto

st.markdown('<div class="title">Reconocimiento de Imágenes con Teachable Machine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Clasifica imágenes usando un modelo exportado desde Teachable Machine.</div>',
    unsafe_allow_html=True
)

st.write("Versión de Python:", platform.python_version())

try:
    model = cargar_modelo()
    class_names = cargar_etiquetas()
except Exception as e:
    st.error(f"Error cargando el modelo o las etiquetas: {e}")
    st.stop()

with st.sidebar:
    st.subheader("Opciones")
    fuente = st.radio("Selecciona la fuente de imagen", ["Cámara", "Subir imagen"])
    st.markdown("---")
    st.info("Esta app usa el modelo exportado desde Teachable Machine en formato Keras.")

archivo = None

if fuente == "Cámara":
    archivo = st.camera_input("Toma una foto")
else:
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])

if archivo is not None:
    try:
        imagen_original = Image.open(archivo)
        imagen_procesada, data = preparar_imagen(imagen_original)

        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = limpiar_nombre_clase(class_names[index])
        confidence_score = float(prediction[0][index])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen original")
            st.image(imagen_original, use_container_width=True)

        with col2:
            st.subheader("Imagen procesada para el modelo")
            st.image(imagen_procesada, use_container_width=True)

        st.subheader("Resultado principal")
        st.markdown(
            f"""
            <div class="box">
                <h3 style="margin-top:0;">Clase detectada: {class_name}</h3>
                <p style="font-size:1.1rem;">Confianza: <strong>{confidence_score:.2%}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Nuevo agregado útil: Top 3 predicciones
        st.subheader("Top 3 predicciones")
        predicciones = prediction[0]
        top_indices = np.argsort(predicciones)[::-1][:3]

        for i in top_indices:
            nombre = limpiar_nombre_clase(class_names[i])
            prob = float(predicciones[i])
            st.write(f"**{nombre}**")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"{prob:.2%}")

        # Agregado extra útil: tabla completa de resultados
        with st.expander("Ver todas las clases"):
            for i, prob in enumerate(predicciones):
                nombre = limpiar_nombre_clase(class_names[i])
                st.write(f"{nombre}: {float(prob):.4f}")

    except Exception as e:
        st.error(f"Error procesando la imagen: {e}")

else:
    st.caption("Toma una foto o sube una imagen para comenzar.")
