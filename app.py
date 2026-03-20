import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import platform

st.set_page_config(
    page_title="Reconocimiento de Imágenes",
    page_icon="📷",
    layout="centered"
)

# Estilo visual ligero
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .subtitle {
        color: #6b7280;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        background-color: rgba(255,255,255,0.03);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .small-text {
        color: #6b7280;
        font-size: 0.95rem;
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

st.markdown('<div class="main-title">Reconocimiento de Imágenes con Teachable Machine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Esta versión usa la cámara normal de Streamlit con tu modelo exportado desde Teachable Machine.</div>',
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
    st.subheader("Información")
    st.write("Toma una foto para clasificarla con el modelo.")
    st.markdown("---")
    st.write("Modelo cargado correctamente.")

img_file_buffer = st.camera_input("Toma una foto")

if img_file_buffer is not None:
    try:
        imagen_original = Image.open(img_file_buffer)
        imagen_procesada, data = preparar_imagen(imagen_original)

        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = limpiar_nombre_clase(class_names[index])
        confidence_score = float(prediction[0][index])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen capturada")
            st.image(imagen_original, use_container_width=True)

        with col2:
            st.subheader("Imagen procesada")
            st.image(imagen_procesada, use_container_width=True)

        st.markdown(
            f"""
            <div class="result-box">
                <h3 style="margin-top:0;">Resultado principal</h3>
                <p><strong>Clase detectada:</strong> {class_name}</p>
                <p><strong>Confianza:</strong> {confidence_score:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Probabilidades")
        predicciones = prediction[0]
        top_indices = np.argsort(predicciones)[::-1]

        for i in top_indices:
            nombre = limpiar_nombre_clase(class_names[i])
            prob = float(predicciones[i])
            st.write(f"**{nombre}**")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"{prob:.2%}")

    except Exception as e:
        st.error(f"Error procesando la imagen: {e}")

else:
    st.markdown(
        '<p class="small-text">Toma una foto para ver la predicción del modelo.</p>',
        unsafe_allow_html=True
    )
