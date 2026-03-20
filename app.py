import streamlit as st
import streamlit.components.v1 as components
import platform

st.set_page_config(
    page_title="Teachable Machine - Reconocimiento de Imágenes",
    page_icon="📷",
    layout="centered"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Reconocimiento de Imágenes con Teachable Machine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Esta versión usa el modelo web exportado desde Teachable Machine</div>',
    unsafe_allow_html=True
)

st.write("Versión de Python:", platform.python_version())

teachable_html = """
<div style="text-align:center; font-family: Arial, sans-serif;">
    <h3 style="margin-bottom: 10px;">Teachable Machine Image Model</h3>

    <button 
        type="button" 
        onclick="init()" 
        style="
            background-color:#111827;
            color:white;
            border:none;
            padding:10px 18px;
            border-radius:10px;
            cursor:pointer;
            font-size:16px;
            margin-bottom:15px;
        "
    >
        Iniciar cámara
    </button>

    <div id="status" style="margin-bottom: 12px; color: #374151;"></div>
    <div id="webcam-container" style="margin-bottom: 15px;"></div>
    <div id="label-container" style="
        text-align:left;
        display:inline-block;
        min-width:260px;
        padding:12px;
        border:1px solid #ddd;
        border-radius:12px;
        background:#f9fafb;
    "></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>

<script type="text/javascript">
    const URL = "./my_model/";

    let model, webcam, labelContainer, maxPredictions;
    let isRunning = false;

    async function init() {
        if (isRunning) return;
        isRunning = true;

        document.getElementById("status").innerHTML = "Cargando modelo y activando cámara...";

        const modelURL = URL + "model.json";
        const metadataURL = URL + "metadata.json";

        try {
            model = await tmImage.load(modelURL, metadataURL);
            maxPredictions = model.getTotalClasses();

            const flip = true;
            webcam = new tmImage.Webcam(260, 260, flip);
            await webcam.setup();
            await webcam.play();
            window.requestAnimationFrame(loop);

            const webcamContainer = document.getElementById("webcam-container");
            webcamContainer.innerHTML = "";
            webcamContainer.appendChild(webcam.canvas);

            labelContainer = document.getElementById("label-container");
            labelContainer.innerHTML = "";

            for (let i = 0; i < maxPredictions; i++) {
                const row = document.createElement("div");
                row.style.marginBottom = "8px";
                row.style.padding = "6px 8px";
                row.style.borderRadius = "8px";
                row.style.background = "white";
                row.style.border = "1px solid #e5e7eb";
                labelContainer.appendChild(row);
            }

            document.getElementById("status").innerHTML = "Cámara activa";
        } catch (error) {
            document.getElementById("status").innerHTML = "Error al cargar el modelo o activar la cámara.";
            console.error(error);
        }
    }

    async function loop() {
        if (!webcam) return;
        webcam.update();
        await predict();
        window.requestAnimationFrame(loop);
    }

    async function predict() {
        const prediction = await model.predict(webcam.canvas);
        prediction.sort((a, b) => b.probability - a.probability);

        for (let i = 0; i < maxPredictions; i++) {
            const classPrediction =
                "<strong>" + prediction[i].className + "</strong>: " + prediction[i].probability.toFixed(2);

            labelContainer.childNodes[i].innerHTML = classPrediction;

            if (i === 0) {
                labelContainer.childNodes[i].style.background = "#dbeafe";
                labelContainer.childNodes[i].style.border = "1px solid #93c5fd";
            } else {
                labelContainer.childNodes[i].style.background = "white";
                labelContainer.childNodes[i].style.border = "1px solid #e5e7eb";
            }
        }
    }
</script>
"""

components.html(teachable_html, height=560)
