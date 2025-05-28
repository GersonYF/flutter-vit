import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
import time
import av
from io import BytesIO
from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Cargar modelo
TARGET_SIZE = (320, 320)
MODEL_NAME = "apple/deeplabv3-mobilevit-small"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = TFMobileViTForSemanticSegmentation.from_pretrained(MODEL_NAME)
id2label = model.config.id2label
num_classes = len(id2label)
np.random.seed(42)
palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    /* Fondo general */
    .stApp {
        background-color: #1e1e1e;
        color: #f5f5f5;
    }

    /* Títulos */
    h1, h2, h3 {
        color: #4a90e2;
    }

    /* Botones */
    .stButton>button {
        background-color: #e63946;
        color: white;
        border-radius: 8px;
        border: none;
    }

    /* Leyenda */
    .legend-box span {
        color: #f0f0f0;
        font-weight: 500;
    }

    /* Inputs */
    .stTextInput>div>input {
        background-color: #2e2e2e;
        color: #f0f0f0;
    }

    .css-1aumxhk {
        background-color: #2e2e2e;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Segmentación Semántica con MobileViT")

with st.sidebar:
    st.title("⚙️ Configuración")
    modo = st.radio("Modo de entrada", ["📷 Cámara en Vivo", "🖼️ Subir Imagen"])

if modo == "📷 Cámara en Vivo":
    class Segmentador(VideoProcessorBase):
        def __init__(self):
            self.last_segment_time = 0
            self.overlay = None
            self.processing = False
            self.last_segmentation = None
            self.class_stats = None


        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            current_time = time.time()
            self.last_update_time = time.time()

            if current_time - self.last_segment_time > 0.5:
                self.processing = True
                self.last_segment_time = current_time
                
                #Segmentacion
                image_pil = Image.fromarray(img_rgb)
                original_size = image_pil.size
                resized_image = image_pil.resize(TARGET_SIZE, Image.BILINEAR)

                inputs = processor(images=resized_image, return_tensors="tf")
                outputs = model(**inputs)
                logits = outputs.logits

                logits_perm = tf.transpose(logits, perm=[0, 2, 3, 1])
                upsampled = tf.image.resize(logits_perm, size=original_size[::-1], method='bilinear')
                upsampled = tf.transpose(upsampled, perm=[0, 3, 1, 2])
                segmentation = tf.argmax(upsampled, axis=1)[0].numpy().astype(np.uint8)

                seg_rgb = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    seg_rgb[segmentation == label] = color
                seg_resized = cv2.resize(seg_rgb, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                self.overlay = (0.5 * img_rgb + 0.5 * seg_resized).astype(np.uint8)
                self.processing = False
                
                self.last_segmentation = segmentation
                unique, counts = np.unique(segmentation, return_counts=True)
                total = segmentation.size
                self.class_stats = [
                    (label, id2label.get(label, f"Clase_{label}"), (counts[i] / total) * 100)
                    for i, label in enumerate(unique)
                ]

            if self.processing:
                bordered = cv2.copyMakeBorder(img_rgb, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 0, 0])
                return av.VideoFrame.from_ndarray(bordered, format="rgb24")
            elif self.overlay is not None:
                return av.VideoFrame.from_ndarray(self.overlay, format="rgb24")
            else:
                return av.VideoFrame.from_ndarray(img_rgb, format="rgb24")

    ctx = webrtc_streamer(
        key="segm_stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=Segmentador,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )


else:
    uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen Original", use_column_width=True)

        with st.spinner("Procesando..."):
            original_size = image.size
            resized_image = image.resize(TARGET_SIZE, Image.BILINEAR)

            inputs = processor(images=resized_image, return_tensors="tf")
            outputs = model(**inputs)
            logits = outputs.logits

            logits_perm = tf.transpose(logits, perm=[0, 2, 3, 1])
            upsampled = tf.image.resize(logits_perm, size=original_size[::-1], method='bilinear')
            upsampled = tf.transpose(upsampled, perm=[0, 3, 1, 2])
            segmentation = tf.argmax(upsampled, axis=1)[0].numpy().astype(np.uint8)


            seg_rgb = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                seg_rgb[segmentation == label] = color

            overlay = (0.5 * np.array(image) + 0.5 * seg_rgb).astype(np.uint8)
            st.image(overlay, caption="Imagen Segmentada", use_column_width=True)
            
            seg_pil = Image.fromarray(overlay)
            buf = BytesIO()
            seg_pil.save(buf, format="PNG")
            st.download_button("💾 Descargar resultado", buf.getvalue(), file_name="segmentado.png", mime="image/png")


with st.expander("Ver leyenda de clases detectables"):
    st.markdown("Clases que el modelo puede detectar, junto con su color:")

    legend_html = '<div class="legend-box" style="display:flex; flex-wrap:wrap; gap:10px;">'

    for label, name in id2label.items():
        color = palette[int(label)]
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        legend_html += (
            "<div style='display:flex; align-items:center; gap:6px; margin-bottom:6px;'>"
            f"<div style='width:20px; height:20px; background-color:{hex_color}; border-radius:3px;'></div>"
            f"<span>{name}</span>"
            "</div>"
        )

    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

with st.expander("ℹ️ Acerca de esta app"):
    st.write("""
    Esta aplicación realiza segmentación semántica en imágenes o video en vivo usando el modelo MobileViT (deeplabv3).
    
    El modelo es capaz de identificar múltiples objetos y representar cada clase con un color diferente. Puedes usar tu cámara o subir imágenes para probarlo.
    https://github.com/GersonYF/flutter-vit
    """)
