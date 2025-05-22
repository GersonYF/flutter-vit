import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
import time
import av
from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Cargar modelo
MODEL_NAME = "apple/deeplabv3-mobilevit-small"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = TFMobileViTForSemanticSegmentation.from_pretrained(MODEL_NAME)
id2label = model.config.id2label
num_classes = len(id2label)
np.random.seed(42)
palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

st.set_page_config(layout="wide")
st.title("🧠 Segmentación Semántica con MobileViT")

# Modo selección
modo = st.radio("Selecciona el modo de entrada", ["📷 Cámara en Vivo", "🖼️ Subir Imagen"])

if modo == "📷 Cámara en Vivo":
    class Segmentador(VideoProcessorBase):
        def __init__(self):
            self.last_segment_time = 0
            self.overlay = None
            self.processing = False

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            current_time = time.time()

            if current_time - self.last_segment_time > 3:
                self.processing = True
                self.last_segment_time = current_time
                image_pil = Image.fromarray(img_rgb)
                inputs = processor(images=image_pil, return_tensors="tf")
                outputs = model(**inputs)
                logits = outputs.logits
                logits_perm = tf.transpose(logits, perm=[0, 2, 3, 1])
                upsampled = tf.image.resize(logits_perm, size=image_pil.size[::-1], method='bilinear')
                upsampled = tf.transpose(upsampled, perm=[0, 3, 1, 2])
                segmentation = tf.argmax(upsampled, axis=1)[0].numpy().astype(np.uint8)
                seg_rgb = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    seg_rgb[segmentation == label] = color
                seg_resized = cv2.resize(seg_rgb, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                self.overlay = (0.5 * img_rgb + 0.5 * seg_resized).astype(np.uint8)
                self.processing = False

            if self.processing:
                bordered = cv2.copyMakeBorder(img_rgb, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 0, 0])
                return av.VideoFrame.from_ndarray(bordered, format="rgb24")
            elif self.overlay is not None:
                return av.VideoFrame.from_ndarray(self.overlay, format="rgb24")
            else:
                return av.VideoFrame.from_ndarray(img_rgb, format="rgb24")

    webrtc_streamer(
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
            inputs = processor(images=image, return_tensors="tf")
            outputs = model(**inputs)
            logits = outputs.logits
            logits_perm = tf.transpose(logits, perm=[0, 2, 3, 1])
            upsampled = tf.image.resize(logits_perm, size=image.size[::-1], method='bilinear')
            upsampled = tf.transpose(upsampled, perm=[0, 3, 1, 2])
            segmentation = tf.argmax(upsampled, axis=1)[0].numpy().astype(np.uint8)

            seg_rgb = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                seg_rgb[segmentation == label] = color

            overlay = (0.5 * np.array(image) + 0.5 * seg_rgb).astype(np.uint8)
            st.image(overlay, caption="Imagen Segmentada", use_column_width=True)

            st.subheader("📋 Leyenda de clases detectadas:")
            labels_present = np.unique(segmentation)
            for label in labels_present:
                label_name = id2label.get(label, f"Clase_{label}")
                color = palette[label] if label < len(palette) else np.array([128, 128, 128])
                st.markdown(
                    f"<div style='display: flex; align-items: center;'>"
                    f"<div style='width: 20px; height: 20px; background-color: rgb{tuple(color)}; margin-right: 10px;'></div>"
                    f"<div>{label_name}</div></div>",
                    unsafe_allow_html=True
                )
