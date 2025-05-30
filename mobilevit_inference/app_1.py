import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
# import tensorflow as tf
import time
import av
from transformers import AutoFeatureExtractor, MobileViTForSemanticSegmentation
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# tf.get_logger().setLevel('ERROR')

# Cargar el modelo y procesador
MODEL_NAME = "apple/deeplabv3-mobilevit-small"
processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = MobileViTForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()

# Enviar modelo a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Mapeo de clases y colores (COCO)
id2label = model.config.id2label
np.random.seed(42)
num_classes = len(id2label)
palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

# Interfaz Streamlit
st.set_page_config(layout="wide")
st.title("Segmentación Semántica con MobileViT")

# Layout dividido
col1, col2 = st.columns([4, 1])
timer_placeholder = col2.empty()

# Mostrar leyenda de clases

with st.expander("🎨 Leyenda de Clases"):
    legend_html = ""
    for label_id, label_name in id2label.items():
        color = palette[int(label_id)]
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        legend_html += f"""
            <div style='display: flex; align-items: center; margin-bottom: 4px;'>
                <div style='width: 16px; height: 16px; background-color: {hex_color}; margin-right: 8px; border: 1px solid #000;'></div>
                <span>{label_name}</span>
            </div>
        """
    st.markdown(legend_html, unsafe_allow_html=True)


class Segmentador(VideoProcessorBase):
    def __init__(self):
        self.last_segment_time = 0
        self.overlay = None
        self.processing = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        current_time = time.time()

        if current_time - self.last_segment_time > 0.1:
            self.processing = True
            self.last_segment_time = current_time

            # Segmentación
            image_pil = Image.fromarray(img_rgb)
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                upsampled = torch.nn.functional.interpolate(
                    logits,
                    size=image_pil.size[::-1],  # (width, height)
                    mode="bilinear",
                    align_corners=False,
                )
                segmentation = torch.argmax(upsampled, dim=1)[0].cpu().numpy().astype(np.uint8)

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

# Streamlit WebRTC
webrtc_streamer(
    key="segm",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=Segmentador,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)