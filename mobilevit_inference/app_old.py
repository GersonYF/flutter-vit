import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import requests
import tensorflow as tf
from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation
import time

tf.get_logger().setLevel('ERROR')

# Cargar el modelo y procesador
MODEL_NAME = "apple/deeplabv3-mobilevit-small"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = TFMobileViTForSemanticSegmentation.from_pretrained(MODEL_NAME)

# Mapeo de clases y colores (COCO)
id2label = model.config.id2label
np.random.seed(42)
num_classes = len(id2label)
palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
# Interfaz Streamlit
st.title("Segmentación Semántica con MobileViT")
st.write("Sube una imagen para segmentarla con el modelo MobileViT.")

uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen original", use_column_width=True)

    start_time = time.time()

    # Preprocesamiento
    inputs = processor(images=image, return_tensors="tf")
    outputs = model(**inputs)
    logits = outputs.logits  # [1, num_labels, h, w]
    logits_perm = tf.transpose(logits, perm=[0, 2, 3, 1])  # [1, H, W, 21]
    upsampled_logits = tf.image.resize(logits_perm, size=image.size[::-1], method='bilinear')  # [1, H', W', 21]
    upsampled_logits = tf.transpose(upsampled_logits, perm=[0, 3, 1, 2])  # [1, 21, H', W']

    segmentation = tf.argmax(upsampled_logits, axis=1)[0].numpy().astype(np.uint8)

    # Crear overlay
    seg_rgb = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        seg_rgb[segmentation == label] = color

    image_np = np.array(image)
    seg_resized = cv2.resize(seg_rgb, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = (0.5 * image_np + 0.5 * seg_resized).astype(np.uint8)

    elapsed_time = time.time() - start_time

    # Mostrar resultado
    st.image(overlay, caption="Imagen con segmentación", use_column_width=True)
    st.success(f"Tiempo total de segmentación: {elapsed_time:.2f} segundos")

    # Mostrar leyenda
    st.markdown("### Leyenda de clases detectadas:")
    labels_present = np.unique(segmentation)
    for label in labels_present:
        label_name = id2label.get(label, f"Clase desconocida ({label})")
        if label < len(palette):
            color = palette[label]
        else:
            color = np.array([128, 128, 128]) 
        st.markdown(
            f"<div style='display: flex; align-items: center;'>"
            f"<div style='width: 20px; height: 20px; background-color: rgb{tuple(color)}; margin-right: 10px;'></div>"
            f"<div>{label_name}</div></div>",
            unsafe_allow_html=True
        )
