import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import tensorflow as tf
from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation

# Configuración del modelo
MODEL_NAME = "apple/deeplabv3-mobilevit-small"

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = TFMobileViTForSemanticSegmentation.from_pretrained(MODEL_NAME)
    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {i: f"Clase_{i}" for i in range(model.config.num_labels)}
    return processor, model, id2label

processor, model, id2label = load_model()

def segment_image(image: Image.Image):
    inputs = processor(images=image, return_tensors="tf")
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = tf.argmax(logits, axis=1)[0].numpy()
    return predictions, logits.shape[1]

def visualize_segmentation(image: Image.Image, predictions: np.ndarray, num_classes: int):
    # Crear un mapa de colores
    cmap = plt.cm.get_cmap('tab20', num_classes) if num_classes <= 20 else plt.cm.get_cmap('viridis', num_classes)
    colors = (cmap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)

    # Crear imagen segmentada
    segmentation_map = colors[predictions]
    segmentation_image = Image.fromarray(segmentation_map)

    # Redimensionar imagen original
    original_image_resized = image.resize(segmentation_image.size)

    # Superposición
    overlay = Image.blend(original_image_resized.convert("RGBA"), segmentation_image.convert("RGBA"), alpha=0.5)

    return original_image_resized, segmentation_image, overlay

def get_class_statistics(predictions: np.ndarray, id2label: dict):
    unique, counts = np.unique(predictions, return_counts=True)
    total = predictions.size
    stats = {int(cls): {"name": id2label.get(int(cls), f"Clase_{cls}"), "count": int(count), "percentage": float(count / total * 100)} for cls, count in zip(unique, counts)}
    return stats

# Interfaz de Streamlit
st.title("🧠 Segmentación Semántica con MobileViT")
st.write("Carga una imagen para realizar la segmentación semántica utilizando el modelo MobileViT.")

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen Original", use_column_width=True)

    with st.spinner("Procesando la imagen..."):
        predictions, num_classes = segment_image(image)
        original_resized, segmentation_image, overlay = visualize_segmentation(image, predictions, num_classes)
        stats = get_class_statistics(predictions, id2label)

    st.subheader("🖼️ Resultados de la Segmentación")
    st.image(segmentation_image, caption="Mapa de Segmentación", use_column_width=True)
    st.image(overlay, caption="Superposición", use_column_width=True)

    st.subheader("📊 Estadísticas de Clases Detectadas")
    for cls_id, info in stats.items():
        st.write(f"Clase {cls_id} ({info['name']}): {info['count']} píxeles ({info['percentage']:.2f}%)")
