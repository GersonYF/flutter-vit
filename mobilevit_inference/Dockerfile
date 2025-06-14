# Usar una imagen base de TensorFlow oficial
FROM tensorflow/tensorflow:2.12.0

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-opencv \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    requests==2.31.0 \
    Pillow==9.5.0 \
    matplotlib==3.7.1 \
    numpy==1.24.3 \
    streamlit==1.24.0 \
    protobuf==3.20.3 \
    streamlit-webrtc

# Pre-descargar el modelo durante la construcción de la imagen
RUN python -c "import warnings; warnings.filterwarnings('ignore'); \
    from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation; \
    model_name='apple/deeplabv3-mobilevit-small'; \
    processor = AutoImageProcessor.from_pretrained(model_name); \
    model = TFMobileViTForSemanticSegmentation.from_pretrained(model_name)"

# Copiar la aplicación
COPY combined.py .
COPY .streamlit /root/.streamlit

# Exponer el puerto de Streamlit
EXPOSE 8080

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "combined.py", "--server.port=8080", "--server.address=0.0.0.0"]
