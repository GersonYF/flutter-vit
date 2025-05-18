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
    transformers==4.30.2 \
    requests==2.31.0 \
    Pillow==9.5.0 \
    matplotlib==3.7.1 \
    numpy==1.24.3 \
    flask==2.3.2 \
    protobuf==3.20.3

# Pre-descargar el modelo durante la construcción de la imagen
RUN python -c "import warnings; warnings.filterwarnings('ignore'); \
    from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation; \
    model_name='apple/deeplabv3-mobilevit-small'; \
    processor = AutoImageProcessor.from_pretrained(model_name); \
    model = TFMobileViTForSemanticSegmentation.from_pretrained(model_name)"

# Copiar la aplicación
COPY app.py .

# Exponer el puerto
EXPOSE 5000

# Usar el flag -u para ver la salida inmediatamente en los logs
CMD ["python", "-u", "app.py"]
