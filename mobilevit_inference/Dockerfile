# Imagen base para Jetson con soporte para CUDA 10.2 y Python 3.6
FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.7-py3
# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar pip y paquetes
RUN python3 -m ensurepip && python3 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir \
    transformers==4.30.0 \
    requests==2.31.0 \
    Pillow==9.5.0 \
    matplotlib==3.5.3 \
    numpy==1.21.6 \
    streamlit==1.22.0 \
    protobuf==3.20.3 \
    streamlit-webrtc==0.45.0 \
    av==10.0.0

RUN python3 -c "import warnings; warnings.filterwarnings('ignore'); \
    from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation; \
    model_name='apple/deeplabv3-mobilevit-small'; \
    processor = AutoImageProcessor.from_pretrained(model_name); \
    model = TFMobileViTForSemanticSegmentation.from_pretrained(model_name)"

# Copiar la aplicación
COPY app_1.py .

# Exponer puerto para Streamlit
EXPOSE 8080

# Ejecutar la app
CMD ["streamlit", "run", "app_1.py", "--server.port=8080", "--server.address=0.0.0.0"]
