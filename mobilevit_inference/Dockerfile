# Imagen base para Jetson con soporte para CUDA 10.2 y Python 3.6
FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3
# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-opencv \
    libgl1-mesa-glx \
    ffmpeg \
    pkg-config \
    libsm6 \
    libxext6 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavfilter-dev \
    libavdevice-dev \
    libavutil-dev \
    libavresample-dev \
    libsrtp2-dev \
    libopus-dev \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Instalar pip y paquetes
RUN python3 -m pip install --upgrade pip setuptools wheel cython

RUN python3 -m pip install --no-cache-dir \
    av==8.0.3 \
    transformers \
    requests \
    Pillow \
    matplotlib \
    numpy \
    streamlit
    
RUN python3 -m pip install --no-cache-dir streamlit-webrtc protobuf

RUN python3 -c "import warnings; warnings.filterwarnings('ignore'); \
    from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation; \
    model_name='apple/deeplabv3-mobilevit-small'; \
    AutoImageProcessor.from_pretrained(model_name); \
    TFMobileViTForSemanticSegmentation.from_pretrained(model_name)"

# Copiar la aplicación
COPY app_1.py .

# Exponer puerto para Streamlit
EXPOSE 8080

# Ejecutar la app
CMD ["streamlit", "run", "app_1.py", "--server.port=8080", "--server.address=0.0.0.0"]
