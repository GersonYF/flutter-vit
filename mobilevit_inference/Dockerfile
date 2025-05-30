# Imagen base para Jetson con soporte para CUDA 10.2 y Python 3.6
FROM nvcr.io/nvidia/l4t-base:r32.6.1
# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    build-essential \
    ffmpeg \
    cython3 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavresample-dev \
    libsrtp2-dev \
    libopus-dev \
    python3-opencv \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Entorno con Python 3.8
RUN python3.8 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar pip y paquetes
RUN pip install --upgrade pip setuptools wheel cython

RUN pip install av==7.0.0

RUN pip install \
    transformers \
    requests \
    Pillow \
    matplotlib \
    numpy \
    streamlit \
    streamlit-webrtc \
    protobuf

RUN python -c "import warnings; warnings.filterwarnings('ignore'); \
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
