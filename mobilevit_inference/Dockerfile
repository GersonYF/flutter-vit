# Imagen base para Jetson con soporte para CUDA 10.2 y Python 3.8
FROM nvcr.io/nvidia/l4t-base:r32.6.1

ENV DEBIAN_FRONTEND=noninteractive

# Establecer zona horaria para evitar bloqueo de tzdata
RUN ln -fs /usr/share/zoneinfo/America/Bogota /etc/localtime && \
    apt-get update && apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-distutils \
    python3.8-venv \
    cython3 \
    python3-opencv \
    libgl1-mesa-glx \
    ffmpeg \
    pkg-config \
    libsm6 \
    libxext6 \
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
    curl \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Setup
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py

# Instalar pip y paquetes
RUN python3.8 -m pip install --upgrade pip setuptools wheel

# Instalar transformers de Python
RUN git clone https://github.com/huggingface/transformers && \
    cd transformers && \
    git checkout v4.36.2 && \
    python3.8 -m pip install .

RUN python3.8 -m pip install --no-cache-dir \
    cython==0.29.21 \
    Pillow \
    requests \
    matplotlib \
    numpy \
    streamlit \
    streamlit-webrtc \
    protobuf

# Instalar PyAV desde pip (requiere compilación con libav*)
RUN python3.8 -m pip install av==7.0.0

# Pre-descargar el modelo durante la construcción de la imagen
RUN python3.8 -c "import warnings; warnings.filterwarnings('ignore'); \
    from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation; \
    model_name='apple/deeplabv3-mobilevit-small'; \
    AutoImageProcessor.from_pretrained(model_name); \
    TFMobileViTForSemanticSegmentation.from_pretrained(model_name)"

# Copiar la aplicación
COPY app_1.py .

# Exponer puerto para Streamlit
EXPOSE 8080

# Ejecutar la app
CMD ["python3.8", "-m", "streamlit", "run", "app_1.py", "--server.port=8080", "--server.address=0.0.0.0"]