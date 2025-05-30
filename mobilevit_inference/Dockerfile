FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

ENV DEBIAN_FRONTEND=noninteractive

# Eliminar entradas de Kitware y configurar zona horaria
RUN find /etc/apt/ -name '*.list' -exec sed -i '/kitware/d' {} + && \
    ln -fs /usr/share/zoneinfo/America/Bogota /etc/localtime && \
    apt-get update || true && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Actualizar pip y setuptools
RUN pip3 install --upgrade pip setuptools wheel

# Instalar paquetes Python compatibles con Python 3.6 en ARM64
RUN pip3 install --no-cache-dir \
    huggingface-hub==0.4.0 \
    transformers==4.17.0 \
    streamlit \
    streamlit-webrtc \
    opencv-python-headless \
    av==8.0.3

# Pre-descargar modelo (puede ajustar en caso de incompatibilidades)
RUN python3 -c "import warnings; warnings.filterwarnings('ignore'); \
    from transformers import AutoImageProcessor, MobileViTForSemanticSegmentation; \
    model_name='apple/deeplabv3-mobilevit-small'; \
    AutoImageProcessor.from_pretrained(model_name); \
    MobileViTForSemanticSegmentation.from_pretrained(model_name)"

# Copiar la app
COPY app_1.py .

EXPOSE 8080

CMD ["streamlit", "run", "app_1.py", "--server.port=8080", "--server.address=0.0.0.0"]
