FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

ENV DEBIAN_FRONTEND=noninteractive

# Establecer zona horaria (opcional)
RUN ln -fs /usr/share/zoneinfo/America/Bogota /etc/localtime && \
    apt-get update && apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias adicionales necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Instalar paquetes Python adicionales
RUN pip3 install --no-cache-dir \
    huggingface-hub==0.19.4 \
    transformers==4.36.2 \
    streamlit streamlit-webrtc \
    opencv-python-headless \
    av==10.0.0

# Pre-descargar el modelo
RUN python3 -c "import warnings; warnings.filterwarnings('ignore'); \
    from transformers import AutoImageProcessor, MobileViTForSemanticSegmentation; \
    model_name='apple/deeplabv3-mobilevit-small'; \
    AutoImageProcessor.from_pretrained(model_name); \
    MobileViTForSemanticSegmentation.from_pretrained(model_name)"

# Copiar la app
COPY app_1.py .

# Exponer puerto
EXPOSE 8080

# Ejecutar app
CMD ["streamlit", "run", "app_1.py", "--server.port=8080", "--server.address=0.0.0.0"]