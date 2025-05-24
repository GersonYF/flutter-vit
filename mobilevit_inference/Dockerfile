FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip setuptools wheel

RUN pip3 install --no-cache-dir \
    numpy \
    Pillow==9.5.0 \
    matplotlib==3.7.1 \
    streamlit==1.24.0 \
    protobuf==3.20.3 \
    streamlit-webrtc \
    requests==2.31.0 \
    transformers==4.31.0

COPY app_old.py .

EXPOSE 8080

CMD ["streamlit", "run", "app_old.py", "--server.port=8080", "--server.address=0.0.0.0"]