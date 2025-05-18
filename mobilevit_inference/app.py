import os
import io
import base64
import requests
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, jsonify, send_file
import tensorflow as tf
from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation

# Modelo específico para segmentación semántica con MobileViT
MODEL_NAME = "apple/deeplabv3-mobilevit-small"

app = Flask(__name__)

# Configurar TensorFlow para usar memoria de GPU de forma dinámica
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible: {len(gpus)} dispositivo(s)")
    except RuntimeError as e:
        print(f"Error al configurar GPU: {e}")

print(f"Cargando modelo {MODEL_NAME} y procesador...")
try:
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = TFMobileViTForSemanticSegmentation.from_pretrained(MODEL_NAME)
    print(f"Modelo {MODEL_NAME} cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    
def visualize_segmentation(logits, original_image):
    """
    Visualizar resultados de segmentación semántica y devolver la imagen como bytes.
    
    Args:
        logits: Tensor de logits TensorFlow de forma (batch_size, num_classes, height, width)
        original_image: Imagen PIL original
        
    Returns:
        bytes: Imagen de visualización en formato bytes (PNG)
    """
    # Convertir logits TensorFlow a NumPy
    logits_np = logits.numpy()
    
    # Obtener predicciones tomando argmax a lo largo de la dimensión de clases
    predictions = np.argmax(logits_np, axis=1)[0]  # Tomar la primera imagen del batch
    
    # Crear un mapa de colores (un color por clase)
    num_classes = logits_np.shape[1]
    # Usar colores más distintivos para las clases
    cmap = plt.cm.get_cmap('tab20', num_classes) if num_classes <= 20 else plt.cm.get_cmap('viridis', num_classes)
    colors = [cmap(i) for i in range(num_classes)]
    
    # Crear un mapa de segmentación colorizado
    segmentation_map = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
    for class_idx in range(num_classes):
        mask = predictions == class_idx
        for c in range(3):  # Canales RGB
            segmentation_map[:, :, c][mask] = int(colors[class_idx][c] * 255)
    
    # Convertir a imagen PIL
    segmentation_image = Image.fromarray(segmentation_map)
    
    # Redimensionar imagen original para que coincida con el tamaño del mapa de segmentación
    original_image_resized = original_image.resize(
        (segmentation_map.shape[1], segmentation_map.shape[0]), 
        Image.BICUBIC  # Usar BICUBIC para compatibilidad con versiones antiguas de PIL
    )
    
    # Crear una figura con subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Graficar imagen original
    axes[0].imshow(original_image_resized)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Graficar mapa de segmentación
    axes[1].imshow(segmentation_map)
    axes[1].set_title('Mapa de Segmentación')
    axes[1].axis('off')
    
    # Crear una superposición combinada (50% original, 50% segmentación)
    # Método seguro para trabajar con diferentes versiones de PIL
    original_array = np.array(original_image_resized).astype(float)
    segmentation_array = np.array(segmentation_image).astype(float)
    overlay_array = (original_array * 0.5 + segmentation_array * 0.5).astype(np.uint8)
    
    axes[2].imshow(overlay_array)
    axes[2].set_title('Superposición')
    axes[2].axis('off')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar la figura en un buffer de bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que el servicio está funcionando."""
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "backend": "TensorFlow",
        "note": "Modelo entrenado específicamente para segmentación semántica"
    })

@app.route('/segment', methods=['POST'])
def segment_image():
    """
    Endpoint para segmentar una imagen.
    
    Acepta:
    - URL de imagen a través del parámetro 'image_url'
    - Archivo de imagen subido con el nombre 'image_file'
    - Imagen en base64 con el parámetro 'image_base64'
    
    Devuelve:
    - La imagen de visualización en formato PNG
    """
    try:
        # Obtener la imagen de una de las fuentes posibles
        if 'image_url' in request.form:
            # Obtener imagen desde URL
            image_url = request.form['image_url']
            response = requests.get(image_url, stream=True)
            if response.status_code != 200:
                return jsonify({"error": f"Error al obtener la imagen de la URL: {response.status_code}"}), 400
            image = Image.open(response.raw)
            
        elif 'image_file' in request.files:
            # Obtener imagen desde archivo subido
            image_file = request.files['image_file']
            image = Image.open(image_file)
            
        elif 'image_base64' in request.form:
            # Obtener imagen desde string base64
            image_data = base64.b64decode(request.form['image_base64'])
            image = Image.open(io.BytesIO(image_data))
            
        else:
            return jsonify({"error": "No se proporcionó ninguna imagen. Use 'image_url', 'image_file' o 'image_base64'"}), 400
        
        # Imprimir información sobre la imagen recibida
        print(f"Imagen recibida: {image.format}, tamaño: {image.size}, modo: {image.mode}")
        
        # Procesar la imagen
        inputs = image_processor(images=image, return_tensors="tf")
        
        # Realizar inferencia con TensorFlow
        outputs = model(**inputs)
        
        # Obtener logits
        logits = outputs.logits
        print(f"Segmentación completada. Forma de logits: {logits.shape}")
        
        # Visualizar resultados
        visualization_buffer = visualize_segmentation(logits, image)
        
        # Determinar el formato de respuesta
        response_format = request.args.get('format', 'image')
        
        if response_format == 'json':
            # Devolver imagen como base64 en JSON
            image_base64 = base64.b64encode(visualization_buffer.getvalue()).decode('utf-8')
            return jsonify({
                "segmentation_image": image_base64,
                "classes": logits.shape[1],
                "width": logits.shape[3],
                "height": logits.shape[2],
                "model": MODEL_NAME
            })
        else:
            # Devolver imagen directamente
            return send_file(
                visualization_buffer,
                mimetype='image/png',
                as_attachment=False,
                download_name='segmentation_result.png'
            )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/segment_url', methods=['GET'])
def segment_url():
    """
    Endpoint simplificado para segmentar una imagen desde URL usando método GET.
    
    Parámetros de consulta:
    - url: URL de la imagen a segmentar
    
    Devuelve:
    - La imagen de visualización en formato PNG
    """
    try:
        image_url = request.args.get('url')
        if not image_url:
            return jsonify({"error": "Parámetro 'url' no proporcionado"}), 400
            
        # Obtener imagen desde URL
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            return jsonify({"error": f"Error al obtener la imagen de la URL: {response.status_code}"}), 400
        image = Image.open(response.raw)
        
        # Procesar la imagen
        inputs = image_processor(images=image, return_tensors="tf")
        
        # Realizar inferencia con TensorFlow
        outputs = model(**inputs)
        
        # Obtener logits
        logits = outputs.logits
        
        # Visualizar resultados
        visualization_buffer = visualize_segmentation(logits, image)
        
        # Devolver imagen directamente
        return send_file(
            visualization_buffer,
            mimetype='image/png',
            as_attachment=False,
            download_name='segmentation_result.png'
        )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ejecutar servidor Flask
    port = int(os.environ.get('PORT', 5000))
    print(f"Iniciando servidor en http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
