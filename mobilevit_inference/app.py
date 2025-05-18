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
    
    # Intentar obtener información de las clases de la configuración
    config = model.config
    id2label = getattr(config, 'id2label', None)
    if id2label:
        print(f"Etiquetas de clase encontradas: {len(id2label)} clases")
    else:
        print("No se encontraron etiquetas de clase en la configuración del modelo")
        # Crear un diccionario de etiquetas genéricas
        id2label = {i: f"Clase_{i}" for i in range(model.config.num_labels)}
    
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    id2label = {}

def get_class_statistics(predictions, num_classes):
    """
    Calcular estadísticas sobre las clases presentes en la predicción.
    
    Args:
        predictions: Array NumPy con las predicciones de clase para cada píxel
        num_classes: Número total de clases
        
    Returns:
        dict: Estadísticas de clases {class_id: {name, count, percentage}}
    """
    # Contar píxeles por clase
    total_pixels = predictions.size
    class_stats = {}
    
    for class_id in range(num_classes):
        # Contar píxeles de esta clase
        count = np.sum(predictions == class_id)
        
        # Solo incluir clases presentes en la imagen
        if count > 0:
            percentage = (count / total_pixels) * 100
            class_stats[int(class_id)] = {
                "name": id2label.get(class_id, f"Clase_{class_id}"),
                "pixel_count": int(count),
                "percentage": float(percentage)
            }
    
    return class_stats
    
def visualize_segmentation(logits, original_image):
    """
    Visualizar resultados de segmentación semántica y devolver la imagen como bytes.
    
    Args:
        logits: Tensor de logits TensorFlow de forma (batch_size, num_classes, height, width)
        original_image: Imagen PIL original
        
    Returns:
        tuple: (buffer de imagen, estadísticas de clases, mapa de predicciones)
    """
    # Convertir logits TensorFlow a NumPy
    logits_np = logits.numpy()
    
    # Obtener predicciones tomando argmax a lo largo de la dimensión de clases
    predictions = np.argmax(logits_np, axis=1)[0]  # Tomar la primera imagen del batch
    
    # Calcular estadísticas de clases
    num_classes = logits_np.shape[1]
    class_stats = get_class_statistics(predictions, num_classes)
    
    # Crear un mapa de colores (un color por clase)
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
    original_array = np.array(original_image_resized).astype(float)
    segmentation_array = np.array(segmentation_image).astype(float)
    overlay_array = (original_array * 0.5 + segmentation_array * 0.5).astype(np.uint8)
    
    axes[2].imshow(overlay_array)
    axes[2].set_title('Superposición')
    axes[2].axis('off')
    
    # Añadir leyenda con las clases principales
    # Ordenar clases por porcentaje (mayor a menor)
    sorted_classes = sorted(
        class_stats.items(), 
        key=lambda x: x[1]['percentage'], 
        reverse=True
    )
    
    # Añadir leyenda solo para las 5 clases principales o menos
    legend_elements = []
    for class_id, stats in sorted_classes[:5]:
        color = colors[class_id]
        label = f"{stats['name']} ({stats['percentage']:.1f}%)"
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                               markerfacecolor=color, markersize=10, label=label))
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='lower center', ncol=min(5, len(legend_elements)))
    
    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15 if legend_elements else 0.05)
    
    # Guardar la figura en un buffer de bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf, class_stats, predictions

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que el servicio está funcionando."""
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "backend": "TensorFlow",
        "num_classes": model.config.num_labels,
        "id2label": id2label
    })

@app.route('/segment', methods=['POST'])
def segment_image():
    """
    Endpoint para segmentar una imagen.
    
    Acepta:
    - URL de imagen a través del parámetro 'image_url'
    - Archivo de imagen subido con el nombre 'image_file'
    - Imagen en base64 con el parámetro 'image_base64'
    
    Parámetros de consulta opcionales:
    - format: 'image' (por defecto) o 'json'
    - include_classes: 'true' para incluir información de clases (solo con format=json)
    - include_map: 'true' para incluir el mapa de clases (solo con format=json)
    
    Devuelve:
    - La imagen de visualización en formato PNG o datos estructurados en JSON
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
        
        # Visualizar resultados y obtener estadísticas
        visualization_buffer, class_stats, predictions = visualize_segmentation(logits, image)
        
        # Determinar el formato de respuesta
        response_format = request.args.get('format', 'image')
        include_classes = request.args.get('include_classes', 'false').lower() == 'true'
        include_map = request.args.get('include_map', 'false').lower() == 'true'
        
        if response_format == 'json':
            # Devolver imagen como base64 en JSON junto con información adicional
            image_base64 = base64.b64encode(visualization_buffer.getvalue()).decode('utf-8')
            
            response_data = {
                "segmentation_image": image_base64,
                "model": MODEL_NAME,
                "image_size": {
                    "width": image.width,
                    "height": image.height
                },
                "segmentation_size": {
                    "width": logits.shape[3],
                    "height": logits.shape[2],
                    "classes": logits.shape[1]
                }
            }
            
            # Incluir información de clases si se solicita
            if include_classes:
                response_data["classes"] = class_stats
            
            # Incluir mapa de clases si se solicita
            if include_map:
                response_data["class_map"] = predictions.tolist()
                
            return jsonify(response_data)
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
    - format: 'image' (por defecto) o 'json'
    - include_classes: 'true' para incluir información de clases (solo con format=json)
    - include_map: 'true' para incluir el mapa de clases (solo con format=json)
    
    Devuelve:
    - La imagen de visualización en formato PNG o datos estructurados en JSON
    """
    try:
        # Obtener URL de la imagen
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
        
        # Visualizar resultados y obtener estadísticas
        visualization_buffer, class_stats, predictions = visualize_segmentation(logits, image)
        
        # Determinar el formato de respuesta
        response_format = request.args.get('format', 'image')
        include_classes = request.args.get('include_classes', 'false').lower() == 'true'
        include_map = request.args.get('include_map', 'false').lower() == 'true'
        
        if response_format == 'json':
            # Devolver imagen como base64 en JSON junto con información adicional
            image_base64 = base64.b64encode(visualization_buffer.getvalue()).decode('utf-8')
            
            response_data = {
                "segmentation_image": image_base64,
                "model": MODEL_NAME,
                "image_size": {
                    "width": image.width,
                    "height": image.height
                },
                "segmentation_size": {
                    "width": logits.shape[3],
                    "height": logits.shape[2],
                    "classes": logits.shape[1]
                }
            }
            
            # Incluir información de clases si se solicita
            if include_classes:
                response_data["classes"] = class_stats
            
            # Incluir mapa de clases si se solicita
            if include_map:
                response_data["class_map"] = predictions.tolist()
                
            return jsonify(response_data)
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

@app.route('/classes', methods=['GET'])
def get_classes():
    """
    Endpoint para obtener información sobre las clases que el modelo puede detectar.
    
    Devuelve:
    - Lista de clases con sus IDs y nombres
    """
    try:
        return jsonify({
            "model": MODEL_NAME,
            "num_classes": model.config.num_labels,
            "classes": id2label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ejecutar servidor Flask
    port = int(os.environ.get('PORT', 5000))
    print(f"Iniciando servidor en http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
