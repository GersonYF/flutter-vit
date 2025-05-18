# API de Segmentación Semántica con DeepLabV3 MobileViT

## Descripción
Esta aplicación ofrece una API REST para realizar segmentación semántica en imágenes utilizando el modelo DeepLabV3 MobileViT. La segmentación semántica clasifica cada píxel de una imagen en diferentes categorías, permitiendo identificar y separar objetos en la imagen.

## Contenido del Proyecto
- `app.py`: Servidor Flask que implementa la API
- `Dockerfile`: Configuración para crear la imagen de Docker
- `test_client.py`: Cliente de prueba para la API

## Requisitos
- Docker instalado en tu sistema
- Conexión a internet (para descargar la imagen de Docker y el modelo)
- (Opcional) GPU compatible con CUDA para un procesamiento más rápido

## Instalación y Ejecución

### 1. Construir la imagen de Docker
```bash
docker build -t segmentacion-api .
```

### 2. Ejecutar el contenedor
```bash
docker run -p 5000:5000 --rm segmentacion-api
```

Si tienes una GPU compatible con CUDA:
```bash
docker run -p 5000:5000 --gpus all --rm segmentacion-api
```

## Endpoints de la API

### Verificación de Estado
```
GET /health
```
Devuelve el estado de la API y la información del modelo.

### Obtener Información de Clases
```
GET /classes
```
Devuelve la lista de todas las clases que el modelo puede detectar, con sus IDs y nombres.

### Segmentación de Imagen (POST)
```
POST /segment
```
Acepta uno de los siguientes parámetros:
- `image_url`: URL de una imagen para segmentar
- `image_file`: Archivo de imagen para subir
- `image_base64`: Imagen codificada en base64

Parámetros de consulta opcionales:
- `format=json`: Devuelve la imagen como base64 en formato JSON
- `include_classes=true`: Incluye estadísticas detalladas de las clases detectadas
- `include_map=true`: Incluye el mapa completo de clases (matriz de IDs de clase por píxel)

### Segmentación de Imagen por URL (GET)
```
GET /segment_url?url={URL_DE_LA_IMAGEN}
```
Endpoint simplificado para segmentar una imagen directamente desde una URL.

También acepta los parámetros opcionales:
- `format=json`: Devuelve la imagen como base64 en formato JSON
- `include_classes=true`: Incluye estadísticas detalladas de las clases detectadas
- `include_map=true`: Incluye el mapa completo de clases (matriz de IDs de clase por píxel)

## Ejemplos de Uso

### Usando curl

#### 1. Verificar que la API está funcionando
```bash
curl http://localhost:5000/health
```

#### 2. Obtener la lista de clases que el modelo puede detectar
```bash
curl http://localhost:5000/classes
```

#### 3. Segmentar una imagen desde URL (método GET)
```bash
curl -o resultado.png "http://localhost:5000/segment_url?url=http://images.cocodataset.org/val2017/000000039769.jpg"
```

#### 4. Segmentar una imagen desde URL con información de clases
```bash
curl "http://localhost:5000/segment_url?url=http://images.cocodataset.org/val2017/000000039769.jpg&format=json&include_classes=true"
```

#### 5. Segmentar una imagen desde URL (método POST)
```bash
curl -X POST -F "image_url=http://images.cocodataset.org/val2017/000000039769.jpg" -o resultado.png http://localhost:5000/segment
```

#### 6. Segmentar una imagen subida
```bash
curl -X POST -F "image_file=@/ruta/a/tu/imagen.jpg" -o resultado.png http://localhost:5000/segment
```

#### 7. Obtener resultados en formato JSON con mapa de clases
```bash
curl -X POST -F "image_url=http://images.cocodataset.org/val2017/000000039769.jpg" "http://localhost:5000/segment?format=json&include_map=true"
```

### Usando el cliente de prueba

El proyecto incluye un script cliente para probar la API:

```bash
# Probar con una imagen desde URL
python test_client.py --api http://localhost:5000 --url http://images.cocodataset.org/val2017/000000039769.jpg

# Probar con una imagen local
python test_client.py --api http://localhost:5000 --file tu_imagen.jpg
```

## Formato de Respuesta JSON

Al usar `format=json` y `include_classes=true`, la API devuelve:

```json
{
  "segmentation_image": "base64_image_data...",
  "model": "apple/deeplabv3-mobilevit-small",
  "image_size": {
    "width": 640, "height": 480
  },
  "segmentation_size": {
    "width": 640, "height": 480, "classes": 21
  },
  "classes": {
    "0": {
      "name": "background",
      "pixel_count": 189201,
      "percentage": 61.55
    },
    "15": {
      "name": "person",
      "pixel_count": 86420,
      "percentage": 28.12
    },
    "17": {
      "name": "cat",
      "pixel_count": 31659,
      "percentage": 10.33
    }
  }
}
```

## Integración en Otras Aplicaciones

### Python
```python
import requests
import json
from PIL import Image
import io
import base64

# Obtener información de clases
response = requests.get("http://localhost:5000/classes")
classes = response.json()
print(f"El modelo puede detectar {len(classes['classes'])} clases diferentes")

# Segmentar imagen y obtener estadísticas de clases
response = requests.post(
    "http://localhost:5000/segment?format=json&include_classes=true",
    data={"image_url": "http://example.com/imagen.jpg"}
)
data = response.json()

# Mostrar clases detectadas
print("Clases detectadas en la imagen:")
for class_id, stats in data["classes"].items():
    print(f"- {stats['name']}: {stats['percentage']:.2f}% ({stats['pixel_count']} píxeles)")

# Mostrar la imagen segmentada
image_data = base64.b64decode(data["segmentation_image"])
image = Image.open(io.BytesIO(image_data))
image.show()
```

### JavaScript/Node.js
```javascript
const fetch = require('node-fetch');
const fs = require('fs');

// Obtener segmentación con información de clases
async function segmentAndAnalyze(imageUrl) {
  const response = await fetch(
    `http://localhost:5000/segment_url?url=${encodeURIComponent(imageUrl)}&format=json&include_classes=true`
  );
  
  const data = await response.json();
  
  // Guardar la imagen
  const imageBuffer = Buffer.from(data.segmentation_image, 'base64');
  fs.writeFileSync('resultado.png', imageBuffer);
  
  // Mostrar estadísticas de clases
  console.log('Clases detectadas:');
  Object.entries(data.classes)
    .sort((a, b) => b[1].percentage - a[1].percentage)
    .forEach(([classId, stats]) => {
      console.log(`- ${stats.name}: ${stats.percentage.toFixed(2)}% (${stats.pixel_count} píxeles)`);
    });
}

segmentAndAnalyze('http://example.com/imagen.jpg');
```

## Despliegue en Producción

Para un entorno de producción, considera estas recomendaciones:

1. **Usar un servidor proxy inverso** como Nginx para manejar las conexiones entrantes
2. **Configurar HTTPS** para conexiones seguras
3. **Implementar limitación de velocidad** para prevenir abusos
4. **Monitorizar el rendimiento** usando herramientas como Prometheus/Grafana

Ejemplo de configuración con Docker Compose para producción:

```yaml
version: '3'
services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api

  api:
    build: .
    expose:
      - "5000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Solución de Problemas

### La API no responde
- Verifica que el contenedor está en ejecución con `docker ps`
- Comprueba los logs con `docker logs [ID_CONTENEDOR]`

### Tiempos de respuesta lentos
- Considera usar una GPU para acelerar el procesamiento
- Verifica si hay suficiente memoria disponible

### Errores al procesar imágenes grandes
- La API puede tener problemas con imágenes muy grandes
- Considera redimensionar las imágenes antes de enviarlas
