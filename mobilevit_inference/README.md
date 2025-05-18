# Aplicación de Segmentación Semántica

## Descripción
Esta aplicación utiliza un modelo de inteligencia artificial (MobileViTV2) para realizar segmentación semántica en imágenes. La segmentación semántica consiste en clasificar cada píxel de una imagen en diferentes categorías, lo que permite identificar y separar objetos en la imagen.

## Requisitos
- Docker instalado en tu sistema
- Conexión a internet (para descargar la imagen base de Docker y el modelo)
- (Opcional) GPU compatible con CUDA para un procesamiento más rápido

## Instalación

### 1. Clonar o crear los archivos necesarios
Crea una carpeta para el proyecto y guarda los siguientes archivos:

1. `segmentation.py`: El script principal de Python
2. `Dockerfile`: La configuración para crear la imagen de Docker

### 2. Construir la imagen de Docker
Abre una terminal en la carpeta del proyecto y ejecuta el siguiente comando:

```bash
docker build -t segmentacion-app .
```

Este proceso puede tardar varios minutos la primera vez, ya que necesita descargar la imagen base de Docker y todas las dependencias.

## Uso

### Ejecutar con la imagen predeterminada
Para procesar la imagen de demostración (un gato):

```bash
docker run --rm -v $(pwd)/output:/app/output segmentacion-app
```

### Ejecutar con una imagen desde URL
Para procesar una imagen desde una URL:

```bash
docker run --rm -v $(pwd)/output:/app/output segmentacion-app python segmentation.py "https://ejemplo.com/tu-imagen.jpg"
```

### Ejecutar con una imagen local
Primero, copia la imagen a un directorio que montarás en Docker:

```bash
mkdir -p input
cp tu-imagen.jpg input/
```

Luego ejecuta:

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output segmentacion-app python segmentation.py "/app/input/tu-imagen.jpg"
```

### Usar GPU (si está disponible)
Si tienes una GPU compatible con CUDA, puedes usarla para acelerar el procesamiento:

```bash
docker run --rm --gpus all -v $(pwd)/output:/app/output segmentacion-app
```

## Salidas

Después de ejecutar el contenedor, se crearán varios archivos en la carpeta `output`:

1. `input_image.jpg`: La imagen original procesada
2. `segmentation_logits.pt`: Un archivo de tensor PyTorch con los resultados brutos de la segmentación
3. `segmentation_visualization.png`: Una visualización que contiene:
   - La imagen original
   - El mapa de segmentación (diferentes colores para diferentes clases)
   - Una superposición del mapa de segmentación sobre la imagen original

## Explicación de la visualización

La visualización final contiene tres paneles:

1. **Imagen Original**: La imagen de entrada sin modificar.
2. **Mapa de Segmentación**: Cada color representa una clase diferente que el modelo ha identificado.
3. **Superposición**: El mapa de segmentación combinado con la imagen original para ver cómo se alinean las clases con los objetos reales.

## Solución de problemas

### Error: No se puede conectar al daemon de Docker
Asegúrate de que Docker esté instalado y en ejecución con:
```bash
docker --version
docker info
```

### Error: No hay suficiente espacio
Limpia imágenes no utilizadas:
```bash
docker system prune -a
```

### Error: La imagen es demasiado grande para procesar
Edita `segmentation.py` para redimensionar la imagen antes de procesarla.

### Error al usar GPU
Asegúrate de tener instalados los controladores NVIDIA y nvidia-docker:
```bash
nvidia-smi
docker info | grep Runtimes
```

## Personalización

### Cambiar el modelo de segmentación
Puedes modificar el modelo usado editando la línea en `segmentation.py`:
```python
model = MobileViTV2ForSemanticSegmentation.from_pretrained("tu-modelo-preferido")
```

### Ajustar la visualización
Puedes modificar los colores del mapa cambiando `'viridis'` por otro mapa de colores como `'jet'`, `'rainbow'` o `'tab20'`.

## Notas adicionales

- La primera ejecución puede ser lenta ya que se descargan los pesos del modelo.
- Los resultados de la segmentación dependen de las clases en las que el modelo fue entrenado.


# API de Segmentación Semántica

## Descripción
Esta aplicación ofrece una API REST para realizar segmentación semántica en imágenes utilizando el modelo MobileViTV2. La segmentación semántica clasifica cada píxel de una imagen en diferentes categorías, permitiendo identificar y separar objetos en la imagen.

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

### Segmentación de Imagen por URL (GET)
```
GET /segment_url?url={URL_DE_LA_IMAGEN}
```
Endpoint simplificado para segmentar una imagen directamente desde una URL.

## Ejemplos de Uso

### Usando curl

#### 1. Verificar que la API está funcionando
```bash
curl http://localhost:5000/health
```

#### 2. Segmentar una imagen desde URL (método GET)
```bash
curl -o resultado.png "http://localhost:5000/segment_url?url=http://images.cocodataset.org/val2017/000000039769.jpg"
```

#### 3. Segmentar una imagen desde URL (método POST)
```bash
curl -X POST -F "image_url=http://images.cocodataset.org/val2017/000000039769.jpg" -o resultado.png http://localhost:5000/segment
```

#### 4. Segmentar una imagen subida
```bash
curl -X POST -F "image_file=@/ruta/a/tu/imagen.jpg" -o resultado.png http://localhost:5000/segment
```

#### 5. Obtener resultados en formato JSON
```bash
curl -X POST -F "image_url=http://images.cocodataset.org/val2017/000000039769.jpg" "http://localhost:5000/segment?format=json"
```

### Usando el cliente de prueba

El proyecto incluye un script cliente para probar la API:

```bash
# Probar con una imagen desde URL
python test_client.py --api http://localhost:5000 --url http://images.cocodataset.org/val2017/000000039769.jpg

# Probar con una imagen local
python test_client.py --api http://localhost:5000 --file tu_imagen.jpg
```

## Integración en Otras Aplicaciones

### Python
```python
import requests
from PIL import Image
import io

# Enviar una imagen desde URL
response = requests.post(
    "http://localhost:5000/segment",
    data={"image_url": "http://example.com/imagen.jpg"}
)

# Guardar o mostrar la imagen resultante
with open("resultado.png", "wb") as f:
    f.write(response.content)

# Alternativa: obtener resultado como JSON
response = requests.post(
    "http://localhost:5000/segment?format=json",
    data={"image_url": "http://example.com/imagen.jpg"}
)
data = response.json()
```

### JavaScript/Node.js
```javascript
const fetch = require('node-fetch');
const fs = require('fs');
const FormData = require('form-data');

// Ejemplo con URL de imagen
fetch('http://localhost:5000/segment_url?url=http://example.com/imagen.jpg')
  .then(response => response.buffer())
  .then(buffer => {
    fs.writeFileSync('resultado.png', buffer);
    console.log('Imagen guardada como resultado.png');
  });

// Ejemplo con archivo
const form = new FormData();
form.append('image_file', fs.createReadStream('imagen.jpg'));

fetch('http://localhost:5000/segment', {
  method: 'POST',
  body: form
})
  .then(response => response.buffer())
  .then(buffer => {
    fs.writeFileSync('resultado.png', buffer);
    console.log('Imagen guardada como resultado.png');
  });
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
