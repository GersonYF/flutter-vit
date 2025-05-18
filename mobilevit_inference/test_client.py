import requests
import argparse
from PIL import Image
import io
import base64

def test_api_url(api_url, image_url):
    """
    Prueba el endpoint de segmentación con una URL de imagen.
    
    Args:
        api_url: URL del endpoint de la API
        image_url: URL de la imagen a segmentar
    """
    print(f"Enviando solicitud a {api_url} con imagen: {image_url}")
    
    # Método 1: Usar el endpoint simplificado GET
    print("\nMétodo 1: Usando endpoint GET con parámetro URL")
    response = requests.get(f"{api_url}/segment_url?url={image_url}")
    
    if response.status_code == 200:
        # Guardar la imagen recibida
        with open("resultado_metodo1.png", "wb") as f:
            f.write(response.content)
        print("Imagen guardada como 'resultado_metodo1.png'")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    # Método 2: Usar el endpoint POST con form data
    print("\nMétodo 2: Usando endpoint POST con form data")
    response = requests.post(
        f"{api_url}/segment",
        data={"image_url": image_url}
    )
    
    if response.status_code == 200:
        # Guardar la imagen recibida
        with open("resultado_metodo2.png", "wb") as f:
            f.write(response.content)
        print("Imagen guardada como 'resultado_metodo2.png'")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    # Método 3: Usar el endpoint POST con formato JSON
    print("\nMétodo 3: Usando endpoint POST con respuesta en formato JSON")
    response = requests.post(
        f"{api_url}/segment?format=json",
        data={"image_url": image_url}
    )
    
    if response.status_code == 200:
        # Procesar la respuesta JSON
        data = response.json()
        # Decodificar la imagen en base64
        image_data = base64.b64decode(data["segmentation_image"])
        # Guardar la imagen
        with open("resultado_metodo3.png", "wb") as f:
            f.write(image_data)
        print("Imagen guardada como 'resultado_metodo3.png'")
        print(f"Información adicional: {data['width']}x{data['height']}, {data['classes']} classes")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_api_file(api_url, image_path):
    """
    Prueba el endpoint de segmentación con un archivo de imagen local.
    
    Args:
        api_url: URL del endpoint de la API
        image_path: Ruta al archivo de imagen local
    """
    print(f"Enviando archivo desde {image_path} a {api_url}")
    
    with open(image_path, "rb") as img_file:
        files = {"image_file": (image_path, img_file, "image/jpeg")}
        response = requests.post(f"{api_url}/segment", files=files)
    
    if response.status_code == 200:
        # Guardar la imagen recibida
        with open("resultado_archivo.png", "wb") as f:
            f.write(response.content)
        print("Imagen guardada como 'resultado_archivo.png'")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_api_base64(api_url, image_path):
    """
    Prueba el endpoint de segmentación con una imagen codificada en base64.
    
    Args:
        api_url: URL del endpoint de la API
        image_path: Ruta al archivo de imagen local que se codificará en base64
    """
    print(f"Codificando imagen {image_path} en base64 y enviando a {api_url}")
    
    # Leer y codificar la imagen en base64
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
    
    # Enviar solicitud con datos base64
    response = requests.post(
        f"{api_url}/segment",
        data={"image_base64": image_base64}
    )
    
    if response.status_code == 200:
        # Guardar la imagen recibida
        with open("resultado_base64.png", "wb") as f:
            f.write(response.content)
        print("Imagen guardada como 'resultado_base64.png'")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def check_health(api_url):
    """
    Verifica si la API está funcionando utilizando el endpoint de health check.
    
    Args:
        api_url: URL base de la API
    """
    print(f"Verificando estado de la API en {api_url}/health")
    
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            print("La API está funcionando correctamente:")
            print(response.json())
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con la API: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente para probar la API de segmentación semántica")
    parser.add_argument("--api", default="http://localhost:5000", help="URL base de la API")
    parser.add_argument("--url", help="URL de una imagen para segmentar")
    parser.add_argument("--file", help="Ruta a un archivo de imagen local para segmentar")
    
    args = parser.parse_args()
    
    # Verificar si la API está funcionando
    if check_health(args.api):
        # Ejecutar pruebas según los argumentos proporcionados
        if args.url:
            test_api_url(args.api, args.url)
        
        if args.file:
            test_api_file(args.api, args.file)
            test_api_base64(args.api, args.file)
    
        if not args.url and not args.file:
            print("\nNo se proporcionó ninguna imagen para probar.")
            print("Uso: python test_client.py --url https://ejemplo.com/imagen.jpg")
            print("  o: python test_client.py --file ruta/a/imagen.jpg")
    else:
        print("No se pudo conectar con la API. Asegúrate de que esté en ejecución.")
