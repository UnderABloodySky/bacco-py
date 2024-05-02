import os
import requests
from bs4 import BeautifulSoup


print("COMENZANDO DESCARGA DE IMAGENES: ")
# Iterar sobre las etiquetas de imagen y descargar las imágenes

beverages= ["fernet", "gin", "ron","wine", "whiskey"]

print("Comenzar descarga: ")
for beverage in beverages:
  # Contador para llevar la cuenta del número de imágenes descargadas
  count = 0
  print(f"Comenzar descarga para el beverage: {beverage.upper()}")
  # URL de búsqueda de imágenes de cada beverage
  url = f"https://www.google.com/search?q={beverage}&tbm=isch"

  # Establecer el User-Agent para evitar problemas con la solicitud
  headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

  # Realizar la solicitud HTTP
  response = requests.get(url, headers=headers)
  soup = BeautifulSoup(response.text, "html.parser")

  # Encontrar todas las etiquetas de imágenes
  image_tags = soup.find_all("img")
  print("CANTIDAD DE TAGS: ", len(image_tags))
  # Directorio donde se guardarán las imágenes
  directory = f"{beverage}"
  os.makedirs(directory, exist_ok=True)
  print(f" ======== Generando imagenes para: {beverage.upper()} ======== ")
  for img_tag in image_tags:
    try:
        img_url = img_tag["src"]
        img_data = requests.get(img_url).content
        filename = os.path.join(directory, f"{beverage}{count}.jpg")
        with open(filename, "wb") as f:
            f.write(img_data)
        print(f"Imagen {count} descargada:", filename)
        count += 1

        # Salir del bucle si se han descargado 2000 imágenes
        if count > 2000:
            break
    except Exception as e:
        print("Error al descargar la imagen:", e)
    print(f" ======== Terminando descarga de imagenes para: {beverage.upper()} ======== ")

print("DESCARGA FINALIZADA")
