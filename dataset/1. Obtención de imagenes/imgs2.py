import os
import requests
from bs4 import BeautifulSoup
import json

key = "AIzaSyASzzXTLpSpK-Lu7LaS5Ezk6JoVYmmA-QM"

def get_google_images(query, count):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": "80a78c7bf7b9b434d",
        "key": key,
        "searchType": "image",
        "num": count
    }
    response = requests.get(url, params=params)
    print("RESPONSE:", response.json())
    data = response.json()
    print(response.text)
    print(data)
    image_urls = [item["link"] for item in data.get("items", [])]
    print(image_urls)
    return image_urls

print("COMENZANDO DESCARGA DE IMÁGENES: ")

beverages = ["fernet", "gin", "ron", "wine", "whiskey"]
count_per_beverage = 30  # Número de imágenes a descargar por cada tipo de bebida

for beverage in beverages:
    print(f"Comenzar descarga para la bebida: {beverage.upper()}")
    directory = beverage
    os.makedirs(directory, exist_ok=True)
    image_urls = get_google_images(beverage, count_per_beverage)
    print(f" ======== Descargando {len(image_urls)} imágenes para la bebida: {beverage.upper()} ======== ")
    for i, img_url in enumerate(image_urls):
        try:
            img_data = requests.get(img_url).content
            filename = os.path.join(directory, f"{beverage}{i}.jpg")
            with open(filename, "wb") as f:
                f.write(img_data)
            print(f"- Imagen {i} descargada: ", filename)
        except Exception as e:
            print("Error al descargar la imagen:", e)
    print(f" ======== Terminando descarga para: {beverage.upper()} ======== ")

print("DESCARGA FINALIZADA")