import http.client
import json
import os
import requests

beverages = ["fernet", "vodka", "ron", "whisky", "gin", "vino", "cerveza"]
print("COMENZANDO DESCARGA DE IMAGENES")
for beverage in beverages:
  count = 0
  no_more_images = False
  for n in list(range(1,11)):
    if no_more_images:
       break
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": beverage,
        "gl": "ar",
        "num": 100,
        "page": n
    })
    headers = {
        'X-API-KEY': '74a2bfeb03430ba0d7b5fed1a79d8dce1c640362',
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/images", payload, headers)
    res = conn.getresponse()
    data = res.read()
    data_decoded = data.decode("utf-8")
    print("----------------------------------------------------------------------------------------------------------------------")
    print("* Data sin decodear: ", data)
    print("----------------------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------------------")
    print("* Data decodeada: ", data_decoded)
    print("----------------------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------------------")
    data_decoded = json.loads(data_decoded)
    print("* Data decodeada en formato JSON", data_decoded)
    print("----------------------------------------------------------------------------------------------------------------------")
    directory = beverage
    os.makedirs(directory, exist_ok=True)
    images = data_decoded["images"]
    if len(images) == 0:
       no_more_images = True
    print("----------------------------------------------------------------------------------------------------------------------")
    print(f" ======== Empezando descarga de imagenes para: {beverage.upper()} ======== ")
    print("----------------------------------------------------------------------------------------------------------------------")
    for image in images:
        try:
            img_url = image["imageUrl"]
            img_data = requests.get(img_url).content
            filename = os.path.join(directory, f"{beverage}{count}.jpg")
            with open(filename, "wb") as f:
                f.write(img_data)
                print("----------------------------------------------------------------------------------------------------------------------")
                print(f"- Imagen {count} descargada:", filename)
                print("----------------------------------------------------------------------------------------------------------------------")
                count += 1
        except Exception as e:
            print("Error al descargar la imagen:", e)
    print("----------------------------------------------------------------------------------------------------------------------")
    print(f" ======== Terminando descarga de imagenes para: {beverage.upper()} ======== ")
    print("----------------------------------------------------------------------------------------------------------------------")
print("DESCARGA DE IMAGENES FINALIZADAS")