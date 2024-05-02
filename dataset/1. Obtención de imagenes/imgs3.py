from google_images_download import google_images_download
from selenium import webdriver

keyword = "cat"  # Cambia "cat" por la categoría de imágenes que deseas descargar
limit = 2000  # Número de imágenes a descargar

# Configura las opciones del navegador
options = webdriver.ChromeOptions()
options.binary_location = '/usr/bin/chromium-browser'   # Especifica la ubicación de Chromium
options.add_argument('--headless')  # Ejecución en modo headless (sin ventana)
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Inicializa el driver de Chrome con las opciones configuradas
driver = webdriver.Chrome(options=options)


def download_images(keyword, limit):

    response = google_images_download.googleimagesdownload()

    # Configuración de los parámetros de descarga
    arguments = {
        "keywords": keyword,
        "limit": limit,
        "print_urls": False,  # No mostrar las URL de las imágenes
        "chromedriver": '/usr/bin/chromedriver'  # Especifica la ubicación del chromedriver
    }

    # Descarga de imágenes
    paths = response.download(arguments)
    print(paths)

    # Cierra el driver después de la descarga
    driver.quit()

download_images(keyword, limit)
