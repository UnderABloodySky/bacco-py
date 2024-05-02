import cv2
import pytesseract

print("Versión de Tesseract:", pytesseract.get_tesseract_version())
# Obtener la ruta al ejecutable de Tesseract que Pytesseract está utilizando
print("Ruta al ejecutable de Tesseract:", pytesseract.pytesseract.tesseract_cmd)

# Carga la imagen
img = cv2.imread("example.jpg")

# Convierte la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplica un filtro de mediana para reducir el ruido
gray = cv2.medianBlur(gray, 3)

# Aplica una umbralización adaptativa
threshold_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Invierte los colores de la imagen (negativo)
inverted_img = cv2.bitwise_not(threshold_img)

# Pasa la imagen a través de Pytesseract
text = pytesseract.image_to_string(inverted_img)

# Imprime el texto extraído
print("Texto extraído:", text)

