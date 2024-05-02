import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

def resize_image(imagen, nuevo_ancho, nuevo_alto):
    print("  ------------------------------------------------------------------------  ")
    print(f"  ====================== START RESIZE OF: {imagen} ======================  ")
    print("  ------------------------------------------------------------------------  ")

    imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto))# Se redimensiona la imagen al nuevo tamaño

    print("  ----------------------------------------------------------------------------------------------------  ")
    print(f" ==================  FINISH RESIZE OF: {imagen}  - RESULT: {imagen_redimensionada} ==================  ")
    print("  ----------------------------------------------------------------------------------------------------  ")

    return imagen_redimensionada


"""
Para normalizar los valores de píxeles de las imágenes para que estén en el rango [0, 1],
se divide los valores de píxeles por 255 (el valor máximo de un píxel en una imagen de 8 bits)
"""

def normalize_image(imagen):
    print("  ------------------------------------------------------------------  ")
    print(f" ================== START NORMALIZE OF: {imagen} ==================  ")
    print("  ------------------------------------------------------------------  ")

    imagen_normalizada = imagen / 255.0 # Normalizar los valores de píxeles

    print("  ---------------------------------------------------------------------------------------------------  ")
    print(f" =================  FINISH NORMALIZE OF: {imagen}  - RESULT: {imagen_normalizada}  =================  ")
    print("  ---------------------------------------------------------------------------------------------------  ")

    return imagen_normalizada



"""
Para mejorar la calidad de la imagen, se aplican diversas técnicas de procesamiento de imágenes,
como la eliminación de ruido o el ajuste del contraste.
Se procede a aplicar un filtro de suavizado para reducción el ruido:
"""

def denoise_image(imagen):
    print("  ------------------------------------------------------------------  ")
    print(f"  ================== START DENOISE OF: {imagen} ==================   ")
    print("  ------------------------------------------------------------------  ")

    plt.imshow(imagen)
    plt.show()
    print("image shape: ", imagen.shape)
    print("image type: ", imagen.dtype)

    imagen_suavizada = cv2.medianBlur(imagen, 3)  #Aplicar un filtro de suavizado (filtro de mediana) para reducir el ruido. El segundo argumento es el tamaño del kernel

    print("  ---------------------------------------------------------------------------------------------------  ")
    print(f"  =================   FINISH DENOISE OF: {imagen}   - RESULT: {imagen_suavizada}   =================  ")
    print("  ---------------------------------------------------------------------------------------------------  ")

    return imagen_suavizada


"""
Algunos modelos requieren que las imágenes se ajuste en escala de grises,
Se convierte imágenes en color a escala de grises utilizando cv2.cvtColor() de OpenCV
"""
def convert_to_grayscale(imagen):
    print("  -----------------------------------------------------------------------------  ")
    print(f"  ================== START CONVERT GRAY SCALE OF: {imagen} ==================   ")
    print("  -----------------------------------------------------------------------------  ")

    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)# Convertir la imagen a escala de grises

    print("  --------------------------------------------------------------------------------------------------------------  ")
    print(f"  =================   FINISH CONVERT GRAY SCALE OF: {imagen}   - RESULT: {imagen_gris}   =================  ")
    print("  --------------------------------------------------------------------------------------------------------------  ")

    return imagen_gris


def preprocess_function(imagen):
    print("  -----------------------------------------------------------------------------  ")
    print(f"   ==================  START PRE PROCCESING OF: {imagen}  ==================    ")
    print("  -----------------------------------------------------------------------------  ")

    img = normalize_image(resize_image(convert_to_grayscale(imagen), 128, 128))

    print("  -----------------------------------------------------------------------------  ")
    print(f"     ==================   END PRE PROCCESING OF: {img}    ==================    ")
    print("  -----------------------------------------------------------------------------  ")

    return img

# Ruta de la folder que contiene las imágenes de los beverages
imgs_folder = "./data/"

data = []
labels = []

# Recorremos cada folder dentro de la folder principal
print("  ------------------------------------------------------------------  ")
print("   ===========================  LABELS  ===========================   ")
print("  ------------------------------------------------------------------  ")

for folder in os.listdir(imgs_folder):
    # Obtener el name de la beverage a partir del name de la folder
    beverage_name = folder

    # Se obtiene la ruta completa de la carpeta de la beverage
    beverage_folder_path = os.path.join(imgs_folder, folder)

    # Recorrer cada imagen dentro de la carpeta del beverage
    for img_file in os.listdir(beverage_folder_path):
        # Se lee la imagen
        img = cv2.imread(os.path.join(beverage_folder_path, img_file))
        if img is None:
            continue
        print("  ------------------------------------------------------------------  ")
        print(f"   =================   IMG: {img_file.upper()}    =================  ")
        print("  ------------------------------------------------------------------  ")

        # plt.imshow(img)
        # plt.title(f'Label: {img_file.upper()}')
        # plt.show()

        print("  ------------------------------------------------------------------  ")
        print("")
        print("  ------------------------------------------------------------------  ")
        print( f"  ===========  PREPROCESSING IMG: {img_file.upper()}  ===========  ")
        print("  ------------------------------------------------------------------  ")

        # Preprocesar la imagen (redimensionar, normalizar, etc.)
        img_preprocesada = preprocess_function(img)

        print("  ------------------------------------------------------------------  ")
        print(f"  ===========  END PREPROCESS IMG: {img_file.upper()}  ===========   ")
        print("  ------------------------------------------------------------------  ")

        # Se guarda la imagen ya procesada
        data.append(img_preprocesada)
        print("  ------------------------------------------------------------------  ")
        print(f"  ==========  ADD PREPROCESSED IMG: {img_file.upper()}  ==========   ")
        print("  ------------------------------------------------------------------  ")

        #Se guarda la etiqueta
        label = beverage_name
        labels.append(label)
        print("  ------------------------------------------------------------------  ")
        print(f" ========  ADD LABEL: {label} BY IMG: {img_file.upper()}  ========   ")
        print("  ------------------------------------------------------------------  ")


print("  ------------------------------------------------------------------  ")
print("  ========================== END LABELS  ==========================   ")
print("  ------------------------------------------------------------------  ")

# Convertir las listas a matrices numpy
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=str)

# Imprimir los arreglos
print("Array data: ")
print(data)
print("")
print("Array labels: ")
print(labels)

#  ------------------------------
#  ------------------------------
#  ------------------------------
#  ------------------------------

# este numero corresponde a la cantidad de clases (es decir, la cant de beverages) cargadas en data (osea cantidad de carpetas)
# por ahora va hardcodeado
num_classes = 3

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create a dictionary mapping each unique label to an integer
label_map = {label: idx for idx, label in enumerate(np.unique(labels))}

# Convert string labels to numeric representations using the mapping
numeric_labels = np.array([label_map[label] for label in labels])

# Now, convert numeric labels to categorical
train_labels = tf.keras.utils.to_categorical(numeric_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(numeric_labels, num_classes)

# Compilo el model
model = build_model()
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(data, train_labels, epochs=10, batch_size=64, validation_data=(data, test_labels))

# Imprimo grafico de perdida etanto en entrenamiento como validacion
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('# Epoch')
plt.ylabel('# Loss')
plt.legend()
plt.show()

# Imprimo grafico de precision de la red neuronal tanto en entrenamiento como en validacion
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluar el modelo
test_loss, test_acc = model.evaluate(data, test_labels)
print('Test accuracy:', test_acc)

# Realización de la predicción para la muestra seleccionada
predictions = model.predict(data)
index = np.random.randint(0, len(data))

# Realización de la predicción para la muestra seleccionada
prediction = np.argmax(predictions[index])

# Visualización de la imagen y su etiqueta predicha
plt.imshow(data[index].reshape(128, 128))
plt.title(f'Predicted Label: {prediction}, True Label: {np.argmax(test_labels[index])}')
plt.show()