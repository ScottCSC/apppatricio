import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from PIL import Image
import os

# Paso 1: Cargar imagen de mano
def cargar_imagen(ruta_imagen):
    imagen = Image.open(ruta_imagen).convert('RGB')  # Color para mejor detección
    imagen = imagen.resize((224, 224))  # Tamaño estándar para modelos de clasificación
    imagen_np = np.array(imagen) / 255.0
    imagen_np = imagen_np.reshape(1, 224, 224, 3)
    return imagen_np

# Paso 2: Crear un modelo CNN mejorado para clasificación de manos
def crear_modelo():
    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')  # 3 clases: normal, inflamación, lesión
    ])
    modelo.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return modelo

# Sistema de recomendación contextual
def generar_recomendacion(prediccion, confianza):
    recomendaciones = {
        'normal': [
            "Mantener la rutina de ejercicios de mano",
            "Continuar con las medidas preventivas básicas"
        ],
        'inflamacion': [
            "Aplicar hielo por 15-20 minutos cada 2-3 horas",
            "Elevar la mano para reducir la inflamación",
            "Considerar antiinflamatorios si es apropiado",
            "Programar cita con especialista si persiste"
        ],
        'lesion': [
            "Buscar atención médica inmediata",
            "Inmovilizar la mano si es necesario",
            "Evitar movimientos que causen dolor",
            "Documentar síntomas y evolución"
        ]
    }
    
    categoria = np.argmax(prediccion)
    categorias = ['normal', 'inflamacion', 'lesion']
    categoria_actual = categorias[categoria]
    
    print("\n📋 Recomendaciones basadas en el diagnóstico:")
    for recomendacion in recomendaciones[categoria_actual]:
        print(f"• {recomendacion}")
    
    if confianza < 0.7:
        print("\n⚠️ Nota: La confianza del diagnóstico es baja. Se recomienda consultar con un especialista.")

# Paso 3: Clasificar imagen y generar recomendaciones
def predecir(modelo, ruta_imagen):
    if not os.path.exists(ruta_imagen):
        print(f"❌ No se encontró la imagen: {ruta_imagen}")
        return
    
    imagen = cargar_imagen(ruta_imagen)
    prediccion = modelo.predict(imagen)
    confianza = np.max(prediccion[0])
    categoria = np.argmax(prediccion[0])
    
    categorias = ['NORMAL', 'INFLAMACIÓN', 'LESIÓN']
    etiqueta = categorias[categoria]
    
    print(f"\n🔍 Resultado del análisis:")
    print(f"Diagnóstico: {etiqueta}")
    print(f"Nivel de confianza: {confianza:.2%}")
    
    generar_recomendacion(prediccion[0], confianza)
    
    # Mostrar la imagen
    plt.figure(figsize=(10, 8))
    plt.imshow(imagen[0])
    plt.title(f"Diagnóstico: {etiqueta}\nConfianza: {confianza:.2%}")
    plt.axis('off')
    plt.show()

# Ejecutar
if __name__ == "__main__":
    modelo = crear_modelo()
    # Nota: Aquí deberías cargar tu modelo entrenado con datos reales de manos
    # modelo.load_weights('modelo_manos.h5')
    
    ruta_imagen = "mano.jpg"  # Cambia por tu imagen de mano
    predecir(modelo, ruta_imagen)
