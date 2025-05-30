import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from utils.preprocessing import crear_lote_imagenes, aumentar_conjunto_datos
import os
import sys

# Mapeo de categorías a números
CATEGORIAS = {
    'normal': 0,
    'inflamacion': 1,
    'lesion': 2
}

def crear_modelo():
    """
    Crea y compila el modelo de clasificación de manos.
    """
    print("Creando modelo...")
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
    
    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Modelo creado exitosamente")
    return modelo

def entrenar_modelo(ruta_datos, epocas=50, batch_size=32):
    """
    Entrena el modelo con los datos proporcionados.
    
    Args:
        ruta_datos (str): Ruta al directorio con las imágenes
        epocas (int): Número de épocas de entrenamiento
        batch_size (int): Tamaño del lote
    """
    try:
        print("Cargando y preprocesando datos...")
        # Cargar y preprocesar datos
        X, y = crear_lote_imagenes(ruta_datos)
        
        if len(X) == 0:
            print("Error: No se encontraron imágenes para entrenar")
            return None, None
        
        print(f"Imágenes cargadas: {len(X)}")
        
        # Convertir etiquetas de texto a números
        y_numerico = np.array([CATEGORIAS[etiqueta] for etiqueta in y])
        print(f"Distribución de clases: {np.unique(y_numerico, return_counts=True)}")
        
        # Convertir etiquetas a one-hot encoding
        y_one_hot = tf.keras.utils.to_categorical(y_numerico, num_classes=len(CATEGORIAS))
        
        # Dividir en conjuntos de entrenamiento y validación
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
        
        print(f"Conjunto de entrenamiento: {len(X_train)} imágenes")
        print(f"Conjunto de validación: {len(X_val)} imágenes")
        
        # Crear modelo
        modelo = crear_modelo()
        
        # Asegurarse de que existe el directorio de modelos
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            'models/mejor_modelo.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        print("\nIniciando entrenamiento...")
        # Entrenar modelo
        historia = modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epocas,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping]
        )
        
        print("\nEntrenamiento completado")
        return modelo, historia
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}", file=sys.stderr)
        return None, None

if __name__ == "__main__":
    # Verificar que existen los directorios necesarios
    if not os.path.exists('data/processed'):
        print("Error: No se encuentra el directorio 'data/processed'", file=sys.stderr)
        sys.exit(1)
    
    # Entrenar modelo
    modelo, historia = entrenar_modelo('data/processed')
    
    if modelo is not None:
        # Guardar modelo final
        modelo.save('models/modelo_final.h5')
        print("\nModelo guardado exitosamente en 'models/modelo_final.h5'")
    else:
        print("\nNo se pudo entrenar el modelo", file=sys.stderr)
        sys.exit(1) 