import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from utils.preprocessing import crear_lote_imagenes, aumentar_conjunto_datos
import os
import sys
import argparse
from utils.detector_radiografias import DetectorRadiografias
from models.clasificador_enfermedades import ClasificadorEnfermedades
from utils.generador_datos import GeneradorDatosRadiografias

# Mapeo de categorías a números
CATEGORIAS = {
    'normal': 0,
    'neumonia': 1,
    'tuberculosis': 2,
    'cancer_pulmon': 3,
    'derrame_pleural': 4,
    'epoc': 5,
    'fibrosis_pulmonar': 6
}

def crear_modelo():
    """
    Crea y compila el modelo de clasificación de radiografías.
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
        Dense(7, activation='softmax')  # 7 clases para las diferentes enfermedades
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

def main():
    # Configurar argumentos
    parser = argparse.ArgumentParser(description='Entrenamiento del modelo de clasificación de radiografías')
    parser.add_argument('--datos', type=str, required=True, help='Ruta al directorio de datos')
    parser.add_argument('--epocas', type=int, default=100, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamaño del batch')
    parser.add_argument('--modelo_salida', type=str, default='modelo_radiografias.h5', help='Ruta para guardar el modelo')
    args = parser.parse_args()
    
    # Verificar directorio de datos
    if not os.path.exists(args.datos):
        raise ValueError(f"El directorio de datos {args.datos} no existe")
    
    try:
        # Inicializar componentes
        print("Inicializando componentes...")
        detector = DetectorRadiografias()
        
        # Inicializar generador de datos
        print("Configurando generador de datos...")
        generador_datos = GeneradorDatosRadiografias(
            directorio_datos=args.datos,
            batch_size=args.batch_size
        )
        
        # Obtener número de clases primero
        print("Obteniendo clases...")
        clases = generador_datos.obtener_clases()
        num_clases = len(clases)
        print(f"Clases detectadas: {clases}")
        print(f"Número de clases: {num_clases}")
        
        # Obtener generadores de datos
        print("Configurando generadores de entrenamiento y validación...")
        generador_entrenamiento, generador_validacion = generador_datos.obtener_generadores()
        
        # Inicializar modelo
        print("Inicializando modelo...")
        clasificador = ClasificadorEnfermedades(num_clases=num_clases)
        
        # Configurar callbacks
        print("Configurando callbacks...")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'mejor_modelo.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        print("Iniciando entrenamiento...")
        historial = clasificador.entrenar(
            generador_entrenamiento=generador_entrenamiento,
            generador_validacion=generador_validacion,
            epocas=args.epocas,
            callbacks=callbacks
        )
        
        # Guardar modelo
        print(f"Guardando modelo en {args.modelo_salida}...")
        clasificador.guardar_modelo(args.modelo_salida)
        
        print("¡Entrenamiento completado!")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}", file=sys.stderr)
        raise

if __name__ == '__main__':
    main() 