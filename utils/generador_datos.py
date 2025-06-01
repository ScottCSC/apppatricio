import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

class GeneradorDatosRadiografias:
    def __init__(self, directorio_datos, tamano_imagen=(224, 224), batch_size=16):
        """
        Inicializa el generador de datos.
        
        Args:
            directorio_datos (str): Ruta al directorio de datos
            tamano_imagen (tuple): Tamaño de las imágenes (alto, ancho)
            batch_size (int): Tamaño del batch
        """
        self.directorio_datos = directorio_datos
        self.tamano_imagen = tamano_imagen
        self.batch_size = batch_size
        
        # Configurar generador de entrenamiento con aumentación
        self.generador_entrenamiento = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Configurar generador de validación sin aumentación
        self.generador_validacion = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
    
    def obtener_generadores(self):
        """
        Obtiene los generadores de datos para entrenamiento y validación.
        
        Returns:
            tuple: (generador_entrenamiento, generador_validacion)
        """
        # Obtener generador de entrenamiento
        generador_entrenamiento = self.generador_entrenamiento.flow_from_directory(
            self.directorio_datos,
            target_size=self.tamano_imagen,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        # Obtener generador de validación
        generador_validacion = self.generador_validacion.flow_from_directory(
            self.directorio_datos,
            target_size=self.tamano_imagen,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        return generador_entrenamiento, generador_validacion
    
    def obtener_clases(self):
        """
        Obtiene las clases disponibles en el directorio de datos.
        
        Returns:
            dict: Diccionario con las clases y sus índices
        """
        # Crear un generador temporal para obtener las clases
        generador_temp = self.generador_entrenamiento.flow_from_directory(
            self.directorio_datos,
            target_size=self.tamano_imagen,
            batch_size=1,
            class_mode='categorical',
            subset='training',
            shuffle=False
        )
        
        return generador_temp.class_indices
    
    def preprocesar_imagen(self, imagen):
        """
        Preprocesa una imagen para predicción.
        
        Args:
            imagen (numpy.ndarray): Imagen a preprocesar
            
        Returns:
            numpy.ndarray: Imagen preprocesada
        """
        # Redimensionar
        imagen = cv2.resize(imagen, self.tamano_imagen)
        
        # Normalizar
        imagen = imagen.astype(np.float32) / 255.0
        
        return imagen 