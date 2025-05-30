import cv2
import numpy as np
from PIL import Image
import os

def preprocesar_imagen(ruta_imagen, tamaño=(224, 224)):
    """
    Preprocesa una imagen para el modelo de clasificación.
    
    Args:
        ruta_imagen (str): Ruta a la imagen
        tamaño (tuple): Tamaño deseado de la imagen (ancho, alto)
    
    Returns:
        numpy.ndarray: Imagen preprocesada
    """
    try:
        # Cargar imagen
        imagen = Image.open(ruta_imagen).convert('RGB')
        
        # Redimensionar
        imagen = imagen.resize(tamaño)
        
        # Convertir a array y normalizar
        imagen_np = np.array(imagen) / 255.0
        
        return imagen_np
    except Exception as e:
        print(f"Error al preprocesar imagen {ruta_imagen}: {str(e)}")
        return None

def aumentar_conjunto_datos(imagen, factor=1.5):
    """
    Aplica transformaciones para aumentar el conjunto de datos.
    
    Args:
        imagen (numpy.ndarray): Imagen a transformar
        factor (float): Factor de zoom
    
    Returns:
        list: Lista de imágenes transformadas
    """
    imagenes_aumentadas = []
    
    # Rotación
    for angulo in [90, 180, 270]:
        matriz_rotacion = cv2.getRotationMatrix2D((imagen.shape[1]/2, imagen.shape[0]/2), angulo, 1)
        imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (imagen.shape[1], imagen.shape[0]))
        imagenes_aumentadas.append(imagen_rotada)
    
    # Volteo horizontal
    imagen_volteada = cv2.flip(imagen, 1)
    imagenes_aumentadas.append(imagen_volteada)
    
    return imagenes_aumentadas

def crear_lote_imagenes(ruta_directorio, tamaño=(224, 224)):
    """
    Crea un lote de imágenes preprocesadas desde un directorio.
    
    Args:
        ruta_directorio (str): Ruta al directorio con imágenes
        tamaño (tuple): Tamaño deseado de las imágenes
    
    Returns:
        tuple: (imagenes, etiquetas)
    """
    imagenes = []
    etiquetas = []
    
    # Verificar que existan las carpetas de categorías
    categorias_esperadas = ['normal', 'inflamacion', 'lesion']
    for categoria in categorias_esperadas:
        ruta_categoria = os.path.join(ruta_directorio, categoria)
        if not os.path.exists(ruta_categoria):
            print(f"Advertencia: No se encontró la carpeta para la categoría '{categoria}'")
            continue
            
        # Procesar imágenes de esta categoría
        for archivo in os.listdir(ruta_categoria):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_imagen = os.path.join(ruta_categoria, archivo)
                imagen = preprocesar_imagen(ruta_imagen, tamaño)
                if imagen is not None:
                    imagenes.append(imagen)
                    etiquetas.append(categoria)
    
    if len(imagenes) == 0:
        print("Error: No se encontraron imágenes válidas en ninguna categoría")
        return np.array([]), np.array([])
    
    return np.array(imagenes), np.array(etiquetas) 