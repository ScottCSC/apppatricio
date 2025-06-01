import cv2
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

class DetectorRadiografias:
    def __init__(self):
        # Cargar modelo base de DenseNet121 pre-entrenado
        self.modelo_base = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Agregar capas para detección de radiografías
        x = self.modelo_base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        self.modelo = Model(inputs=self.modelo_base.input, outputs=x)
    
    def preprocesar_imagen(self, imagen):
        """
        Preprocesa la imagen para el modelo.
        
        Args:
            imagen (numpy.ndarray): Imagen a procesar
            
        Returns:
            numpy.ndarray: Imagen preprocesada
        """
        try:
            # Verificar si la imagen es válida
            if imagen is None or imagen.size == 0:
                print("Error: Imagen inválida o vacía")
                return None
                
            print(f"Dimensiones originales: {imagen.shape}")
            
            # Convertir a escala de grises si es necesario
            if len(imagen.shape) == 3:
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
            # Normalizar
            imagen = imagen.astype(np.float32) / 255.0
            
            # Redimensionar manteniendo la relación de aspecto
            h, w = imagen.shape
            target_size = (224, 224)
            aspect_ratio = w / h
            
            if aspect_ratio > 1:
                new_w = target_size[0]
                new_h = int(new_w / aspect_ratio)
            else:
                new_h = target_size[1]
                new_w = int(new_h * aspect_ratio)
                
            imagen = cv2.resize(imagen, (new_w, new_h))
            
            # Crear una imagen negra del tamaño objetivo
            final = np.zeros(target_size, dtype=np.float32)
            
            # Calcular posición para centrar la imagen
            y_offset = (target_size[1] - new_h) // 2
            x_offset = (target_size[0] - new_w) // 2
            
            # Colocar la imagen redimensionada en el centro
            final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = imagen
            
            # Convertir a RGB (requerido por DenseNet)
            final = np.stack([final] * 3, axis=-1)
            
            print(f"Dimensiones finales: {final.shape}")
            return final
            
        except Exception as e:
            print(f"Error en preprocesamiento: {str(e)}")
            return None
    
    def detectar_radiografia(self, imagen):
        """
        Detecta si una imagen es una radiografía de tórax válida.
        
        Args:
            imagen (numpy.ndarray): Imagen a analizar
            
        Returns:
            tuple: (bool, numpy.ndarray) - (es_radiografia, imagen_procesada)
        """
        try:
            # Preprocesar imagen
            imagen_procesada = self.preprocesar_imagen(imagen)
            if imagen_procesada is None:
                print("Error: No se pudo preprocesar la imagen")
                return False, None
            
            # Convertir a escala de grises
            gris = cv2.cvtColor(imagen_procesada, cv2.COLOR_RGB2GRAY)
            
            # Convertir a uint8 para Canny
            gris_uint8 = (gris * 255).astype(np.uint8)
            
            # 1. Verificar características específicas de radiografías de tórax
            # Calcular histograma
            hist = cv2.calcHist([gris_uint8], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Características del histograma
            media = np.mean(gris)
            std = np.std(gris)
            entropia = -np.sum(hist * np.log2(hist + 1e-10))
            
            # 2. Detectar características anatómicas del tórax
            # Detectar bordes
            bordes = cv2.Canny(gris_uint8, 100, 200)
            densidad_bordes = np.sum(bordes > 0) / (bordes.shape[0] * bordes.shape[1])
            
            # Detectar simetría vertical (característica de radiografías de tórax)
            mitad_izq = gris[:, :gris.shape[1]//2]
            mitad_der = gris[:, gris.shape[1]//2:]
            simetria = np.abs(np.mean(mitad_izq) - np.mean(mitad_der))
            
            # 3. Criterios de validación extremadamente flexibles
            criterios = {
                'intensidad': 0.1 < media < 0.9,     # Muy flexible
                'contraste': 0.01 < std < 0.6,       # Muy flexible
                'entropia': entropia > 2.0,          # Muy flexible
                'bordes': 0.01 < densidad_bordes < 0.5,  # Muy flexible
                'simetria': simetria < 0.4,          # Muy flexible
                'tamano': True,                      # Ya no verificamos tamaño mínimo
                'aspecto': True                      # Ya no verificamos relación de aspecto
            }
            
            # Verificar todos los criterios
            es_radiografia = all(criterios.values())
            
            # Mostrar diagnóstico detallado
            print("\nDiagnóstico de la imagen:")
            print(f"Dimensiones: {imagen_procesada.shape}")
            print(f"Media: {media:.3f} (debe estar entre 0.1 y 0.9)")
            print(f"Desviación estándar: {std:.3f} (debe estar entre 0.01 y 0.6)")
            print(f"Entropía: {entropia:.3f} (debe ser > 2.0)")
            print(f"Densidad de bordes: {densidad_bordes:.3f} (debe estar entre 0.01 y 0.5)")
            print(f"Simetría: {simetria:.3f} (debe ser < 0.4)")
            
            if not es_radiografia:
                print("\nRazones de rechazo:")
                for criterio, cumple in criterios.items():
                    if not cumple:
                        print(f"- No cumple el criterio de {criterio}")
            
            return es_radiografia, imagen_procesada
            
        except Exception as e:
            print(f"Error al detectar radiografía: {str(e)}")
            return False, imagen_procesada
    
    def extraer_caracteristicas(self, imagen):
        """
        Extrae características de la imagen para el modelo.
        
        Args:
            imagen (numpy.ndarray): Imagen preprocesada
            
        Returns:
            numpy.ndarray: Vector de características de forma (1024,)
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(imagen.shape) == 3:
                imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
            else:
                imagen_gris = imagen
            
            # Redimensionar a 32x32 para extraer características
            imagen_pequena = cv2.resize(imagen_gris, (32, 32))
            
            # Normalizar
            imagen_pequena = imagen_pequena.astype(np.float32) / 255.0
            
            # Aplanar y asegurar que tiene la forma correcta
            caracteristicas = imagen_pequena.flatten()
            
            # Si el vector es más corto que 1024, rellenar con ceros
            if len(caracteristicas) < 1024:
                caracteristicas = np.pad(caracteristicas, (0, 1024 - len(caracteristicas)))
            # Si es más largo, truncar
            elif len(caracteristicas) > 1024:
                caracteristicas = caracteristicas[:1024]
            
            return caracteristicas
            
        except Exception as e:
            print(f"Error al extraer características: {str(e)}")
            raise
    
    def ajustar_imagen(self, imagen):
        # Ajustar contraste
        alpha = 1.5  # Contraste
        beta = 10    # Brillo
        return cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta) 