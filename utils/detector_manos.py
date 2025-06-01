import mediapipe as mp
import cv2
import numpy as np

class DetectorManos:
    def __init__(self):
        self.mp_manos = mp.solutions.hands
        self.manos = self.mp_manos.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def detectar_mano(self, imagen):
        """
        Detecta si hay una mano en la imagen.
        
        Args:
            imagen (numpy.ndarray): Imagen a analizar (formato BGR de OpenCV)
            
        Returns:
            tuple: (hay_mano, imagen_recortada)
                - hay_mano (bool): True si se detectó una mano
                - imagen_recortada (numpy.ndarray): Imagen recortada alrededor de la mano
        """
        # Convertir a RGB (MediaPipe usa RGB)
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        # Detectar manos
        resultados = self.manos.process(imagen_rgb)
        
        if not resultados.multi_hand_landmarks:
            return False, None
        
        # Obtener coordenadas de la mano
        landmarks = resultados.multi_hand_landmarks[0].landmark
        
        # Convertir coordenadas normalizadas a píxeles
        h, w = imagen.shape[:2]
        puntos = []
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            puntos.append((x, y))
        
        # Encontrar el rectángulo que contiene la mano
        puntos = np.array(puntos)
        x_min, y_min = np.min(puntos, axis=0)
        x_max, y_max = np.max(puntos, axis=0)
        
        # Agregar margen
        margen = 20
        x_min = max(0, x_min - margen)
        y_min = max(0, y_min - margen)
        x_max = min(w, x_max + margen)
        y_max = min(h, y_max + margen)
        
        # Recortar la imagen
        imagen_recortada = imagen[y_min:y_max, x_min:x_max]
        
        return True, imagen_recortada
    
    def __del__(self):
        self.manos.close() 