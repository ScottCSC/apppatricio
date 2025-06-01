import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import os
from utils.detector_radiografias import DetectorRadiografias
from models.clasificador_enfermedades import ClasificadorEnfermedades
from models.recomendador_contextual import RecomendadorContextual
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import subprocess

class AplicacionRadiografias:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Análisis de Radiografías de Tórax")
        self.root.geometry("1200x800")
        
        # Inicializar componentes
        self.detector = DetectorRadiografias()
        self.clasificador = ClasificadorEnfermedades()
        self.recomendador = RecomendadorContextual()
        
        # Asegurarse de que existe el directorio de modelos
        os.makedirs('models', exist_ok=True)
        
        # Cargar modelo
        self.modelo_cargado = False
        self.cargar_modelo()
        
        # Definir enfermedades
        self.enfermedades = [
            'Normal',
            'Neumonía',
            'Tuberculosis',
            'Cáncer de pulmón',
            'Derrame pleural',
            'EPOC',
            'Fibrosis pulmonar'
        ]
        
        # Variables
        self.imagen_actual = None
        self.ruta_imagen = None
        self.contexto_paciente = {
            'edad': 50,
            'fumador': False,
            'historial_familiar': False,
            'sintomas': [],
            'enfermedades_previas': []
        }
        
        self.crear_interfaz()
    
    def cargar_modelo(self):
        """Intenta cargar el modelo y actualiza el estado."""
        try:
            # Intentar cargar desde la raíz del proyecto
            self.modelo = tf.keras.models.load_model('modelo_radiografias.h5')
            self.modelo_cargado = True
            print("Modelo cargado exitosamente desde modelo_radiografias.h5")
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            self.modelo_cargado = False
    
    def crear_interfaz(self):
        # Frame principal
        self.frame_principal = ttk.Frame(self.root, padding="10")
        self.frame_principal.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar el grid para que se expanda
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.frame_principal.grid_columnconfigure(0, weight=1)
        self.frame_principal.grid_columnconfigure(1, weight=1)
        
        # Título
        ttk.Label(self.frame_principal, 
                 text="Sistema de Análisis de Radiografías de Tórax", 
                 font=('Arial', 16)).grid(row=0, column=0, columnspan=2, pady=20)
        
        # Estado del modelo
        self.label_estado = ttk.Label(self.frame_principal, 
                                    text="Estado del modelo: No cargado" if not self.modelo_cargado else "Estado del modelo: Cargado",
                                    font=('Arial', 12))
        self.label_estado.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Frame izquierdo para imagen y controles
        self.frame_izquierdo = ttk.Frame(self.frame_principal)
        self.frame_izquierdo.grid(row=2, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame para la imagen
        self.frame_imagen = ttk.LabelFrame(self.frame_izquierdo, text="Radiografía", padding="10")
        self.frame_imagen.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas para mostrar la imagen
        self.canvas = tk.Canvas(self.frame_imagen, width=400, height=400)
        self.canvas.grid(row=0, column=0, padx=5, pady=5)
        
        # Frame para controles
        self.frame_controles = ttk.LabelFrame(self.frame_izquierdo, text="Controles", padding="10")
        self.frame_controles.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Botones
        ttk.Button(self.frame_controles, 
                  text="Seleccionar Radiografía", 
                  command=self.seleccionar_imagen).grid(row=0, column=0, pady=5, padx=5)
        
        ttk.Button(self.frame_controles, 
                  text="Analizar", 
                  command=self.analizar_imagen).grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Button(self.frame_controles, 
                  text="Organizar Radiografías", 
                  command=self.abrir_organizador).grid(row=1, column=0, pady=5, padx=5)
        
        ttk.Button(self.frame_controles, 
                  text="Entrenar Modelo", 
                  command=self.entrenar_modelo).grid(row=1, column=1, pady=5, padx=5)
        
        # Frame derecho para contexto y resultados
        self.frame_derecho = ttk.Frame(self.frame_principal)
        self.frame_derecho.grid(row=2, column=1, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame para contexto del paciente
        self.frame_contexto = ttk.LabelFrame(self.frame_derecho, text="Contexto del Paciente", padding="10")
        self.frame_contexto.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Edad
        ttk.Label(self.frame_contexto, text="Edad:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.edad_var = tk.StringVar(value="50")
        ttk.Entry(self.frame_contexto, textvariable=self.edad_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        # Fumador
        self.fumador_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.frame_contexto, text="Fumador", variable=self.fumador_var).grid(row=1, column=0, columnspan=2, padx=5, pady=2, sticky=tk.W)
        
        # Historial familiar
        self.historial_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.frame_contexto, text="Historial familiar", variable=self.historial_var).grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky=tk.W)
        
        # Síntomas
        ttk.Label(self.frame_contexto, text="Síntomas (0-1):").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.sintomas_var = tk.StringVar(value="0.5")
        ttk.Entry(self.frame_contexto, textvariable=self.sintomas_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        # Frame para resultados
        self.frame_resultados = ttk.LabelFrame(self.frame_derecho, text="Resultados y Recomendaciones", padding="10")
        self.frame_resultados.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Área de texto para resultados con scrollbar
        self.scrollbar = ttk.Scrollbar(self.frame_resultados)
        self.scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.texto_resultados = tk.Text(self.frame_resultados, height=15, width=50, yscrollcommand=self.scrollbar.set)
        self.texto_resultados.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.scrollbar.config(command=self.texto_resultados.yview)
        
        # Frame para feedback
        self.frame_feedback = ttk.LabelFrame(self.frame_derecho, text="Feedback", padding="10")
        self.frame_feedback.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(self.frame_feedback, text="Calidad de la recomendación:").grid(row=0, column=0, padx=5, pady=2)
        self.feedback_var = tk.StringVar(value="0.5")
        ttk.Scale(self.frame_feedback, from_=-1, to=1, orient=tk.HORIZONTAL, variable=self.feedback_var).grid(row=0, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame_feedback, text="Enviar Feedback", command=self.enviar_feedback).grid(row=0, column=2, padx=5, pady=2)
        
        # Configurar expansión de frames
        self.frame_izquierdo.grid_rowconfigure(0, weight=1)
        self.frame_izquierdo.grid_columnconfigure(0, weight=1)
        self.frame_derecho.grid_rowconfigure(1, weight=1)
        self.frame_derecho.grid_columnconfigure(0, weight=1)
        self.frame_resultados.grid_rowconfigure(0, weight=1)
        self.frame_resultados.grid_columnconfigure(0, weight=1)
    
    def seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png")]
        )
        if ruta:
            print(f"Formato de imagen: {ruta.split('.')[-1].lower()}")
            imagen = cv2.imread(ruta)
            print(f"Dimensiones: {imagen.shape}")
            print(f"Tipo de datos: {imagen.dtype}")
            self.ruta_imagen = ruta
            self.mostrar_imagen(ruta)
    
    def mostrar_imagen(self, ruta):
        # Cargar imagen
        imagen = cv2.imread(ruta)
        if imagen is None:
            messagebox.showerror("Error", "No se pudo cargar la imagen")
            return
        
        # Verificar si es una radiografía válida
        es_radiografia, imagen_procesada = self.detector.detectar_radiografia(imagen)
        
        if not es_radiografia:
            messagebox.showwarning("Advertencia", 
                "La imagen no parece ser una radiografía de tórax válida. Por favor, selecciona una radiografía clara.")
            return
        
        # Convertir a RGB para mostrar
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        # Guardar la imagen original como array de NumPy
        self.imagen_actual = imagen_rgb
        
        # Convertir a PIL para mostrar
        imagen_pil = Image.fromarray(imagen_rgb)
        
        # Redimensionar para mostrar
        imagen_pil = imagen_pil.resize((400, 400), Image.Resampling.LANCZOS)
        
        # Convertir para tkinter
        imagen_tk = ImageTk.PhotoImage(imagen_pil)
        
        # Mostrar en canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imagen_tk)
        self.canvas.image = imagen_tk  # Mantener referencia
    
    def analizar_imagen(self):
        if self.imagen_actual is None:
            messagebox.showerror("Error", "Por favor seleccione una imagen primero")
            return
        
        try:
            # Actualizar contexto del paciente
            self.contexto_paciente = {
                'edad': int(self.edad_var.get()),
                'fumador': self.fumador_var.get(),
                'historial_familiar': self.historial_var.get(),
                'sintomas': [float(self.sintomas_var.get())],
                'enfermedades_previas': []
            }
            
            # Asegurar que la imagen está en el formato correcto (array de NumPy)
            if isinstance(self.imagen_actual, Image.Image):
                self.imagen_actual = np.array(self.imagen_actual)
            
            # Detectar si es una radiografía válida
            es_radiografia, imagen_procesada = self.detector.detectar_radiografia(self.imagen_actual)
            
            if not es_radiografia:
                messagebox.showerror("Error", "La imagen no parece ser una radiografía de tórax válida")
                return
            
            # Extraer características
            caracteristicas = self.detector.extraer_caracteristicas(imagen_procesada)
            
            # Obtener recomendación
            recomendacion, probabilidades = self.recomendador.obtener_recomendacion(
                caracteristicas,
                self.contexto_paciente
            )
            
            # Mostrar resultados
            self.texto_resultados.delete(1.0, tk.END)
            self.texto_resultados.insert(tk.END, f"Recomendación: {recomendacion}\n\n")
            self.texto_resultados.insert(tk.END, "Probabilidades:\n")
            for i, prob in enumerate(probabilidades):
                self.texto_resultados.insert(tk.END, f"- {self.recomendador.acciones[i]}: {prob:.2%}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al analizar la imagen: {str(e)}")
    
    def abrir_organizador(self):
        os.system('python utils/organizar_radiografias.py')
    
    def entrenar_modelo(self):
        # Verificar si hay imágenes para entrenar
        if not os.path.exists('data/processed'):
            messagebox.showerror("Error", 
                "No hay radiografías organizadas para entrenar. Por favor, usa 'Organizar Radiografías' primero.")
            return
        
        if messagebox.askyesno("Confirmar", 
            "¿Deseas entrenar el modelo? Esto puede tomar varios minutos."):
            try:
                # Ejecutar el entrenamiento
                proceso = subprocess.Popen(['python', 'train.py', '--datos', 'data/processed', '--epocas', '50', '--batch_size', '32', '--modelo_salida', 'modelo_radiografias.h5'], 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE,
                                        text=True)
                
                # Mostrar progreso
                ventana_progreso = tk.Toplevel(self.root)
                ventana_progreso.title("Entrenando Modelo")
                ventana_progreso.geometry("400x200")
                
                texto_progreso = tk.Text(ventana_progreso, height=10, width=50)
                texto_progreso.pack(padx=10, pady=10)
                
                def actualizar_progreso():
                    salida = proceso.stdout.readline()
                    if salida:
                        texto_progreso.insert(tk.END, salida)
                        texto_progreso.see(tk.END)
                        ventana_progreso.after(100, actualizar_progreso)
                    elif proceso.poll() is None:
                        ventana_progreso.after(100, actualizar_progreso)
                    else:
                        ventana_progreso.destroy()
                        if proceso.returncode == 0:
                            messagebox.showinfo("Éxito", "Modelo entrenado exitosamente")
                            self.cargar_modelo()  # Recargar el modelo después del entrenamiento
                            self.label_estado.config(text="Estado del modelo: Cargado")
                        else:
                            error = proceso.stderr.read()
                            messagebox.showerror("Error", f"Error al entrenar el modelo: {error}")
                
                actualizar_progreso()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al entrenar el modelo: {str(e)}")

    def enviar_feedback(self):
        """Envía el feedback del usuario al recomendador contextual."""
        try:
            # Obtener valor del feedback
            feedback = float(self.feedback_var.get())
            
            # Verificar que tenemos una imagen y recomendación previa
            if self.imagen_actual is None:
                messagebox.showerror("Error", "Por favor seleccione y analice una imagen primero")
                return
            
            # Obtener la última acción recomendada
            caracteristicas = self.detector.extraer_caracteristicas(self.imagen_actual)
            recomendacion, _ = self.recomendador.obtener_recomendacion(
                caracteristicas,
                self.contexto_paciente
            )
            
            # Encontrar el índice de la acción
            accion_idx = list(self.recomendador.acciones.values()).index(recomendacion)
            
            # Actualizar el modelo con el feedback
            self.recomendador.actualizar_modelo(
                caracteristicas,
                self.contexto_paciente,
                accion_idx,
                feedback
            )
            
            messagebox.showinfo("Éxito", "Feedback enviado correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al enviar feedback: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionRadiografias(root)
    root.mainloop() 