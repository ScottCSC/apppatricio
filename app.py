import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import os
from utils.preprocessing import preprocesar_imagen
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import subprocess

class AplicacionClasificacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Clasificación de Manos")
        self.root.geometry("1200x800")
        
        # Asegurarse de que existe el directorio de modelos
        os.makedirs('models', exist_ok=True)
        
        # Cargar modelo
        self.modelo_cargado = False
        self.cargar_modelo()
        
        self.crear_interfaz()
    
    def cargar_modelo(self):
        """Intenta cargar el modelo y actualiza el estado."""
        try:
            self.modelo = tf.keras.models.load_model('models/modelo_final.h5')
            self.modelo_cargado = True
        except:
            self.modelo_cargado = False
    
    def crear_interfaz(self):
        # Frame principal
        self.frame_principal = ttk.Frame(self.root, padding="10")
        self.frame_principal.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Título
        ttk.Label(self.frame_principal, 
                 text="Sistema de Clasificación de Manos", 
                 font=('Arial', 16)).grid(row=0, column=0, columnspan=2, pady=20)
        
        # Estado del modelo
        self.label_estado = ttk.Label(self.frame_principal, 
                                    text="Estado del modelo: No cargado" if not self.modelo_cargado else "Estado del modelo: Cargado",
                                    font=('Arial', 12))
        self.label_estado.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Frame para la imagen
        self.frame_imagen = ttk.LabelFrame(self.frame_principal, text="Imagen", padding="10")
        self.frame_imagen.grid(row=2, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas para mostrar la imagen
        self.canvas = tk.Canvas(self.frame_imagen, width=400, height=400)
        self.canvas.grid(row=0, column=0, padx=5, pady=5)
        
        # Frame para controles
        self.frame_controles = ttk.LabelFrame(self.frame_principal, text="Controles", padding="10")
        self.frame_controles.grid(row=2, column=1, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Botones
        ttk.Button(self.frame_controles, 
                  text="Seleccionar Imagen", 
                  command=self.seleccionar_imagen).grid(row=0, column=0, pady=5)
        
        ttk.Button(self.frame_controles, 
                  text="Clasificar", 
                  command=self.clasificar_imagen).grid(row=1, column=0, pady=5)
        
        ttk.Button(self.frame_controles, 
                  text="Organizar Imágenes", 
                  command=self.abrir_organizador).grid(row=2, column=0, pady=5)
        
        ttk.Button(self.frame_controles, 
                  text="Entrenar Modelo", 
                  command=self.entrenar_modelo).grid(row=3, column=0, pady=5)
        
        # Frame para resultados
        self.frame_resultados = ttk.LabelFrame(self.frame_principal, text="Resultados", padding="10")
        self.frame_resultados.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky=(tk.W, tk.E))
        
        # Área de texto para resultados
        self.texto_resultados = tk.Text(self.frame_resultados, height=10, width=80)
        self.texto_resultados.grid(row=0, column=0, padx=5, pady=5)
        
        # Variables
        self.ruta_imagen = None
        self.imagen_original = None
    
    def seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png")]
        )
        if ruta:
            self.ruta_imagen = ruta
            self.mostrar_imagen(ruta)
    
    def mostrar_imagen(self, ruta):
        # Cargar y redimensionar imagen
        imagen = Image.open(ruta)
        imagen = imagen.resize((400, 400), Image.Resampling.LANCZOS)
        self.imagen_original = imagen
        
        # Convertir para tkinter
        imagen_tk = ImageTk.PhotoImage(imagen)
        
        # Mostrar en canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imagen_tk)
        self.canvas.image = imagen_tk  # Mantener referencia
    
    def clasificar_imagen(self):
        if not self.modelo_cargado:
            messagebox.showerror("Error", "No hay modelo cargado para clasificar. Por favor, entrena el modelo primero.")
            return
        
        if not self.ruta_imagen:
            messagebox.showerror("Error", "Por favor, selecciona una imagen primero")
            return
        
        # Preprocesar imagen
        imagen_procesada = preprocesar_imagen(self.ruta_imagen)
        if imagen_procesada is None:
            messagebox.showerror("Error", "No se pudo procesar la imagen")
            return
        
        # Realizar predicción
        prediccion = self.modelo.predict(imagen_procesada.reshape(1, 224, 224, 3))
        categoria = np.argmax(prediccion[0])
        confianza = prediccion[0][categoria]
        
        categorias = ['Normal', 'Inflamación', 'Lesión']
        resultado = f"Diagnóstico: {categorias[categoria]}\n"
        resultado += f"Confianza: {confianza:.2%}\n\n"
        
        # Agregar recomendaciones
        recomendaciones = {
            'Normal': [
                "• Mantener la rutina de ejercicios de mano",
                "• Continuar con las medidas preventivas básicas"
            ],
            'Inflamación': [
                "• Aplicar hielo por 15-20 minutos cada 2-3 horas",
                "• Elevar la mano para reducir la inflamación",
                "• Considerar antiinflamatorios si es apropiado",
                "• Programar cita con especialista si persiste"
            ],
            'Lesión': [
                "• Buscar atención médica inmediata",
                "• Inmovilizar la mano si es necesario",
                "• Evitar movimientos que causen dolor",
                "• Documentar síntomas y evolución"
            ]
        }
        
        resultado += "Recomendaciones:\n"
        for rec in recomendaciones[categorias[categoria]]:
            resultado += f"{rec}\n"
        
        # Mostrar resultados
        self.texto_resultados.delete(1.0, tk.END)
        self.texto_resultados.insert(tk.END, resultado)
    
    def abrir_organizador(self):
        os.system('python utils/organizar_imagenes.py')
    
    def entrenar_modelo(self):
        # Verificar si hay imágenes para entrenar
        if not os.path.exists('data/processed/normal') or \
           not os.path.exists('data/processed/inflamacion') or \
           not os.path.exists('data/processed/lesion'):
            messagebox.showerror("Error", 
                "No hay imágenes organizadas para entrenar. Por favor, usa 'Organizar Imágenes' primero.")
            return
        
        if messagebox.askyesno("Confirmar", 
            "¿Deseas entrenar el modelo? Esto puede tomar varios minutos."):
            try:
                # Ejecutar el entrenamiento
                proceso = subprocess.Popen(['python', 'train.py'], 
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
                            self.cargar_modelo()
                            self.label_estado.config(text="Estado del modelo: Cargado")
                        else:
                            error = proceso.stderr.read()
                            messagebox.showerror("Error", f"Error al entrenar el modelo: {error}")
                
                actualizar_progreso()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al entrenar el modelo: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionClasificacion(root)
    root.mainloop() 