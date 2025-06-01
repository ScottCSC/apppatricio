import os
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from detector_radiografias import DetectorRadiografias

class OrganizadorRadiografias:
    def __init__(self, root):
        self.root = root
        self.root.title("Organizador de Radiografías")
        self.root.geometry("800x600")
        
        # Inicializar detector
        self.detector = DetectorRadiografias()
        
        # Directorios de destino
        self.directorio_base = 'data/processed'
        self.categorias = [
            'normal',
            'neumonia',
            'tuberculosis',
            'cancer_pulmon',
            'derrame_pleural',
            'epoc',
            'fibrosis_pulmonar'
        ]
        
        # Crear directorios si no existen
        for categoria in self.categorias:
            os.makedirs(os.path.join(self.directorio_base, categoria), exist_ok=True)
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        # Frame principal
        self.frame_principal = ttk.Frame(self.root, padding="10")
        self.frame_principal.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Título
        ttk.Label(self.frame_principal, 
                 text="Organizador de Radiografías", 
                 font=('Arial', 16)).grid(row=0, column=0, columnspan=2, pady=20)
        
        # Frame para la imagen
        self.frame_imagen = ttk.LabelFrame(self.frame_principal, text="Radiografía", padding="10")
        self.frame_imagen.grid(row=1, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas para mostrar la imagen
        self.canvas = tk.Canvas(self.frame_imagen, width=400, height=400)
        self.canvas.grid(row=0, column=0, padx=5, pady=5)
        
        # Frame para controles
        self.frame_controles = ttk.LabelFrame(self.frame_principal, text="Controles", padding="10")
        self.frame_controles.grid(row=1, column=1, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Botones
        ttk.Button(self.frame_controles, 
                  text="Seleccionar Radiografía", 
                  command=self.seleccionar_imagen).grid(row=0, column=0, pady=5)
        
        # Lista de categorías
        ttk.Label(self.frame_controles, 
                 text="Selecciona la categoría:").grid(row=1, column=0, pady=5)
        
        self.categoria_var = tk.StringVar()
        self.combo_categorias = ttk.Combobox(self.frame_controles, 
                                           textvariable=self.categoria_var,
                                           values=self.categorias)
        self.combo_categorias.grid(row=2, column=0, pady=5)
        self.combo_categorias.set(self.categorias[0])
        
        ttk.Button(self.frame_controles, 
                  text="Organizar", 
                  command=self.organizar_imagen).grid(row=3, column=0, pady=5)
        
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
        imagen_pil = Image.fromarray(imagen_rgb)
        
        # Redimensionar para mostrar
        imagen_pil = imagen_pil.resize((400, 400), Image.Resampling.LANCZOS)
        self.imagen_original = imagen_pil
        
        # Convertir para tkinter
        imagen_tk = ImageTk.PhotoImage(imagen_pil)
        
        # Mostrar en canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imagen_tk)
        self.canvas.image = imagen_tk  # Mantener referencia
    
    def organizar_imagen(self):
        if not self.ruta_imagen:
            messagebox.showerror("Error", "Por favor, selecciona una radiografía primero")
            return
        
        categoria = self.categoria_var.get()
        if not categoria:
            messagebox.showerror("Error", "Por favor, selecciona una categoría")
            return
        
        # Crear nombre de archivo único
        nombre_archivo = os.path.basename(self.ruta_imagen)
        nombre_base, extension = os.path.splitext(nombre_archivo)
        contador = 1
        
        while os.path.exists(os.path.join(self.directorio_base, categoria, nombre_archivo)):
            nombre_archivo = f"{nombre_base}_{contador}{extension}"
            contador += 1
        
        # Copiar archivo a la categoría seleccionada
        try:
            shutil.copy2(self.ruta_imagen, 
                        os.path.join(self.directorio_base, categoria, nombre_archivo))
            messagebox.showinfo("Éxito", 
                              f"Radiografía organizada en la categoría: {categoria}")
            
            # Limpiar selección
            self.ruta_imagen = None
            self.canvas.delete("all")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al organizar la imagen: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OrganizadorRadiografias(root)
    root.mainloop() 