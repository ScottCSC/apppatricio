import os
import shutil
from tkinter import Tk, filedialog, messagebox
import tkinter as tk

def crear_estructura_directorios():
    """Crea la estructura de directorios necesaria."""
    directorios = [
        'data/processed/normal',
        'data/processed/inflamacion',
        'data/processed/lesion'
    ]
    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)

def organizar_imagenes():
    """Interfaz gráfica para organizar imágenes en categorías."""
    root = Tk()
    root.title("Organizador de Imágenes")
    root.geometry("600x400")
    
    def seleccionar_imagenes():
        archivos = filedialog.askopenfilenames(
            title="Seleccionar imágenes",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png")]
        )
        if archivos:
            for archivo in archivos:
                # Crear ventana para categoría
                ventana_categoria = tk.Toplevel(root)
                ventana_categoria.title("Seleccionar Categoría")
                ventana_categoria.geometry("300x200")
                
                def mover_imagen(categoria):
                    nombre_archivo = os.path.basename(archivo)
                    destino = os.path.join('data/processed', categoria, nombre_archivo)
                    shutil.copy2(archivo, destino)
                    ventana_categoria.destroy()
                    messagebox.showinfo("Éxito", f"Imagen movida a {categoria}")
                
                # Botones para cada categoría
                tk.Button(ventana_categoria, text="Normal", 
                         command=lambda: mover_imagen('normal')).pack(pady=10)
                tk.Button(ventana_categoria, text="Inflamación", 
                         command=lambda: mover_imagen('inflamacion')).pack(pady=10)
                tk.Button(ventana_categoria, text="Lesión", 
                         command=lambda: mover_imagen('lesion')).pack(pady=10)
    
    # Crear estructura de directorios
    crear_estructura_directorios()
    
    # Interfaz principal
    tk.Label(root, text="Organizador de Imágenes para Entrenamiento", 
             font=('Arial', 14)).pack(pady=20)
    
    tk.Button(root, text="Seleccionar Imágenes", 
             command=seleccionar_imagenes).pack(pady=10)
    
    tk.Label(root, text="Instrucciones:", font=('Arial', 12)).pack(pady=10)
    tk.Label(root, text="1. Haz clic en 'Seleccionar Imágenes'").pack()
    tk.Label(root, text="2. Selecciona las imágenes que quieres organizar").pack()
    tk.Label(root, text="3. Para cada imagen, selecciona su categoría").pack()
    
    root.mainloop()

if __name__ == "__main__":
    organizar_imagenes() 