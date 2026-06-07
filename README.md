# apppatricio

Aplicacion Python/Tkinter para analisis de radiografias de torax con TensorFlow/Keras. Clasifica imagenes en categorias como normal, neumonia, tuberculosis, cancer pulmonar, derrame pleural, EPOC y fibrosis pulmonar, y genera recomendaciones contextuales segun datos del paciente.

## Restaurar despues de clonar

1. Clonar el repositorio con Git LFS instalado:

```powershell
git lfs install
git clone https://github.com/ScottCSC/apppatricio.git
cd apppatricio
git checkout Test-Torax
git lfs pull
```

2. Crear entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

4. Ejecutar la aplicacion:

```powershell
python app.py
```

## Archivos grandes

Los modelos grandes dentro de `models/*.h5` se respaldan con Git LFS porque superan el limite normal de GitHub para archivos individuales.
