# Clasificador_Edad
Red Neuronal para clasificar edad de las personas a partir de una foto de su rostro.

## Instalación

### 1. Clonar el repositorio

git clone 

### 2. Crear entorno virtual
python -m venv .clasificador

### 3. Instalar dependencias
pip install -r requirements.txt

### 4. Descargar y organizar datos
python reorganizar_datos.py
Obs. Si abres el csv con Excel, los intervalos de edad 03-09 y 10-29 pueden ser confundidos con fechas, pero tratado como csv no hay problema

## Comparación de modelos
python comparar_modelos.py
Se creó el script comparar_modelos.py para leer archivos json con los resultados de los modelos, de manera que devuelve una tabla resumen, gráfica de pérdidas (comparativa entre modelos y entre conjuntos de entrenamiento y validación), gráfica de precisión (también comparativa), y la matriz de confusión de cada modelo.
