# Clasificador_Edad
Proyecto de creación de clasificador por Redes Neuronales para clasificar personas en rangos de edad a partir de imágenes faciales utilizando modelos Vision Transformer y redes neuronales convolucionales.

Autoras: Andrea Itzel Gregorio Martínez y Hannah Sarahi

## Índice

- [Objetivo](#objetivo)
- [Instalación](#instalación)
- [Uso](#uso)
- [Resultados](#resultados)
- [Descripción de contenido](#contenido)

## Objetivo

Desarrollar y comparar modelos de visión por computadora capaces de clasificar personas en rangos de edad a partir de imágenes faciales. El proyecto evalúa distintas arquitecturas de aprendizaje profundo, incluyendo redes neuronales convolucionales y modelos basados en Transformers, con el fin de identificar aquellas que ofrecen mejor desempeño y mayor interpretabilidad.
---

## Instalación
### Requisitos
* Python 3.10 o superior
* Dependencias incluidas en `requirements.txt`

### Pasos

1. Clonar el repositorio

```bash
git clone https://github.com/gregoria-i/Clasificador_Edad.git
```

2. Crear entorno virtual
```bash
python -m venv .clasificador
```
3. Instalar dependencias
```bash
pip install -r requirements.txt
```
4. Descargar y organizar datos
```bash
python dev_model\reorganizar_datos.py
```
Este archivo ya hará la descarga del dataset FairFace de Kaggle.
Obs. Si abres el csv con Excel, los intervalos de edad 03-09 y 10-29 pueden ser confundidos con fechas. Este problema no afecta el procesamiento del archivo como CSV.


## Uso

### Entrenamiento de modelos ViT

```bash
python dev_model/vit/{nombre_del_archivo}.py
```

Cada script de la carpeta `vit/` está preparado para entrenar un modelo Vision Transformer. Los resultados se almacenan en formato JSON dentro de `dev_model/resultados/`.


###  Comparación de modelos

```bash
python dev_model/comparar_modelos.py
```

Este script analiza los archivos JSON generados por los entrenamientos y produce:

* Tabla resumen de métricas organizada de mayor a menor valor de One-off.
* Gráfica comparativa de pérdida.
* Gráfica comparativa de accuracy.
* Matrices de confusión por frecuencia.
* Matrices de confusión normalizadas.

### Comunicación FastAPI y frontend

Actualmente en desarrollo.

### Resultados

Comparación de modelos ViT con capas descongeladas utilizando 20% del dataset.
| Modelo   | Test Accuracy | One-Off Accuracy |
| -------- | -------- | ---------------- |
| deit3_small_patch_16_224.fb_in22k_ft_ink1 | 52.2%  | 91.2 %          |
| vit_tiny_patch16_224.augreg_in21k | 52.2 %  | 90.7 %          |
| xcit_small_12_p16_224.fb_in1k |  48.9%  | 89.9 %          |
| swin_tiny_patch4_window7_224.ms_ink1 | 49.6%  | 89.7 %          |
| deit_tiny_patch16_224.fb_in1k | 49%  | 89.3 %          |
| xcit_tiny_12_p16_224.fb_in1k |  48.6%  | 89.1 %          |

Luego se probaron versiones más grandes de los primeros 4 modelos con el dataset al 100%, y el mejor modelo resultó ser: **swin_small_patch4_window7_224.ms_ink1** que alcanzó:
- Test Accuracy : 55.1%
- One-off Accuracy: 94.1%

## Contenido
### Carpeta `dev_model/`

Contiene:

* `EDA.py`, que realiza el análisis exploratorio de datos (EDA) del dataset FairFace.
* `reorganizar_datos.py`, encargado de crear la estructura de carpetas para entrenamiento, validación y prueba.
* `preparar_dataset.py`, que genera los objetos Dataset utilizados por los modelos.
* La carpeta `dev_model/vit`: Scripts de entrenamiento para modelos Vision Transformer 

* La carpeta `dev_model/cnn`: Scripts de entrenamiento para modelo de red neuronal convolucionales ResNet50.

* La carpeta `resultados/`, que almacena los archivos JSON generados durante el entrenamiento.

Cada archivo JSON contiene:

* Nombre del modelo: "model_name"
* Valores de pérdida por época: "train_loss" y "val_loss"
* Valores de accuracy por época: "train_acc" y "val_acc"
* Métrica One-Off Accuracy.: "one-off"
* Valores reales: "y_true"
* Predicciones finales: "y_pred"

### Explicabilidad del modelo

El archivo `grad_cam.py` permite visualizar las regiones de atención del modelo seleccionado mediante Grad-CAM.

En particular, se utiliza el modelo `swin_small_patch4_window7_224.ms_in22k_ft_in1k`, seleccionado por su desempeño en las métricas evaluadas. La herramienta genera visualizaciones de las zonas de la imagen que reciben mayor atención en las 4 etapas (*stages*) de la arquitectura Swin Transformer.
