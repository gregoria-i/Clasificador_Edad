"""
reorganize_dataset.py
Este script reorganiza el dataset de FairFace para poder separar los datos de entrenamiento, 
validación y prueba en carpetas distintas, cuidando los csv de etiquetas correspondientes a cada conjunto.
@author: Andrea Gregorio
"""
# Obtenemos el conjunto de datos
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class reorganizador():
  def __init__(self, dataset_kaggle, dataset_name):
     self.dataset_kaggle = dataset_kaggle
     self.dataset_name = dataset_name

     self.obtener_datos()
     self.reorganizar_etiquetas_70_15_15()
     self.reorganizar_imagenes_en_carpetas()

  def obtener_datos(self):
    path = kagglehub.dataset_download(self.dataset_kaggle)
    self.path = os.path.join(path, self.dataset_name)
    print("Path to dataset files:", self.path)
   
    # Rutas a carpetas de imágenes
    self.train_images = os.path.join(self.path, "train")
    self.val_images = os.path.join(self.path, "val")

    # Rutas a CSV
    self.train_csv = os.path.join(self.path, "train_labels.csv")
    self.val_csv = os.path.join(self.path, "val_labels.csv")

  def reorganizar_etiquetas_70_15_15(self):
    # Separación actual de los datos
    train_df = pd.read_csv(self.train_csv)
    val_df = pd.read_csv(self.val_csv)

    df = pd.concat([train_df, val_df])

    # Juntar conjuntos de imágenes y distribuir 70/15/15
    label_col = "age"  # mantener proporciones entre train, test y validation
    self.output_path = "fairface_reorganizado"

    try:
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Directorio creado o ya existe: {self.output_path}")
    except OSError as e:
        print(f"Error creando el directorio '{self.output_path}': {e}")


    if "file" in df.columns:
      df = df.drop_duplicates(subset="file")
    elif "image" in df.columns:
      df = df.drop_duplicates(subset="image")

    self.train_df_new, temp_df = train_test_split(df,test_size=0.3, stratify=df[label_col], random_state=121)  # 70 y 30
    self.val_df_new, self.test_df_new = train_test_split(temp_df, test_size=0.5, stratify=temp_df[label_col], random_state=121)  # 15 y 15

    self.train_df_new.to_csv(os.path.join(self.output_path, "entrenamiento.csv"), index=False)
    self.val_df_new.to_csv(os.path.join(self.output_path, "validacion.csv"), index=False)
    self.test_df_new.to_csv(os.path.join(self.output_path, "prueba.csv"), index=False)

  def reorganizar_imagenes_en_carpetas(self):
    ruta_prueba = os.path.join(self.output_path, "Imagenes_prueba")
    os.makedirs(ruta_prueba, exist_ok=True)
    for n in range(len(self.test_df_new)):
      try:
        carpeta, nombre = self.test_df_new["file"].iloc[n].split("/")
        if carpeta == "train":
          ruta_actual = os.path.join(self.train_images, nombre)
        elif carpeta == "val":
          ruta_actual = os.path.join(self.val_images, nombre)
      
        ruta_nueva = os.path.join(ruta_prueba, nombre)
        os.replace(ruta_actual, ruta_nueva)
      except Exception as e:
        print(f"Error procesando la imagen: {e}")
        continue

    ruta_val = os.path.join(self.output_path, "Imagenes_validacion")
    os.makedirs(ruta_val, exist_ok=True)
    for n in range(len(self.val_df_new)):
      try:
        carpeta, nombre = self.val_df_new["file"].iloc[n].split("/")
        if carpeta == "train":
          ruta_actual = os.path.join(self.train_images, nombre)
        elif carpeta == "val":
          ruta_actual = os.path.join(self.val_images, nombre)
      
        ruta_nueva = os.path.join(ruta_val, nombre)
        os.replace(ruta_actual, ruta_nueva)
      except Exception as e:
        print(f"Error procesando la imagen: {e}")
        continue

    ruta_ent = os.path.join(self.output_path, "Imagenes_entrenamiento")
    os.makedirs(ruta_ent, exist_ok=True)
    for n in range(len(self.train_df_new)):
      try:
        carpeta, nombre = self.train_df_new["file"].iloc[n].split("/")
        if carpeta == "train":
          ruta_actual = os.path.join(self.train_images, nombre)
        elif carpeta == "val":
          ruta_actual = os.path.join(self.val_images, nombre)
      
        ruta_nueva = os.path.join(ruta_ent, nombre)
        os.replace(ruta_actual, ruta_nueva)
      except Exception as e:
        print(f"Error procesando la imagen: {e}")
        continue

  def mostrar_imagen(self, ruta):
    img = mpimg.imread(ruta)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
   obj = reorganizador("aibloy/fairface", "FairFace")
