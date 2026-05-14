"""
Created on May 5 2026
EDA: Análisis exploratorio del dataset
@author: Andrea Gregorio
@author: Hannah García
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import kagglehub
import matplotlib.image as mpimg


class EDA:
  def __init__(self, df, train_images=None, dataset_path=None):
    self.df = df
    self.train_images = train_images
    self.dataset_path = dataset_path

  def obtener_total_imagenes(self):
    return len(self.df)

  def obtener_total_clases(self):
    return len(self.df['age'].unique())

  def obtener_clases(self):
    return self.df['age'].unique()

  def mostrar_distribucion_edades(self):
    age_order = [
        '0-2','3-9','10-19','20-29',
        '30-39','40-49','50-59',
        '60-69','more than 70'
    ]

    age_counts = self.df['age'].value_counts().reindex(age_order)
    print(age_counts)

    plt.figure(figsize=(8,5))
    age_counts.plot(kind='bar')
    plt.title("Edad | FairFace")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

  def mostrar_distribucion_genero(self):
    gender_counts = self.df['gender'].value_counts()

    print("Conteo de género:")
    print(gender_counts)

    plt.bar(gender_counts.index, gender_counts.values)
    plt.title("Género | FairFace")
    plt.xlabel("Género")
    plt.ylabel("Frecuencia")
    plt.show()

  def mostrar_distribucion_raza(self):
    race_counts = self.df['race'].value_counts()

    print("Conteo de raza:")
    print(race_counts)

    plt.bar(race_counts.index, race_counts.values)
    plt.title("Raza | FairFace")
    plt.xlabel("Raza")
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=45)
    plt.show()

  def reporte_sesgo(self):
    tabla_raza = pd.crosstab(self.df["age"].reset_index(drop=True),
                              self.df["race"].reset_index(drop=True))
    tabla_genero = pd.crosstab(self.df["age"].reset_index(drop=True),
                                self.df["gender"].reset_index(drop=True))

    print("\nDistribución de raza por edad:")
    print(tabla_raza)

    print("\nDistribución de género por edad:")
    print(tabla_genero)

if __name__=='__main__':
    """Obtener el conjunto de datos para luego hacer la exploración"""
    # Obtener datos
    path = kagglehub.dataset_download("aibloy/fairface")
    dataset_path_2 = path + "/FairFace"

    print("Path to dataset files:", path)

    # Rutas a carpetas de imágenes
    train_images = os.path.join(dataset_path_2, "train")
    val_images = os.path.join(dataset_path_2, "val")

    # Rutas a CSV
    train_csv = os.path.join(dataset_path_2, "train_labels.csv")
    val_csv = os.path.join(dataset_path_2, "val_labels.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    df = pd.concat([train_df, val_df])

    eda_FairFace = EDA(df)
    clases = eda_FairFace.obtener_clases()
    print("Clases de edad:", clases)
    """eda_FairFace.mostrar_distribucion_edades()
    eda_FairFace.mostrar_distribucion_genero()
    eda_FairFace.mostrar_distribucion_raza()
    eda_FairFace.reporte_sesgo()

    # Imprimir una imagen y su etiqueta
    n = 300
    img = mpimg.imread(os.path.join(train_images,f"{n}.jpg"))
    plt.imshow(img)
    plt.show()
    print(train_df.iloc[n-1])
  """