"""
preparar_dataset.py
script para crear los objetos Dataset a partir del dataset reorganizado de FairFace.
@author: Andrea Gregorio
@date: 2024-06
"""
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):

        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        self.orden_clases = {"0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3, "30-39": 4, "40-49": 5, "50-59": 6, "60-69": 7, "more than 70": 8}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        idx = int(idx)

        fila = self.df.iloc[idx]

        nombre_imagen = fila["file"].split("/")[1]

        ruta_imagen = os.path.join(self.img_dir, nombre_imagen)

        image = Image.open(ruta_imagen).convert("RGB")

        if self.transform:
            image = self.transform(image)

        edad = fila["age"]

        label = self.orden_clases[edad]

        return image, label
