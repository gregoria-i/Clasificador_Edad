"""
modelo_resnet.py
Clasificación de edades usando ResNet50
sobre el dataset FairFace.

@author: Hannah García
@date: 2026-05
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)


class ModeloResNet:

    def __init__(self, ruta_dataset):

        self.ruta_dataset = ruta_dataset

        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 10

        self.cargar_datos()

        self.crear_modelo()

    def cargar_datos(self):

        train_csv = os.path.join(
            self.ruta_dataset,
            "entrenamiento.csv"
        )

        val_csv = os.path.join(
            self.ruta_dataset,
            "validacion.csv"
        )

        test_csv = os.path.join(
            self.ruta_dataset,
            "prueba.csv"
        )

        self.train_df = pd.read_csv(train_csv)
        self.val_df = pd.read_csv(val_csv)
        self.test_df = pd.read_csv(test_csv)

        self.encoder = LabelEncoder()

        y_train = self.encoder.fit_transform(
            self.train_df["age"]
        )

        y_val = self.encoder.transform(
            self.val_df["age"]
        )

        y_test = self.encoder.transform(
            self.test_df["age"]
        )

        self.num_classes = len(np.unique(y_train))

        self.y_train = to_categorical(
            y_train,
            self.num_classes
        )

        self.y_val = to_categorical(
            y_val,
            self.num_classes
        )

        self.y_test = to_categorical(
            y_test,
            self.num_classes
        )

        self.X_train = self.cargar_imagenes(
            self.train_df,
            "Imagenes_entrenamiento"
        )

        self.X_val = self.cargar_imagenes(
            self.val_df,
            "Imagenes_validacion"
        )

        self.X_test = self.cargar_imagenes(
            self.test_df,
            "Imagenes_prueba"
        )

    def cargar_imagenes(self, df, carpeta):

        imagenes = []

        ruta_carpeta = os.path.join(
            self.ruta_dataset,
            carpeta
        )

        for archivo in df["file"]:

            nombre = archivo.split("/")[-1]

            ruta = os.path.join(
                ruta_carpeta,
                nombre
            )

            try:

                img = load_img(
                    ruta,
                    target_size=self.img_size
                )

                img = img_to_array(img)

                img = img / 255.0

                imagenes.append(img)

            except Exception as e:

                print(f"Error cargando imagen: {e}")

        return np.array(imagenes)

    def crear_modelo(self):

        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

        base_model.trainable = False

        self.model = Sequential([

            base_model,

            GlobalAveragePooling2D(),

            Dense(
                256,
                activation="relu"
            ),

            Dropout(0.4),

            Dense(
                self.num_classes,
                activation="softmax"
            )
        ])

        self.model.compile(

            optimizer=Adam(
                learning_rate=0.001
            ),

            loss="categorical_crossentropy",

            metrics=["accuracy"]
        )

    def entrenar(self):

        self.history = self.model.fit(

            self.X_train,
            self.y_train,

            validation_data=(
                self.X_val,
                self.y_val
            ),

            epochs=self.epochs,

            batch_size=self.batch_size
        )

    def evaluar(self):

        predicciones = self.model.predict(
            self.X_test
        )

        y_pred = np.argmax(
            predicciones,
            axis=1
        )

        y_true = np.argmax(
            self.y_test,
            axis=1
        )

        acc = accuracy_score(
            y_true,
            y_pred
        )

        print(
            f"Accuracy en prueba: {acc:.4f}"
        )

        resultados = {

            "model_name": "ResNet50",

            "test_acc": float(acc),

            "train_loss":
                self.history.history["loss"],

            "val_loss":
                self.history.history["val_loss"],

            "train_acc":
                self.history.history["accuracy"],

            "val_acc":
                self.history.history["val_accuracy"],

            "y_true":
                y_true.tolist(),

            "y_pred":
                y_pred.tolist()
        }

        os.makedirs(
            "resultados",
            exist_ok=True
        )

        with open(
            "resultados/resnet50_resultados.json",
            "w"
        ) as f:

            json.dump(
                resultados,
                f
            )

        print(
            "Resultados guardados"
        )

    def graficar_historial(self):

        plt.plot(
            self.history.history["accuracy"],
            label="Entrenamiento"
        )

        plt.plot(
            self.history.history["val_accuracy"],
            label="Validación"
        )

        plt.title("Precisión")

        plt.xlabel("Épocas")

        plt.ylabel("Accuracy")

        plt.legend()

        plt.grid()

        plt.show()


if __name__ == "__main__":

    ruta_dataset = "fairface_reorganizado"

    modelo = ModeloResNet(
        ruta_dataset
    )

    modelo.entrenar()

    modelo.evaluar()

    modelo.graficar_historial()
    