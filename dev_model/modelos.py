import os
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    EfficientNetB0
)

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score


# configuración

IMG_SIZE = 224
BATCH_SIZE = 32

# 5 épocas iniciales + 5 de fine-tuning
EPOCHS_BASE = 5
EPOCHS_FINE = 5

# usar solamente el 20% del dataset
PORCENTAJE_DATASET = 0.20

NUM_CLASES = 8

RUTA_DATASET = "dataset"


# métrica one-off

def one_off_accuracy_numpy(y_true, y_pred):

    diferencia = np.abs(y_true - y_pred)

    return np.mean(diferencia <= 1)


def one_off_accuracy(y_true, y_pred):

    y_true = tf.argmax(y_true, axis=1)

    y_pred = tf.argmax(y_pred, axis=1)

    diferencia = tf.abs(y_true - y_pred)

    return tf.reduce_mean(
        tf.cast(diferencia <= 1, tf.float32)
    )


# tomar solo el 20% del dataset

def obtener_subset(ruta_base, porcentaje=0.2):

    rutas = []
    etiquetas = []

    clases = sorted(os.listdir(ruta_base))

    for indice, clase in enumerate(clases):

        carpeta = os.path.join(ruta_base, clase)

        if not os.path.isdir(carpeta):
            continue

        imagenes = [
            img for img in os.listdir(carpeta)
            if img.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        cantidad = max(1, int(len(imagenes) * porcentaje))

        seleccionadas = random.sample(imagenes, cantidad)

        for img in seleccionadas:

            rutas.append(os.path.join(carpeta, img))

            etiquetas.append(indice)

    return rutas, etiquetas, clases


# carga de imágenes

def cargar_imagen(ruta, etiqueta):

    imagen = tf.io.read_file(ruta)

    imagen = tf.image.decode_image(imagen, channels=3)

    imagen = tf.image.resize(imagen, [IMG_SIZE, IMG_SIZE])

    imagen = tf.cast(imagen, tf.float32) / 255.0

    etiqueta = tf.one_hot(etiqueta, depth=NUM_CLASES)

    return imagen, etiqueta


def crear_dataset(rutas, etiquetas):

    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))

    ds = ds.map(cargar_imagen)

    ds = ds.shuffle(1000)

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# obtener datos

train_paths, train_labels, clases = obtener_subset(
    os.path.join(RUTA_DATASET, "train"),
    PORCENTAJE_DATASET
)

val_paths, val_labels, _ = obtener_subset(
    os.path.join(RUTA_DATASET, "val"),
    PORCENTAJE_DATASET
)

test_paths, test_labels, _ = obtener_subset(
    os.path.join(RUTA_DATASET, "test"),
    PORCENTAJE_DATASET
)

train_ds = crear_dataset(train_paths, train_labels)

val_ds = crear_dataset(val_paths, val_labels)

test_ds = crear_dataset(test_paths, test_labels)


# crear modelo

def crear_modelo(base_model):

    base_model.trainable = False

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = base_model(inputs, training=False)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.3)(x)

    outputs = Dense(NUM_CLASES, activation='softmax')(x)

    modelo = Model(inputs, outputs)

    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', one_off_accuracy]
    )

    return modelo


# modelos a comparar

modelos = {

    "MobileNetV2": MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    "ResNet50": ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    "EfficientNetB0": EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
}


# entrenamiento

resultados = {}

for nombre, base in modelos.items():

    print("\nentrenando:", nombre)

    modelo = crear_modelo(base)

    # entrenamiento inicial

    historia = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_BASE
    )

    # fine-tuning

    base.trainable = True

    for layer in base.layers[:-20]:
        layer.trainable = False

    modelo.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', one_off_accuracy]
    )

    historia_fine = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE
    )

    # evaluación

    predicciones = []
    reales = []

    for x_batch, y_batch in test_ds:

        y_pred = modelo.predict(x_batch, verbose=0)

        predicciones.extend(np.argmax(y_pred, axis=1))

        reales.extend(np.argmax(y_batch.numpy(), axis=1))

    accuracy = accuracy_score(reales, predicciones)

    oneoff = one_off_accuracy_numpy(
        np.array(reales),
        np.array(predicciones)
    )

    resultados[nombre] = {
        "accuracy": float(accuracy),
        "one_off_accuracy": float(oneoff)
    }

    print(f"\naccuracy: {accuracy:.4f}")

    print(f"one-off accuracy: {oneoff:.4f}")


# guardar resultados

with open("resultados_modelos.json", "w") as archivo:

    json.dump(resultados, archivo, indent=4)


# gráficas

nombres = list(resultados.keys())

accs = [resultados[n]["accuracy"] for n in nombres]

oneoffs = [resultados[n]["one_off_accuracy"] for n in nombres]


fig, axs = plt.subplots(1, 2, figsize=(12, 5))


# accuracy

axs[0].bar(
    nombres,
    accs,
    color=['royalblue', 'darkorange', 'seagreen']
)

axs[0].set_title("accuracy")

axs[0].set_ylim(0, 1)

axs[0].grid(alpha=0.3)


# one-off accuracy

axs[1].bar(
    nombres,
    oneoffs,
    color=['crimson', 'purple', 'teal']
)

axs[1].set_title("one-off accuracy")

axs[1].set_ylim(0, 1)

axs[1].grid(alpha=0.3)

plt.tight_layout()

plt.show()