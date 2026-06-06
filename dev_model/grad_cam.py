"""
grad_cam.py
Script para generar mapas de calor a partir de una red neuronal entrenada
@author: Andrea Gregorio
@date: 2026-05
"""
# 1. Librerías y funciones auxiliares
import os

import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import timm


class ExplicaGradCAM:
    def __init__(self, carpeta, pth_path):
        self.carpeta = carpeta
        self.pth_path = pth_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")

        # Cargar modelo
        self.modelo = self.cargar_modelo()

        # Seleccionar imagen
        random.seed(10)
        self.img_path = self.seleccionar_imagen_aleatoria(self.carpeta)
        print(f"Imagen usada: {self.img_path}")

        # Construir input
        self.rgb_img, self.input_tensor = self.construir_input()
        self.input_tensor = self.input_tensor.to(self.device)

        # predecir clase con el modelo ya entrenado
        self.clase_predicha = self.predecir_clase()
        print(f"Clase predicha: {self.clase_predicha}")

        # generar gradcam
        self.visualizacion = self.generar_gradcam()


    def seleccionar_imagen_aleatoria(self, carpeta_img):
        "Selecciona una imagen de ejemplo válida para probar gradCam"
        lista_img = os.listdir(carpeta_img)
        # seleccionar aleatoriamente una imagen de esa carpeta
        img = os.path.join(carpeta_img, random.choice(lista_img))    
        return img
    
    def construir_input(self):
        """Mismas transformaciones que en los modelos de ViT que trabajé"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        pil_img = Image.open(self.img_path).convert("RGB")
        rgb_img = np.array(pil_img.resize((224, 224)), dtype=np.float32) / 255.0
        input_tensor = transform(pil_img).unsqueeze(0)
        return rgb_img, input_tensor
    
    def reshape_transform(self, tensor):
        print(tensor.shape)
        result = tensor.permute(0, 3, 1, 2)
        return result

    def cargar_modelo(self):
        modelo = timm.create_model("swin_small_patch4_window7_224.ms_in22k_ft_in1k",
                                   pretrained=False, num_classes=9)
        print(modelo)
        # cargar pesos
        state_dict = torch.load(self.pth_path, map_location=self.device)
        modelo.load_state_dict(state_dict)
        modelo = modelo.to(self.device)

        print("Modelo cargado correctamente")
        return modelo

    def predecir_clase(self):
        output = self.modelo(self.input_tensor)
        pred = output.argmax(dim=1).item()

        return pred

    def generar_gradcam(self):
        # Capa objetivo
        target_layers = [self.modelo.layers[-1].blocks[-1].norm1]

        # objetivo
        targets = [ClassifierOutputTarget(self.clase_predicha)]

        # gradcam en grises
        cam = GradCAM(
            model=self.modelo,
            target_layers=target_layers,
            reshape_transform=self.reshape_transform)

        grayscale_cam = cam(
            input_tensor=self.input_tensor,
            targets=targets
        )

        grayscale_cam = grayscale_cam[0]

        # visualización
        visualizacion = show_cam_on_image(self.rgb_img, grayscale_cam, use_rgb=True)

        return visualizacion

    def mostrar_comparacion(self):
        # Visualizacion comparativa
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))

        axes[0].imshow(self.rgb_img)
        axes[0].set_title("Imagen original")
        axes[0].axis("off")

        axes[1].imshow(self.visualizacion)
        axes[1].set_title("ViT: Grad-CAM")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

class ExplicaProcesoGradCAM:
    def __init__(self, carpeta, pth_path):
        self.carpeta = carpeta
        self.pth_path = pth_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")

        # Cargar modelo
        self.modelo = self.cargar_modelo()

        # Seleccionar imagen
        random.seed(10)
        self.img_path = self.seleccionar_imagen_aleatoria(self.carpeta)
        print(f"Imagen usada: {self.img_path}")

        # Construir input
        self.rgb_img, self.input_tensor = self.construir_input()
        self.input_tensor = self.input_tensor.to(self.device)

        # predecir clase con el modelo ya entrenado
        self.clase_predicha = self.predecir_clase()
        print(f"Clase predicha: {self.clase_predicha}")

        # generar gradcam
        self.visualizacion = self.generar_gradcam()


    def seleccionar_imagen_aleatoria(self, carpeta_img):
        "Selecciona una imagen de ejemplo válida para probar gradCam"
        lista_img = os.listdir(carpeta_img)
        # seleccionar aleatoriamente una imagen de esa carpeta
        img = os.path.join(carpeta_img, random.choice(lista_img))    
        return img
    
    def construir_input(self):
        """Mismas transformaciones que en los modelos de ViT que trabajé"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        pil_img = Image.open(self.img_path).convert("RGB")
        rgb_img = np.array(pil_img.resize((224, 224)), dtype=np.float32) / 255.0
        input_tensor = transform(pil_img).unsqueeze(0)
        return rgb_img, input_tensor
    
    def reshape_transform(self, tensor):
        print(tensor.shape)
        result = tensor.permute(0, 3, 1, 2)
        return result

    def cargar_modelo(self):
        modelo = timm.create_model("swin_small_patch4_window7_224.ms_in22k_ft_in1k",
                                   pretrained=False, num_classes=9)
        print(modelo)
        # cargar pesos
        state_dict = torch.load(self.pth_path, map_location=self.device)
        modelo.load_state_dict(state_dict)
        modelo = modelo.to(self.device)

        print("Modelo cargado correctamente")
        return modelo

    def predecir_clase(self):
        output = self.modelo(self.input_tensor)
        pred = output.argmax(dim=1).item()

        return pred

    def generar_gradcam(self):
        # Capas objetivo
        capas = {
        "Stage 1\n(56x56)": self.modelo.layers[0].blocks[-1].norm1,
        "Stage 2\n(28x28)": self.modelo.layers[1].blocks[-1].norm1,
        "Stage 3\n(14x14)": self.modelo.layers[2].blocks[-1].norm1,
        "Stage 4\n(7x7)": self.modelo.layers[3].blocks[-1].norm1,
        }

        # predicción del modelo
        with torch.no_grad():

            output = self.modelo(
                self.input_tensor.to(self.device)
            )

            clase_pred = output.argmax(dim=1).item()

        targets = [ClassifierOutputTarget(clase_pred)]

        # guardar visualizaciones
        self.visualizaciones = {}

        for nombre_capa, target_layer in capas.items():

            cam = GradCAM(
                model=self.modelo,
                target_layers=[target_layer],
                reshape_transform=self.reshape_transform
            )

            grayscale_cam = cam(
                input_tensor=self.input_tensor.to(self.device),
                targets=targets
            )[0, :]

            # visualización
            visualizacion = show_cam_on_image(self.rgb_img, grayscale_cam, use_rgb=True)

            self.visualizaciones[nombre_capa] = visualizacion

    def mostrar_comparacion(self):
        # Visualizacion comparativa de las capas
        n = len(self.visualizaciones) + 1
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))

        axes[0].imshow(self.rgb_img)
        axes[0].set_title("Imagen original")
        axes[0].axis("off")

        for i , (nombre, visualizacion) in enumerate(self.visualizaciones.items()):
            axes[i+1].imshow(visualizacion)
            axes[i+1].set_title(nombre)
            axes[i+1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__=="__main__":
    carpeta_img = os.path.join("dev_model", "fairface_reorganizado", "Imagenes_prueba")
    pth_path = os.path.join("dev_model", "swin_small_patch4_window7_224.ms_in22k_ft_in1k.pth")

    explicador = ExplicaGradCAM(carpeta_img, pth_path)
    explicador.mostrar_comparacion()

    explicador_proceso = ExplicaProcesoGradCAM(carpeta_img, pth_path)
    explicador_proceso.mostrar_comparacion()
