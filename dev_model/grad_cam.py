"""
grad_cam.py
Script para generar mapas de calor a partir de una red neuronal entrenada
@author: Andrea Gregorio
@date: 2024-06
"""
# Librerías
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 1. Funciones auxiliares
def find_image_path() -> str:
    """Busca una imagen de ejemplo valida para ejecutar el notebook sin cambios manuales."""
    candidates = [
        "dev_model/fairface_reorganizado/Imagenes_entrenamiento/5.jpg",
        "dev_model/fairface_reorganizado/Imagenes_entrenamiento/20.jpg",
        "dev_model/fairface_reorganizado/Imagenes_entrenamiento/35.jpg",
        "dev_model/fairface_reorganizado/Imagenes_entrenamiento/48.jpg",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    data_dir = Path("dev_model/fairface_reorganizado/Imagenes_entrenamiento")
    if data_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            found = list(data_dir.rglob(ext))
            if found:
                return str(found[0])

    raise FileNotFoundError(
        "No se encontro una imagen valida. Ajusta image_path a una ruta existente."
    )


def build_input(image_path: str):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    pil_img = Image.open(image_path).convert("RGB")
    rgb_img = np.array(pil_img.resize((224, 224)), dtype=np.float32) / 255.0
    input_tensor = transform(pil_img).unsqueeze(0)
    return rgb_img, input_tensor


def reshape_transform_vit(tensor, height=14, width=14):
    """Convierte tokens del ViT [B, N, C] a formato espacial [B, C, H, W]."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)

# 2. Entrada y objetivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

image_path = find_image_path()
print(f"Imagen usada: {image_path}")
rgb_img, input_tensor = build_input(image_path)
input_tensor = input_tensor.to(device)

weights = ResNet50_Weights.DEFAULT
cnn_model = resnet50(weights=weights).to(device).eval()


with torch.no_grad():
    logits = cnn_model(input_tensor)
pred_class = int(torch.argmax(logits, dim=1).item())
targets = [ClassifierOutputTarget(pred_class)]
print(f"Clase objetivo: {pred_class}")

# 5. Aplicar Grad-CAM en un ViT
# Como en ViT no hay mapas convolucionales clásicos, usamos reshape_transform
# para convertir tokens de parches en una grilla espacial

vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(device).eval()
vit_target_layers = [vit_model.encoder.layers[-1].ln_1]

vit_gradcam = GradCAM(
    model=vit_model,
    target_layers=vit_target_layers,
    reshape_transform=reshape_transform_vit,
 )
cam_vit = vit_gradcam(input_tensor=input_tensor, targets=targets)[0, :]
vis_vit_gc = show_cam_on_image(rgb_img, cam_vit, use_rgb=True)

# Visualizacion comparativa
fig, axes = plt.subplots(1, 2, figsize=(20, 5))

axes[0].imshow(rgb_img)
axes[0].set_title("Imagen original")
axes[0].axis("off")

axes[1].imshow(vis_vit_gc)
axes[1].set_title("ViT: Grad-CAM")
axes[1].axis("off")

plt.tight_layout()
plt.show()