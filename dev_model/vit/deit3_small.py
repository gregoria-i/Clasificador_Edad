"""
deit3_small.py
Este script es para probar una red de Vision Transformer (ViT) en el dataset reorganizado de FairFace.
@author: Andrea Gregorio
@date: 2024-06
"""
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)  # Para poder importar preparar_dataset.py 

import json
import copy
import timm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.optim import AdamW
from preparar_dataset import CustomDataset


def aplicar_subset(dataset, fraction):
    """Para utilizar solo una fracción del dataset y hacer pruebas más rápidas"""
    if fraction >= 1.0:
        return dataset

    n = int(len(dataset) * fraction)

    indices = torch.randperm(len(dataset))[:n]

    return Subset(dataset, indices)

def crear_transform(mean_list = [0.485, 0.456, 0.406], std_list = [0.229, 0.224, 0.225], tam=224):
    """Para ajustar a los modelos de ViT, que esperan imágenes de 224x224 y normalizadas con estos valores"""
    return transforms.Compose([

        transforms.Resize((tam, tam)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.90, 1.0)),
        transforms.Normalize(
            mean=mean_list,
            std=std_list
        )
    ])

def crear_datasets(transform, subset_fraction):

    train_dataset = CustomDataset(csv_path=os.path.join("fairface_reorganizado", "entrenamiento.csv"),
                                  img_dir=os.path.join("fairface_reorganizado","Imagenes_entrenamiento"),
                                  transform=transform)

    val_dataset = CustomDataset(csv_path=os.path.join("fairface_reorganizado", "validacion.csv"),
                                img_dir=os.path.join("fairface_reorganizado","Imagenes_validacion"),
                                transform=transform)

    test_dataset = CustomDataset(csv_path=os.path.join("fairface_reorganizado", "prueba.csv"),
                                 img_dir=os.path.join("fairface_reorganizado", "Imagenes_prueba"),
                                 transform=transform)
    # Reducir el tamaño de los datasets para pruebas más rápidas
    train_dataset = aplicar_subset(train_dataset, subset_fraction)
    val_dataset = aplicar_subset(val_dataset, subset_fraction)
    test_dataset = aplicar_subset(test_dataset, subset_fraction)

    return train_dataset, val_dataset, test_dataset

def crear_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers):
    pin_memory = True if torch.cuda.is_available() else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader

def crear_modelo(model_name, num_classes, device):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    return model

def calcular_accuracy_one_off(y_true, y_pred):
    """Calcula la métrica One-Off Accuracy, que considera como correctas las predicciones que están a una clase de distancia de la verdadera"""
    # Aquí los vectores ya tienen los labels de 0 a 8 que indican las 9 clases.
    correct = 0
    total = len(y_true)

    for true, pred in zip(y_true, y_pred):
        if abs(true - pred) <= 1:  # o es idéntica o solo hay 1 de distancia
            correct += 1

    return correct / total

def evaluar(model, loader, criterion, device):
    model.eval()
    loss_total = 0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss_total += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    loss_promedio = loss_total / len(loader)
    acc = correct / total

    return {"loss": loss_promedio, "acc": acc, "y_true": y_true, "y_pred": y_pred}

def entrenar(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience, warmup_epochs):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float('inf')
    best_model = None
    early_stopping_counter = 0
    base_lr = optimizer.param_groups[0]["lr"]

    print("Inicia entrenamiento")
    for epoch in range(num_epochs):
        # Para ajustar el learning rate durante las primeras épocas (warmup)
        if epoch < warmup_epochs:
            warmup_lr = (base_lr * (epoch + 1) / warmup_epochs)

            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        val_metrics = evaluar(model, val_loader, criterion, device)

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]

        print(f"Época {epoch+1}/{num_epochs} - LR: {optimizer.param_groups[0]['lr']:.6f} - Train Loss: {train_loss:.4f}\
              - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0

        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("\nEarly stopping")
                break

    model.load_state_dict(best_model)

    return model, history

def guardar_json(save_path, model_name, history, test_acc, one_off, y_true, y_pred):
    """Guarda un json estándar para poder comparar con otros modelos"""
    resultados = {"model_name": model_name,
                  "train_loss": history["train_loss"],
                  "val_loss": history["val_loss"],
                  "train_acc": history["train_acc"],
                  "val_acc": history["val_acc"],
                  "test_acc": test_acc,
                  "one_off": one_off,
                  "y_true": [int(x) for x in y_true],
                  "y_pred": [int(x) for x in y_pred]}
    
    with open(save_path, "w") as f:
        json.dump(resultados, f, indent=4)


if __name__ == '__main__':
    model_name = "deit3_small_patch16_224.fb_in22k_ft_in1k"

    num_classes = 9
    batch_size = 32
    lr = 1e-4
    num_epochs = 25
    patience = 5
    warmup_epochs = 2
    subset_fraction = 0.2  #Para usar solo el 20% del dataset

    num_workers = 4

    # CPU O GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')

    # Crear transformacion para que sean compatibles con sus imágenes
    transform = crear_transform()

    train_dataset, val_dataset, test_dataset = crear_datasets( transform, subset_fraction )

    print("Datasets cargados")

    # Crear DataLoaders
    train_loader, val_loader, test_loader = crear_dataloaders( train_dataset, val_dataset, test_dataset, batch_size, num_workers )
    print("DataLoaders creados")

    model = crear_modelo(model_name, num_classes, device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr)

    model, history = entrenar(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience, warmup_epochs)

    # Evaluar el modelo en el conjunto de prueba y calcular métricas
    test_metrics = evaluar(model, test_loader, criterion, device)

    test_acc = test_metrics["acc"]
    print(f"\nTest Accuracy: {test_acc:.4f}")

    one_off = calcular_accuracy_one_off(test_metrics["y_true"], test_metrics["y_pred"])
    print(f"One-Off Accuracy: {one_off:.4f}")

    # Guardar resultados en JSON para comparar con otros modelos
    os.makedirs("resultados", exist_ok=True)
    save_path = os.path.join("resultados", f"{model_name}.json")

    guardar_json(save_path, model_name, history, test_acc, one_off, test_metrics["y_true"], test_metrics["y_pred"])
    print(f"Archivo {save_path} guardado")
