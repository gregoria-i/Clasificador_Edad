# age_api.py
import io
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import torch
import timm

from PIL import Image
from torchvision import transforms


MODEL_NAME = "swin_small_patch4_window7_224.ms_in22k_ft_in1k"
MODEL_PATH = Path("models/swin_small_patch4_window7_224.ms_in22k_ft_in1k.pth")
CLASS_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", ">70"]


def extraer_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net"
        ]:

            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint inválido")

    cleaned = {}

    for k, v in checkpoint.items():

        if isinstance(k, str):
            k = k.replace("module.", "")

        cleaned[k] = v

    return cleaned


def build_transform():

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


class AgeModelService:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.transform = build_transform()
        self.classes = CLASS_LABELS

    def load(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No existe el modelo: {MODEL_PATH}"
            )

        print("Cargando modelo...")

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=False
        )

        state_dict = extraer_state_dict(
            checkpoint
        )

        self.model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=len(self.classes)
        )

        missing, unexpected = self.model.load_state_dict(
            state_dict,
            strict=False
        )

        if missing:
            print(
                f"Claves faltantes: {len(missing)}"
            )

        if unexpected:
            print(
                f"Claves inesperadas: {len(unexpected)}"
            )

        self.model.to(self.device)

        self.model.eval()

        print("Modelo cargado correctamente")

    @torch.no_grad()
    def predict(self, image_bytes):

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        tensor = self.transform(
            image
        )

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        logits = self.model(
            tensor
        )

        probs = torch.softmax(
            logits,
            dim=1
        ).squeeze(0)

        top_probs, top_idx = torch.topk(
            probs,
            k=5
        )

        predictions = []

        for prob, idx in zip(
            top_probs,
            top_idx
        ):

            predictions.append({
                "class_name": self.classes[int(idx)],
                "probability": float(prob)
            })

        return predictions


# Empiezan las instancias y la API

service = AgeModelService()
service.load()  # con esto se carga el modelo del pth

app = FastAPI(title="Clasificador de Edad")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": service.model is not None, "device": str(service.device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Debes subir una imagen")

    image_bytes = await file.read()
    predictions = service.predict(image_bytes)

    return {"prediction": predictions[0], "second_prediction": predictions[1], "top5": predictions}


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "age_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )