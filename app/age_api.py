# age_api.py
import io
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import torch
import timm

from PIL import Image
from torchvision import transforms

# Para PROMPT y conexión con API de Gemini
from google import genai
import json

# Para leer el archivo privado .env con la llave de gemini
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))



MODEL_NAME = "swin_small_patch4_window7_224.ms_in22k_ft_in1k"
MODEL_PATH = Path("app/models/swin_small_patch4_window7_224.ms_in22k_ft_in1k.pth")
CLASS_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", ">70"]
GEMINI_API_KEY =os.getenv("GEMINI_API_KEY", "")

def mostrar_directorio_actual():
    try:
        # Método 1: usando os
        dir_os = os.getcwd()
        print(f"Directorio actual (os): {dir_os}")

        # Método 2: usando pathlib
        dir_pathlib = Path.cwd()
        print(f"Directorio actual (pathlib): {dir_pathlib}")

    except Exception as e:
        print(f"Error al obtener el directorio actual: {e}")
mostrar_directorio_actual()

def gemini_recommend_simple(top1: dict, top2: dict) -> dict:
    if not GEMINI_API_KEY: 
        return { 
            "title": "Análisis demográfico", 
            "summary": "GEMINI_API_KEY no configurada.", 
            "prediction": { 
                "primary_age_range": top1["class_name"], 
                "primary_confidence": f"{top1['probability']*100:.1f}%", 
                "secondary_age_range": top2["class_name"], 
                "secondary_confidence": f"{top2['probability']*100:.1f}%" }, 
                "demographic_profile": { 
                    "segment_name": "N/D", 
                    "summary": "N/D", 
                    "age_transition_note": "N/D" 
                    }, 
                "insights": {
                    "education_and_learning": [], 
                    "career_and_development": [], 
                    "technology_and_media": [], 
                    "consumer_interests": [], 
                    "service_preferences": [] 
                    } 
                }

    top1_str = top1['class_name']
    prob1_str = top1['probability']*100
    top2_str = top2['class_name']
    prob2_str = top2['probability']*100

    prompt = (
        f"El modelo clasificó una persona en los siguientes rangos de edad:\n"
    f"Edad principal:\n"
    f"Rango: {top1_str}\n"
    f"Confianza: {prob1_str:.1f}%\n"
    f"Edad secundaria:\n"
    f"Rango: {top2_str}\n"
    f"Confianza: {prob2_str:.1f}%\n"
    "Genera un análisis demográfico considerando ambas predicciones.\n"
    "IMPORTANTE:\n"
    "No afirmes características personales específicas.\n"
    "Habla únicamente de tendencias típicas del segmento demográfico.\n"
    "Si las edades son cercanas, menciona la posible transición entre grupos.\n"
    "Responde ÚNICAMENTE JSON válido(sin markdown, sin comentarios) con esta estructura exacta:\n"
    "{\n"
    "\"title\": \"...\",\n"
    "\"summary\": \"...\",\n"
    "\"prediction\": {\n"
    "\"primary_age_range\": \"...\",\n"
    "\"primary_confidence\": \"...\",\n"
    "\"secondary_age_range\": \"...\",\n"
    "\"secondary_confidence\": \"...\"\n"
    "},\n"
    "\"demographic_profile\": {\n"
    "\"segment_name\": \"...\",\n"
    "\"summary\": \"...\",\n"
    "\"age_transition_note\": \"...\"\n"
    "},\n"
    "\"insights\": {\n"
    "\"education_and_learning\": [],\n"
    "\"career_and_development\": [],\n"
    "\"technology_and_media\": [],\n"
    "\"consumer_interests\": [],\n"
    "\"service_preferences\": []\n"
    "}\n"
    "}\n"
    "Todo en español, conciso y claro para interfaz web.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = (response.text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        return {
            "title": "Análisis demográfico",
            "summary": text,
            "prediction": {
                "primary_age_range": top1["class_name"],
                "primary_confidence": f"{top1['probability']*100:.1f}%",
                "secondary_age_range": top2["class_name"],
                "secondary_confidence": f"{top2['probability']*100:.1f}%"
            },
            "demographic_profile": {
                "segment_name": "No disponible",
                "summary": text,
                "age_transition_note": ""
            },
            "insights": {
                "education_and_learning": [],
                "career_and_development": [],
                "technology_and_media": [],
                "consumer_interests": [],
                "service_preferences": []
            }
        }

def extraer_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.transform = build_transform()
        self.classes = CLASS_LABELS

    def load(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No existe el modelo: {MODEL_PATH}")

        print("Cargando modelo...")

        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)

        state_dict = extraer_state_dict(checkpoint)

        self.model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(self.classes))

        missing, unexpected = self.model.load_state_dict(state_dict,strict=False)

        if missing:
            print(f"Claves faltantes: {len(missing)}")

        if unexpected:
            print(f"Claves inesperadas: {len(unexpected)}")

        self.model.to(self.device)

        self.model.eval()

        print("Modelo cargado correctamente")

    @torch.no_grad()
    def predict(self, image_bytes):

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        logits = self.model(tensor)

        probs = torch.softmax(logits, dim=1).squeeze(0)

        top_probs, top_idx = torch.topk(probs,k=5)

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
    # Integra la respuesta de gemini al resultado
    result = {"prediction": predictions[0], "second_prediction": predictions[1], "top5": predictions}
    result["recommendation"] = gemini_recommend_simple(result["prediction"], result["second_prediction"])
    return result


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "age_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )