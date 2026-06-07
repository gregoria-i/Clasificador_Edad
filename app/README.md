# API

Contiene el código fuente de la FastAPI que expone el modelo entrenado para realizar inferencias. 


## Índice
- [Objetivo](#objetivoapi)
- [Uso](#usoapi)
- [Flujo general](#flujo)
- [Endpoint predict](#endpoint)

## Contenido
### `age_api.py`

Archivo principal de la API.

Contiene:

* Configuración de FastAPI.
* Endpoint `GET /health` para verificar el estado de la aplicación.
* Endpoint `POST /predict` para recibir imágenes y generar predicciones.
* Integración con Gemini para generar el análisis demográfico.
* Configuración de CORS para permitir la comunicación con el frontend.

### `.env`

Archivo de variables de entorno.

Actualmente almacena:

* `GEMINI_API_KEY`, utilizada para autenticar las solicitudes a la API de Gemini.

Este archivo se encuentra incluido en `.gitignore`.

### `swin_inferencia_probabilidades.ipynb`
Este archivo fue generado por el profesor Eduardo Barrios para mostrarnos el flujo adecuado para lograr la inferencia de probabilidades a partir de nuestro modelo. 

Muestra la predicción y el top5 de las probabilidades más altas.

### `swin_inferencia_probabilidades.py`
Este archivo es una adaptación a .py del archivo generado por el profesor. Se realizó para lograr su ejecución y pruebas en IDEs que no ejecutan archivos .ipynb.

## Uso

### Iniciar la API

Desde la carpeta raíz del proyecto:

```bash
uvicorn app.age_api:app --reload
```

La aplicación estará disponible en:

```text
http://127.0.0.1:8000
```

### Verificar funcionamiento

```text
http://127.0.0.1:8000/health
```

Respuesta esperada:

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

### Documentación automática

FastAPI genera automáticamente una interfaz para probar los endpoints:

```text
http://127.0.0.1:8000/docs
```

## Flujo general

1. El usuario selecciona una imagen desde el frontend.
2. El frontend envía la imagen al endpoint `POST /predict`.
3. La API procesa la imagen utilizando el modelo Swin Transformer entrenado.
4. Se obtienen las predicciones más probables de rango de edad.
5. Gemini genera un análisis demográfico basado en las dos predicciones con mayor probabilidad.
6. La API devuelve la información al frontend en formato JSON.

## Endpoint `/predict`

La respuesta incluye:

* Predicción principal (`prediction`).
* Segunda predicción más probable (`second_prediction`).
* Lista Top-5 de predicciones (`top5`).
* Análisis demográfico generado por Gemini (`recommendation`).

