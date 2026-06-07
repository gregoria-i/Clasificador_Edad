# Frontend generado por IA (Vibe Coding)

La carpeta frontend contiene una interfaz básica diseñada para consumir la API de clasificación de edad.

Autoras: Andrea Itzel Gregorio Martínez y Hannah Sarahi García Salas

## Índice

- [Resumen](#resumen)
- [Características](#características)
- [Prompt utilizado](#prompt)
- [Resultados](#resultados)
- [Descripción de contenido](#contenido)

## Resumen
El frontend fue generado utilizando la metodología Vibe Coding, es decir, desarrollado mediante instrucciones en lenguaje natural dirigidas a una IA sin escribir el código manualmente en su totalidad.

## Características
Vanilla Web: Creado exclusivamente con HTML5, CSS3 y JavaScript puro.
Sin Frameworks: No utiliza React, Vue, Angular, Bootstrap ni librerías externas (cero red tape).
Responsive: Diseño adaptable a dispositivos móviles y escritorio.
Integración API: Configurados para usar fetch y FormData apuntando al backend en Python (FastAPI).

## Prompt utilizado
El siguiente fue la directriz principal que la IA utilizó para maquetar la estructura completa (HTML, CSS, JS):

```Quiero que generes un frontend basico, sin frameworks, para consumir una API de clasificacion de edad. Debe estar hecho solo con HTML, CSS y JavaScript vanilla.

Contexto de la API:

* Base URL configurable en una constante llamada API_URL, por defecto http://127.0.0.1:8000
* Endpoint GET /health para verificar estado
* Endpoint POST /predict que recibe multipart/form-data con el campo file (imagen)

Si todo sale bien, /predict responde JSON con esta forma aproximada:

* prediction: { class_name, probability }
* second_prediction: { class_name, probability }
* top5: arreglo de 5 elementos con { class_name, probability }
* recommendation: {
  title,
  summary,
  prediction: {
  primary_age_range,
  primary_confidence,
  secondary_age_range,
  secondary_confidence
  },
  demographic_profile: {
  segment_name,
  summary,
  age_transition_note
  },
  insights: {
  education_and_learning: [],
  career_and_development: [],
  technology_and_media: [],
  consumer_interests: [],
  service_preferences: []
  }
  }
* Si hay error, responde con detail.

Requisitos de interfaz:

* Titulo y descripcion corta.
* Boton para probar conexion con /health y mostrar resultado.
* Input para subir imagen, con vista previa.
* Boton "Analizar imagen" para llamar /predict.
* Estado de carga mientras procesa.
* Manejo de errores legible para el usuario.
* Mostrar resultado en tarjetas o secciones visualmente separadas:

  * Predicción principal (rango de edad y probabilidad en porcentaje).
  * Segunda predicción.
  * Top 5 predicciones con barras de progreso.
  * recommendation.title.
  * recommendation.summary.
  * prediction (primera y segunda predicción con sus probabilidades).
  * demographic_profile.
  * insights.

Diseño:
* Limpio, moderno y responsive.
* Sin librerías externas.
* Estilo tipo dashboard profesional de inteligencia artificial.
* Uso de tarjetas (cards), bordes redondeados, sombras suaves y buena jerarquía visual.
* Mucho espacio en blanco y apariencia similar a herramientas modernas de machine learning.
* Que parezca una pequeña herramienta profesional.
* Paleta sobria basada en violeta, blanco y grises claros.

Requisitos tecnicos:

* Entrega 3 archivos separados: index.html, styles.css y model.js.
* Usa fetch + FormData.
* Valida que el archivo exista y sea imagen antes de enviar.
* Convierte probabilidades a porcentaje con 2 decimales.
* Comenta brevemente las partes importantes del JS.
* No uses React, Vue, Bootstrap ni dependencias externas.

Entrega final:
Devuelveme el contenido completo de los 3 archivos, en secciones separadas y claramente rotuladas.```
