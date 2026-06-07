// URL base de la API (configurable)
const API_URL = "http://127.0.0.1:8000";

// Referencias a elementos del DOM
const healthCheckBtn = document.getElementById("healthCheckBtn");
const healthStatus = document.getElementById("healthStatus");
const healthMessage = document.getElementById("healthMessage");

const imageInput = document.getElementById("imageInput");
const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const analyzeBtn = document.getElementById("analyzeBtn");
const uploadError = document.getElementById("uploadError");

const loadingIndicator = document.getElementById("loadingIndicator");
const globalError = document.getElementById("globalError");

const resultsContainer = document.getElementById("resultsContainer");
const primaryPredictionEl = document.getElementById("primaryPrediction");
const secondaryPredictionEl = document.getElementById("secondaryPrediction");
const top5Container = document.getElementById("top5Container");

const recTitle = document.getElementById("recTitle");
const recSummary = document.getElementById("recSummary");
const recPrediction = document.getElementById("recPrediction");
const recDemographic = document.getElementById("recDemographic");
const recInsights = document.getElementById("recInsights");

// Utilidad: formatea probabilidad [0,1] a porcentaje con 2 decimales
function toPercent(prob) {
  if (typeof prob !== "number" || isNaN(prob)) return "—";
  return (prob * 100).toFixed(2) + "%";
}

// Utilidad: limpia mensajes de error
function clearErrors() {
  uploadError.textContent = "";
  globalError.textContent = "";
}

// Manejo de vista previa de imagen
imageInput.addEventListener("change", (event) => {
  clearErrors();
  const file = event.target.files[0];

  if (!file) {
    previewContainer.classList.add("hidden");
    imagePreview.src = "";
    return;
  }

  if (!file.type.startsWith("image/")) {
    uploadError.textContent = "El archivo seleccionado no es una imagen válida.";
    imageInput.value = "";
    previewContainer.classList.add("hidden");
    imagePreview.src = "";
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    previewContainer.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
});

// Probar conexión con /health
healthCheckBtn.addEventListener("click", async () => {
  clearErrors();
  healthStatus.classList.remove("hidden");
  healthMessage.textContent = "Verificando estado de la API...";
  healthMessage.classList.remove("health-ok", "health-fail");

  try {
    const response = await fetch(`${API_URL}/health`, {
      method: "GET"
    });

    if (!response.ok) {
      throw new Error(`Código de estado: ${response.status}`);
    }

    const data = await response.json().catch(() => ({}));
    healthMessage.textContent =
      "API en línea. Respuesta: " + JSON.stringify(data);
    healthMessage.classList.add("health-ok");
  } catch (error) {
    healthMessage.textContent =
      "No se pudo conectar con la API. Detalle: " + error.message;
    healthMessage.classList.add("health-fail");
  }
});

// Enviar imagen a /predict
analyzeBtn.addEventListener("click", async () => {
  clearErrors();

  const file = imageInput.files[0];
  if (!file) {
    uploadError.textContent = "Por favor, selecciona una imagen antes de analizar.";
    return;
  }

  if (!file.type.startsWith("image/")) {
    uploadError.textContent = "El archivo seleccionado no es una imagen válida.";
    return;
  }

  // Mostrar estado de carga
  loadingIndicator.classList.remove("hidden");
  resultsContainer.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      // Si la API devuelve error con campo detail
      const detail = data && data.detail ? data.detail : "Error desconocido.";
      throw new Error(detail);
    }

    // Renderizar resultados si todo va bien
    renderResults(data);
  } catch (error) {
    globalError.textContent =
      "Ocurrió un problema al procesar la imagen: " + error.message;
    resultsContainer.classList.add("hidden");
  } finally {
    loadingIndicator.classList.add("hidden");
  }
});

// Renderizado principal de resultados
function renderResults(data) {
  resultsContainer.classList.remove("hidden");

  renderPrimarySecondary(data);
  renderTop5(data.top5 || []);
  renderRecommendation(data.recommendation);
}

// Renderiza predicción principal y secundaria
function renderPrimarySecondary(data) {
  const prediction = data.prediction || {};
  const secondPrediction = data.second_prediction || {};

  // Predicción principal
  primaryPredictionEl.innerHTML = "";
  const mainAge =
    prediction.class_name ||
    (data.recommendation &&
      data.recommendation.prediction &&
      data.recommendation.prediction.primary_age_range) ||
    "Desconocido";

  const mainProb = prediction.probability;

  const mainAgeEl = document.createElement("p");
  mainAgeEl.className = "prediction-age-range";
  mainAgeEl.textContent = mainAge;

  const mainProbEl = document.createElement("p");
  mainProbEl.className = "prediction-prob";
  mainProbEl.textContent = "Probabilidad: " + toPercent(mainProb);

  primaryPredictionEl.appendChild(mainAgeEl);
  primaryPredictionEl.appendChild(mainProbEl);

  // Segunda predicción
  secondaryPredictionEl.innerHTML = "";
  const secondAge =
    secondPrediction.class_name ||
    (data.recommendation &&
      data.recommendation.prediction &&
      data.recommendation.prediction.secondary_age_range) ||
    "Desconocido";

  const secondProb = secondPrediction.probability;

  const secondAgeEl = document.createElement("p");
  secondAgeEl.className = "prediction-age-range";
  secondAgeEl.textContent = secondAge;

  const secondProbEl = document.createElement("p");
  secondProbEl.className = "prediction-prob";
  secondProbEl.textContent = "Probabilidad: " + toPercent(secondProb);

  secondaryPredictionEl.appendChild(secondAgeEl);
  secondaryPredictionEl.appendChild(secondProbEl);
}

// Renderiza top 5 predicciones con barras
function renderTop5(top5) {
  top5Container.innerHTML = "";

  if (!Array.isArray(top5) || top5.length === 0) {
    const p = document.createElement("p");
    p.className = "placeholder";
    p.textContent = "No se recibieron predicciones adicionales.";
    top5Container.appendChild(p);
    return;
  }

  top5.forEach((item, index) => {
    const className = item.class_name || `Clase ${index + 1}`;
    const prob = item.probability;

    const wrapper = document.createElement("div");
    wrapper.className = "top5-item";

    const header = document.createElement("div");
    header.className = "top5-header";

    const label = document.createElement("span");
    label.className = "top5-class";
    label.textContent = `${index + 1}. ${className}`;

    const probSpan = document.createElement("span");
    probSpan.className = "top5-prob";
    probSpan.textContent = toPercent(prob);

    header.appendChild(label);
    header.appendChild(probSpan);

    const bar = document.createElement("div");
    bar.className = "progress-bar";

    const fill = document.createElement("div");
    fill.className = "progress-fill";
    fill.style.width = `${Math.min(100, Math.max(0, prob * 100 || 0))}%`;

    bar.appendChild(fill);

    wrapper.appendChild(header);
    wrapper.appendChild(bar);

    top5Container.appendChild(wrapper);
  });
}

// Renderiza recommendation completa
function renderRecommendation(recommendation) {
  if (!recommendation) {
    recTitle.textContent = "Sin recomendación disponible";
    recSummary.textContent =
      "La API no devolvió información de recomendación para esta imagen.";
    recPrediction.innerHTML = '<p class="placeholder">Sin datos.</p>';
    recDemographic.innerHTML = '<p class="placeholder">Sin datos.</p>';
    recInsights.innerHTML = '<p class="placeholder">Sin datos.</p>';
    return;
  }

  // Título y resumen
  recTitle.textContent = recommendation.title || "Recomendación del modelo";
  recSummary.textContent =
    recommendation.summary ||
    "No se proporcionó un resumen detallado para esta predicción.";

  // Predicciones de rango de edad
  const pred = recommendation.prediction || {};
  recPrediction.innerHTML = "";

  const primaryLine = document.createElement("p");
  primaryLine.innerHTML =
    `<span class="rec-label">Primaria:</span> ` +
    `${pred.primary_age_range || "N/D"} ` +
    `<span class="rec-tag">${toPercent(pred.primary_confidence)}</span>`;

  const secondaryLine = document.createElement("p");
  secondaryLine.innerHTML =
    `<span class="rec-label">Secundaria:</span> ` +
    `${pred.secondary_age_range || "N/D"} ` +
    `<span class="rec-tag">${toPercent(pred.secondary_confidence)}</span>`;

  recPrediction.appendChild(primaryLine);
  recPrediction.appendChild(secondaryLine);

  // Perfil demográfico
  const demo = recommendation.demographic_profile || {};
  recDemographic.innerHTML = "";

  const segment = document.createElement("p");
  segment.innerHTML =
    `<span class="rec-label">Segmento:</span> ${demo.segment_name || "N/D"}`;

  const demoSummary = document.createElement("p");
  demoSummary.textContent = demo.summary || "Sin resumen de perfil demográfico.";

  const transition = document.createElement("p");
  transition.innerHTML =
    `<span class="rec-label">Transición de edad:</span> ` +
    `${demo.age_transition_note || "Sin información."}`;

  recDemographic.appendChild(segment);
  recDemographic.appendChild(demoSummary);
  recDemographic.appendChild(transition);

  // Insights
  const insights = recommendation.insights || {};
  recInsights.innerHTML = "";

  const insightSections = [
    { key: "education_and_learning", label: "Educación y aprendizaje" },
    { key: "career_and_development", label: "Carrera y desarrollo" },
    { key: "technology_and_media", label: "Tecnología y medios" },
    { key: "consumer_interests", label: "Intereses de consumo" },
    { key: "service_preferences", label: "Preferencias de servicios" }
  ];

  let hasAnyInsight = false;

  insightSections.forEach((section) => {
    const items = insights[section.key];
    if (!Array.isArray(items) || items.length === 0) return;

    hasAnyInsight = true;

    const group = document.createElement("div");
    group.className = "insight-group";

    const title = document.createElement("p");
    title.className = "insight-title";
    title.textContent = section.label;

    const list = document.createElement("ul");
    list.className = "insight-list";

    items.forEach((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      list.appendChild(li);
    });

    group.appendChild(title);
    group.appendChild(list);
    recInsights.appendChild(group);
  });

  if (!hasAnyInsight) {
    recInsights.innerHTML =
      '<p class="placeholder">No se proporcionaron insights adicionales.</p>';
  }
}
