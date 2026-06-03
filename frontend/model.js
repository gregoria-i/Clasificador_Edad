// Configuración global de la API
const API_URL = 'http://127.0.0.1:8000';

// Elementos del DOM
const btnHealth = document.getElementById('btn-health');
const healthStatus = document.getElementById('health-status');
const imageInput = document.getElementById('image-input');
const imagePreview = document.getElementById('image-preview');
const previewContainer = document.getElementById('preview-container');
const btnSubmit = document.getElementById('btn-submit');
const uploadForm = document.getElementById('upload-form');
const loader = document.getElementById('loader');
const errorMessage = document.getElementById('error-message');
const resultsContainer = document.getElementById('results-container');

// --- MANEJO DEL ESTADO DE LA API (/health) ---
btnHealth.addEventListener('click', async () => {
    healthStatus.textContent = 'Verificando...';
    healthStatus.className = 'status-badge';
    
    try {
        const response = await fetch(`${API_URL}/health`);
        if (!response.ok) throw new Error();
        
        healthStatus.textContent = 'En línea';
        healthStatus.className = 'status-badge online';
    } catch (error) {
        healthStatus.textContent = 'Fuera de línea';
        healthStatus.className = 'status-badge offline';
    }
});

// --- MANEJO DE VISTA PREVIA Y VALIDACIÓN DE IMAGEN ---
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    
    // Validar que exista archivo y sea una imagen
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        
        reader.onload = (event) => {
            imagePreview.src = event.target.result;
            previewContainer.classList.remove('hidden');
            btnSubmit.disabled = false; // Habilitar envío si es válido
            clearError();
        };
        
        reader.readAsDataURL(file);
    } else {
        showError('Por favor, selecciona un archivo de imagen válido (.jpg, .png, etc.)');
        resetUpload();
    }
});

// --- ENVÍO DEL FORMULARIO Y CONSUMO DE /PREDICT ---
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const file = imageInput.files[0];
    if (!file) return;

    // Preparar FormData
    const formData = new FormData();
    formData.append('file', file);

    // Ajustar UI para estado de carga
    setLoading(true);
    clearError();
    resultsContainer.classList.add('hidden');

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            // Capturar el campo "detail" mandado por el backend o fallback genérico
            throw new Error(data.detail || 'Ocurrió un error al procesar la imagen.');
        }

        // Renderizar los resultados si todo fue exitoso
        renderResults(data);

    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
});

// --- FUNCIONES AUXILIARES Y RENDERIZADO ---

// Transforma valores decimales (0.856) a formato porcentaje (85.60%)
function formatPercentage(value) {
    // Si la API ya devuelve de 0-100 en vez de 0-1, remover el "* 100"
    const parsedValue = value <= 1 ? value * 100 : value;
    return `${parsedValue.toFixed(2)}%`;
}

// Renderiza listas genéricas para la sección de insights/recommendations
function fillList(elementId, itemsArray) {
    const ul = document.getElementById(elementId);
    ul.innerHTML = ''; // Limpiar previo
    if (itemsArray && itemsArray.length > 0) {
        itemsArray.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            ul.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = 'No hay datos disponibles';
        li.style.color = '#6b7280';
        ul.appendChild(li);
    }
}

function renderResults(data) {
    // 1. Predicción Principal
    document.getElementById('main-class').textContent = data.top1.class_name;
    document.getElementById('main-prob').textContent = formatPercentage(data.top1.probability);

    // 2. Lista Top 5 (Top K)
    const topkList = document.getElementById('topk-list');
    topkList.innerHTML = '';
    data.topk.forEach(item => {
        const li = document.createElement('li');
        li.innerHTML = `<span>${item.class_name}</span> <strong>${formatPercentage(item.probability)}</strong>`;
        topkList.appendChild(li);
    });

    // 3. Bloque de recomendaciones principal y predicción estructurada
    const rec = data.recommendation;
    document.getElementById('rec-title').textContent = rec.title;
    document.getElementById('rec-summary').textContent = rec.summary;

    document.getElementById('pred-primary-range').textContent = rec.prediction.primary_age_range;
    document.getElementById('pred-primary-conf').textContent = formatPercentage(rec.prediction.primary_confidence);
    document.getElementById('pred-secondary-range').textContent = rec.prediction.secondary_age_range || 'N/A';
    document.getElementById('pred-secondary-conf').textContent = rec.prediction.secondary_confidence ? formatPercentage(rec.prediction.secondary_confidence) : '0.00%';

    // 4. Perfil Demográfico
    document.getElementById('demo-segment').textContent = rec.demographic_profile.segment_name;
    document.getElementById('demo-summary').textContent = rec.demographic_profile.summary;
    document.getElementById('demo-note').textContent = rec.demographic_profile.age_transition_note || 'Ninguna';

    // 5. Inyección de listas dinámicas (Insights)
    fillList('insight-education', rec.insights.education_and_learning);
    fillList('insight-career', rec.insights.career_and_development);
    fillList('insight-tech', rec.insights.technology_and_media);
    fillList('insight-consumer', rec.insights.consumer_interests);
    fillList('insight-services', rec.insights.service_preferences);

    // 6. Inyección de listas dinámicas (Recommendations)
    fillList('rec-priority', rec.recommendations.priority_actions);
    fillList('rec-services', rec.recommendations.suggested_services);
    fillList('rec-channels', rec.recommendations.communication_channels);

    // 7. Metadata técnica final
    document.getElementById('meta-info').textContent = `Archivo: ${data.filename} | Dispositivo: ${data.device} | Modelo: ${data.model_path}`;

    // Mostrar el contenedor global
    resultsContainer.classList.remove('hidden');
}

// Controladores de UI básicos
function setLoading(isLoading) {
    if (isLoading) {
        loader.classList.remove('hidden');
        btnSubmit.disabled = true;
    } else {
        loader.classList.add('hidden');
        btnSubmit.disabled = false;
    }
}

function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove('hidden');
}

function clearError() {
    errorMessage.textContent = '';
    errorMessage.classList.add('hidden');
}

function resetUpload() {
    imageInput.value = '';
    previewContainer.classList.add('hidden');
    imagePreview.src = '#';
    btnSubmit.disabled = true;
}