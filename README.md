<div align="center">

# 🩺 AI Pediatric Pneumonia Triage System
### Clasificación automática de radiografías mediante ResNet18 y XAI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
</p>

</div>

---

## 📌 Resumen Ejecutivo
Este repositorio contiene el desarrollo de un sistema de triaje clínico diseñado para detectar neumonía en **radiografías de tórax pediátricas (pacientes de 1 a 5 años)**. El objetivo de la herramienta es asistir a los médicos de urgencias en la priorización de pacientes, minimizando el riesgo de pasar por alto casos positivos (Falsos Negativos). 

Para garantizar la transparencia algorítmica, el modelo no opera como una "caja negra", sino que integra **Grad-CAM** (Gradient-weighted Class Activation Mapping) para generar mapas de calor que justifican visualmente cada diagnóstico clínico.

## 🔬 Metodología de Desarrollo

### 1. Pre-procesamiento de Imágenes
Se implementó un pipeline de transformación matemática para estandarizar la entrada a la red neuronal:
* Redimensionado a **224x224 píxeles**.
* Conversión a Tensores de PyTorch.
* **Normalización** basada en los estándares de ImageNet (Media: `[0.485, 0.456, 0.406]`, Desviación Estándar: `[0.229, 0.224, 0.225]`).

### 2. Arquitectura y Entrenamiento (Transfer Learning)
Se optó por **ResNet18** pre-entrenada para aprovechar su capacidad de extracción de características semánticas.
* Se congelaron las capas base (`requires_grad = False`).
* Se modificó la capa de salida para una clasificación binaria: **Sano (0) vs. Neumonía (1)**.
* Se utilizó la función de pérdida `CrossEntropyLoss` y el optimizador `Adam`.
* Se implementó **Early Stopping** monitorizando la pérdida de validación para evitar el *overfitting*.

### 3. Resultados y Auditoría Clínica
El modelo fue evaluado en un *Test Set* ciego de 624 pacientes, obteniendo métricas de grado médico enfocadas en la seguridad:
* **Sensibilidad (Recall): 97%** (Métrica clave en triaje para evitar Falsos Negativos).
* **Precisión (Precision): 83%** (Sesgo conservador priorizando la seguridad).
* **Falsos Negativos Críticos:** 12 sobre 390 casos positivos reales.

### 4. Inteligencia Artificial Explicable (XAI)
Para auditar los 12 Falsos Negativos, se implementó el algoritmo Grad-CAM. El análisis visual reveló un sesgo algorítmico: el modelo ignoraba ocasionalmente la opacidad pulmonar difusa y anclaba su decisión en artefactos localizados en los ángulos costofrénicos, diagnosticando "la esquina" en lugar del pulmón. 

---

## 🚀 Próximos Pasos (V2)
- [ ] Implementación de *Data Augmentation* avanzado (Random Crops) para mitigar el sesgo espacial.
- [ ] Desarrollo de un pipeline de inferencia ("1-click script") para simular un entorno de producción hospitalario.

<br>

> **📚 Bibliografía Principal:**
> * He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. En *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778). https://doi.org/10.1109/CVPR.2016.90
> * Mooney, P. (2018). *Chest X-Ray Images (Pneumonia)*. Kaggle. https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Mooney, P. (2018). Chest X-Ray Images (Pneumonia). Kaggle. https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. En Proceedings of the IEEE international conference on computer vision (pp. 618-626). https://doi.org/10.1109/ICCV.2017.74
