<div align="center">

# 🫁 Clasificación de Neumonía Pediátrica mediante Deep Learning
**Clasificación automática de radiografías mediante Deep Learning (ResNet18) y XAI.**

</div>

---

## 📌 Resumen del Proyecto
Este proyecto consiste en el entrenamiento de un modelo de red neuronal convolucional para la clasificación binaria de radiografías de tórax, distinguiendo entre pulmones sanos y pulmones con neumonía en pacientes pediátricos. 

Como parte de mi formación autónoma en inteligencia artificial aplicada a la ingeniería biomédica, el objetivo ha sido entender el flujo de trabajo completo: desde el preprocesamiento de los datos hasta la interpretación de las decisiones del modelo mediante mapas de activación (Grad-CAM).

Para garantizar la transparencia algorítmica, el modelo no opera como una "caja negra", sino que integra **Grad-CAM** (Gradient-weighted Class Activation Mapping) para generar mapas de calor que justifican visualmente cada diagnóstico clínico.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3
* **Deep Learning Framework:** PyTorch & Torchvision
* **Arquitectura:** ResNet18 (Transfer Learning)
* **XAI & Visión por Computador:** Grad-CAM, OpenCV, Scikit-Learn
* **Análisis de Datos:** NumPy, Matplotlib, Seaborn

## 🔬 Metodología de Desarrollo

### 1. Pre-procesamiento de Imágenes
Se implementó un pipeline de transformación matemática para estandarizar la entrada a la red neuronal:
* Redimensionado a **224x224 píxeles**.
* Conversión a Tensores de PyTorch.
* **Normalización** basada en los estándares de ImageNet (Media: `[0.485, 0.456, 0.406]`, Desviación Estándar: `[0.229, 0.224, 0.225]`).

### 2. Arquitectura y Entrenamiento (Transfer Learning)
Se optó por **ResNet18** pre-entrenada para aprovechar su capacidad de extracción de características semánticas (bordes, texturas pulmonares).
* Se congelaron las capas base (`requires_grad = False`).
* Se modificó la capa de salida para una clasificación binaria: **Sano (0) vs. Neumonía (1)**.
* Se utilizó la función de pérdida `CrossEntropyLoss` y el optimizador `Adam`.
* Se implementó **Early Stopping** (paciencia = 4 épocas) monitorizando la pérdida de validación para evitar el *overfitting*.

### 3. Resultados y Auditoría Clínica
El modelo fue evaluado en un *Test Set* ciego de 624 pacientes, obteniendo métricas de grado médico enfocadas en la seguridad del paciente:
* **Sensibilidad (Recall): 97%** (Métrica clave en triaje para evitar Falsos Negativos).
* **Precisión (Precision): 83%** (Sesgo conservador priorizando la seguridad).
* **Falsos Negativos Críticos:** 12 sobre 390 casos positivos reales.

### 4. Inteligencia Artificial Explicable (XAI)
Para auditar los 12 Falsos Negativos, se implementó el algoritmo Grad-CAM. El análisis visual reveló un sesgo algorítmico: el modelo ignoraba ocasionalmente la opacidad pulmonar difusa y anclaba su decisión en artefactos localizados en los ángulos costofrénicos, diagnosticando "la esquina" en lugar del pulmón. 

---

## 🚀 Próximos Pasos (V2)
* Implementación de *Data Augmentation* avanzado (Random Crops centrados) para mitigar el sesgo espacial detectado por Grad-CAM.
* Desarrollo de un pipeline de inferencia ("1-click script") para simular un entorno de producción hospitalario.

<br>
> **📚 Bibliografía Principal:**

* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. En *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778). https://doi.org/10.1109/CVPR.2016.90
  
* Mooney, P. (2018). *Chest X-Ray Images (Pneumonia)*. Kaggle. https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
  
* Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. En *Proceedings of the IEEE international conference on computer vision* (pp. 618-626). https://doi.org/10.1109/ICCV.2017.74
