🩺 AI Pediatric Pneumonia Triage System (ResNet18 + Grad-CAM)
Clasificación automática de radiografías de tórax mediante Deep Learning e Inteligencia Artificial Explicable (XAI).

📌 Resumen Ejecutivo
Este repositorio contiene el desarrollo de un sistema de triaje clínico diseñado para detectar neumonía en radiografías de tórax pediátricas (pacientes de 1 a 5 años). El objetivo de la herramienta es asistir a los médicos de urgencias en la priorización de pacientes, minimizando el riesgo de pasar por alto casos positivos (Falsos Negativos).

Para garantizar la transparencia algorítmica, el modelo no opera como una "caja negra", sino que integra Grad-CAM (Gradient-weighted Class Activation Mapping) para generar mapas de calor que justifican visualmente cada diagnóstico clínico.

🛠️ Stack Tecnológico
Lenguaje: Python 3

Deep Learning Framework: PyTorch & Torchvision

Arquitectura: ResNet18 (Transfer Learning)

XAI & Visión por Computador: Grad-CAM, OpenCV, Scikit-Learn

Análisis de Datos: NumPy, Matplotlib, Seaborn

📊 El Dataset
Se utilizó el dataset oficial del Guangzhou Women and Children's Medical Center. Consta de 5,856 radiografías de tórax (rayos X) validadas de forma independiente por dos médicos especialistas antes de ser incluidas en la base de datos para el entrenamiento de la IA.

🔬 Metodología de Desarrollo
1. Pre-procesamiento de Imágenes
Se implementó un pipeline de transformación matemática para estandarizar la entrada a la red neuronal:

Redimensionado a 224x224 píxeles.

Conversión a Tensores de PyTorch.

Normalización basada en los estándares de ImageNet (Media: [0.485, 0.456, 0.406], Desviación Estándar: [0.229, 0.224, 0.225]).

2. Arquitectura y Entrenamiento (Transfer Learning)
Se optó por ResNet18 pre-entrenada para aprovechar su capacidad de extracción de características semánticas (bordes, texturas pulmonares).

Se congelaron las capas base (requires_grad = False).

Se modificó la capa de salida (modelo.fc) para una clasificación binaria: Sano (0) vs. Neumonía (1).

Se utilizó la función de pérdida CrossEntropyLoss y el optimizador Adam.

Se implementó Early Stopping (paciencia = 4 épocas) monitorizando la pérdida de validación para evitar el sobreajuste (overfitting).

3. Resultados y Auditoría Clínica
El modelo fue evaluado en un Test Set ciego de 624 pacientes, obteniendo métricas de grado médico enfocadas en la seguridad del paciente:

Sensibilidad (Recall): 97% (Métrica clave en triaje para evitar Falsos Negativos).

Precisión (Precision): 83% (El modelo asume un sesgo conservador, preferiendo Falsos Positivos a Falsos Negativos).

Exactitud Global (Accuracy): 86%

Falsos Negativos Críticos: 12 sobre 390 casos positivos reales.

4. Inteligencia Artificial Explicable (XAI)
Para auditar los 12 Falsos Negativos, se implementó el algoritmo Grad-CAM extrayendo los gradientes de la última capa convolucional (layer4[-1]). El análisis visual reveló un sesgo algorítmico: en ocasiones, el modelo ignoraba la opacidad pulmonar difusa y anclaba su decisión en artefactos localizados en los ángulos costofrénicos, diagnosticando "la esquina" en lugar del pulmón.

🚀 Próximos Pasos (V2)
Implementación de Data Augmentation avanzado (Random Crops centrados) para mitigar el sesgo espacial detectado por Grad-CAM.

Desarrollo de un pipeline de inferencia ("1-click script") para simular un entorno de producción hospitalario.

📚 Bibliografía
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. En Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778). https://doi.org/10.1109/CVPR.2016.90

Mooney, P. (2018). Chest X-Ray Images (Pneumonia). Kaggle. https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. En Proceedings of the IEEE international conference on computer vision (pp. 618-626). https://doi.org/10.1109/ICCV.2017.74
