BioClinicalBERT Multilabel API
Este repositorio contiene una solución al desafío de clasificación multilabel de abstracts clínicos utilizando BioClinicalBERT fine-tuned.
La solución incluye:

Entrenamiento y evaluación del modelo
API implementada con FastAPI
Script de evaluación a partir de archivos CSV

Descripción del Proyecto
Modelos tradicionales (baseline)
Se ensayaron enfoques clásicos con TF-IDF y clasificadores lineales (SVM, Logistic Regression, Ridge, Naive Bayes).
Obtuvieron métricas sólidas, pero con limitaciones semánticas al tratar terminología médica.
Problemas: incapacidad para entender sinónimos y contexto clínico.
Migración a BERT
Se utilizó BioClinicalBERT (emilyalsentzer/Bio_ClinicalBERT), un modelo pre-entrenado en texto clínico.
Se entrenó en Google Colab debido a la demanda computacional.
Permite capturar relaciones semánticas complejas y sinónimos médicos.
Despliegue
El modelo se empaquetó en una API con FastAPI.
Se desplegó en Hugging Face Spaces para facilitar su consumo y evitar problemas con archivos locales pesados.
Descarga del Modelo
Los pesos del modelo entrenado están alojados en Hugging Face Hub debido a las limitaciones de tamaño de archivo. Puedes descargar y usar el modelo de las siguientes maneras:
Opción 1: Descarga directa con Transformers (Recomendado)
pythonfrom transformers import AutoModel, AutoTokenizer, AutoConfig

# Descargar y cargar el modelo directamente desde Hugging Face
model = AutoModel.from_pretrained("siderbrand/bioclinicalbert-finetuned")
tokenizer = AutoTokenizer.from_pretrained("siderbrand/bioclinicalbert-finetuned")
config = AutoConfig.from_pretrained("siderbrand/bioclinicalbert-finetuned")

# El modelo se almacenará automáticamente en caché local
Opción 2: Descarga a directorio local
pythonfrom transformers import AutoModel, AutoTokenizer

# Descargar y guardar en directorio local
model = AutoModel.from_pretrained("siderbrand/bioclinicalbert-finetuned")
tokenizer = AutoTokenizer.from_pretrained("siderbrand/bioclinicalbert-finetuned")

# Guardar localmente
model.save_pretrained("./models/bioclinicalbert_finetuned_final/")
tokenizer.save_pretrained("./models/bioclinicalbert_finetuned_final/")
Opción 3: Script de configuración automática
Crea y ejecuta este script:
python# setup_model.py
import os
from transformers import AutoModel, AutoTokenizer

def download_model():
    """Descargar el modelo fine-tuned desde Hugging Face"""
    
    model_name = "siderbrand/bioclinicalbert-finetuned"
    local_path = "./models/bioclinicalbert_finetuned_final/"
    
    print("Descargando modelo desde Hugging Face...")
    
    # Crear directorio si no existe
    os.makedirs(local_path, exist_ok=True)
    
    # Descargar modelo y tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Guardar localmente
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)
    
    print(f"Modelo descargado exitosamente en {local_path}")

if __name__ == "__main__":
    download_model()
Luego ejecutar:
bashpython setup_model.py
Estructura del Repositorio
bioclinicalbert-multilabel/
│── app.py                      # API con FastAPI
│── evaluate.py                 # Script de evaluación en CSV
│── requirements.txt            # Dependencias del proyecto
│── setup_model.py              # Script para descargar modelo
│── models/
│   └── bioclinicalbert_finetuned_final/
│       ├── config.json
│       ├── tokenizer.json
│       ├── vocab.txt
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       └── best_thresholds.json
│── README.md
Instalación
Se recomienda crear un entorno virtual:
bashpython -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

pip install -r requirements.txt
Ejecución de la API Local
bashuvicorn app:app --reload --port 8000
Endpoints disponibles:

/ → mensaje de bienvenida
/health → estado de la API
/predict → recibe texto(s) y devuelve predicciones

Ejemplo de request:
bashcurl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"texts": ["Patient presents with chest pain and elevated troponin levels."]}'
Ejemplo de respuesta:
json[
  {
    "input": "Patient presents with chest pain and elevated troponin levels.",
    "proba": {
      "cardio": 0.91,
      "hepato": 0.07,
      "neuro": 0.04,
      "onco": 0.02
    },
    "labels_short": ["cardio"]
  }
]
Evaluación con CSV
Uso del Script evaluate.py
bashpython evaluate.py input.csv
(dentro del entorno requerido)
Formato del CSV de Entrada
Columnas requeridas:

title - Título del documento médico
abstract - Resumen/abstract del documento
group - Etiqueta verdadera

Valores válidos para group:

cardiovascular (cardio)
hepatorenal (hepato)
neurological (neuro)
oncological (onco)

Ejemplo de CSV:
csvtitle,abstract,group
"Heart attack case","Patient presents with chest pain and elevated troponin levels.","cardiovascular"
"Liver failure case","Signs of cirrhosis and hepatorenal syndrome.","hepatorenal"
"Stroke evaluation","Acute onset neurological deficits with imaging findings.","neurological"
"Cancer screening","Abnormal tumor markers requiring further oncological workup.","oncological"
Salida Esperada

Archivo generado: predictions.csv con columna group_predicted
Métricas en consola:

F1-SCORE PONDERADO: 0.8732

REPORTE DE CLASIFICACIÓN:
                precision    recall  f1-score   support

cardiovascular      0.938     0.938     0.938       128
  hepatorenal       0.924     0.926     0.925       121  
  neurological      0.920     0.920     0.920       125
    oncological      0.922     0.924     0.923       129

     accuracy                           0.927       503
    macro avg       0.926     0.927     0.926       503
 weighted avg       0.927     0.927     0.927       503

MATRIZ DE CONFUSIÓN:
                cardiovascular hepatorenal neurological oncological
cardiovascular             120           2            5           1
hepatorenal                  3         112            4           2  
neurological                 2           5          115           3
oncological                  1           3            6         119
Métrica Principal
F1-Score Ponderado (Weighted F1)

Considera el desbalance entre clases
Refleja mejor el rendimiento global en contextos multilabel
Promedia F1 de cada clase pesado por su soporte

Métricas adicionales:

Precision/Recall/F1 por clase individual
Matriz de confusión para análisis detallado de errores
Accuracy global del sistema

Nota Importante sobre Evaluación
Evitar datasets demasiado simples: Si tu CSV tiene solo 1 ejemplo por clase con palabras clave muy obvias (ej: "heart" → cardiovascular), obtendrás scores perfectos (1.000) que no reflejan el rendimiento real.
Para evaluación realista: Usa datasets con ≥20-50 muestras por clase y casos que no tengan palabras clave tan directas.
Enlaces

Modelo en Hugging Face: siderbrand/bioclinicalbert-finetuned
API en Hugging Face Spaces: siderbrand-biobert1
BioClinicalBERT Original: emilyalsentzer/Bio_ClinicalBERT

Requisitos
bashpip install transformers torch datasets scikit-learn pandas numpy
Inicio Rápido
pythonfrom transformers import AutoModel, AutoTokenizer
import torch

# Cargar el modelo
model = AutoModel.from_pretrained("siderbrand/bioclinicalbert-finetuned")
tokenizer = AutoTokenizer.from_pretrained("siderbrand/bioclinicalbert-finetuned")

# Ejemplo de uso
text = "Patient presents with chest pain and shortness of breath"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    # Procesar outputs para tu tarea específica
Nota
Los pesos del modelo (pytorch_model.bin) no están incluidos en este repositorio debido a las limitaciones de tamaño de archivo de GitHub. Deben descargarse desde Hugging Face Hub usando uno de los métodos descritos arriba.