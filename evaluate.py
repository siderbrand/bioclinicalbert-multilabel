import argparse
import pandas as pd
import torch
import torch.nn as nn
import json
from pathlib import Path
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel
import numpy as np
import sys

class BertForMultilabel(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BertForMultilabel, self).__init__()
        self.bert = bert_model
        hidden_size = bert_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

label_cols = ["cardiovascular", "hepatorenal", "neurological", "oncological"]

MODEL_DIR = Path("models/bioclinicalbert_finetuned_final")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {device}")

def check_model_files():
    """Verifica que todos los archivos necesarios existan"""
    required_files = [
        "config.json", 
        "pytorch_model.bin",
        "tokenizer_config.json", 
        "vocab.txt",
        "best_thresholds.json"
    ]
    
    if not MODEL_DIR.exists():
        print(f"ERROR: Directorio del modelo no existe: {MODEL_DIR}")
        return False
    
    missing = []
    for file in required_files:
        if not (MODEL_DIR / file).exists():
            missing.append(file)
    
    if missing:
        print(f"ERROR: Archivos faltantes en {MODEL_DIR}:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    print("Todos los archivos del modelo encontrados")
    return True

def load_csv(csv_path: str):
    """Carga y valida el CSV de entrada"""
    print(f"Cargando CSV: {csv_path}")
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")
    
    try:
        df = pd.read_csv(
            csv_path,
            sep=",",
            engine="python",
            quotechar='"',
            quoting=3,
            on_bad_lines="skip",
            encoding="utf-8"
        )
        print(f"CSV cargado: {len(df)} filas")
    except Exception as e:
        raise RuntimeError(f"Error leyendo {csv_path}: {e}")

    # Normalizar nombres de columnas
    original_cols = df.columns.tolist()
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"Columnas encontradas: {original_cols}")

    required_cols = {"title", "abstract", "group"}
    found_cols = set(df.columns)
    
    if not required_cols.issubset(found_cols):
        missing = required_cols - found_cols
        raise ValueError(f"Columnas faltantes: {missing}. Encontradas: {found_cols}")

    # Verificar datos no vacíos
    print(f"Grupos únicos en datos: {df['group'].unique()}")
    
    return df

def load_model():
    """Carga el modelo, tokenizer y umbrales"""
    print("Cargando modelo...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        bert_model = AutoModel.from_pretrained(MODEL_DIR)
        print("BERT y tokenizer cargados")
        
        model = BertForMultilabel(bert_model, num_labels=len(label_cols))
        
        # Cargar pesos del modelo
        model_path = MODEL_DIR / "pytorch_model.bin"
        print(f"Cargando pesos desde: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print("Modelo cargado y configurado")

        # Cargar umbrales
        thresholds_path = MODEL_DIR / "best_thresholds.json"
        with open(thresholds_path, "r", encoding="utf-8") as f:
            best_thresholds = json.load(f)
        print(f"Umbrales cargados: {best_thresholds}")

        return tokenizer, model, best_thresholds
        
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        raise

def predict_texts(texts, tokenizer, model, thresholds, batch_size=16):
    """Predice etiquetas para una lista de textos"""
    print(f"Prediciendo para {len(texts)} textos...")
    
    all_preds = []
    all_probs = []
    
    # Procesar en lotes para evitar problemas de memoria
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"   Procesando lote {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Tokenizar
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        # Inferencia
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Aplicar umbrales
        for prob in probs:
            labels = []
            for j, label in enumerate(label_cols):
                if prob[j] >= thresholds[label]:
                    labels.append(label)
            
            # Si no supera ningún umbral, asignar la clase con mayor probabilidad
            if not labels:
                max_idx = np.argmax(prob)
                labels = [label_cols[max_idx]]
            
            all_preds.append(labels)
            all_probs.append(prob)

    print("Predicciones completadas")
    return all_preds, np.array(all_probs)

def evaluate(csv_path):
    """Función principal de evaluación"""
    print("Iniciando evaluación...")
    
    # Verificar archivos del modelo
    if not check_model_files():
        print("No se puede continuar sin los archivos del modelo")
        sys.exit(1)
    
    # Cargar modelo
    try:
        tokenizer, model, thresholds = load_model()
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        sys.exit(1)
    
    # Cargar datos
    try:
        df = load_csv(csv_path)
    except Exception as e:
        print(f"Error cargando CSV: {e}")
        sys.exit(1)
    
    # Preparar textos
    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
    print(f"Preparados {len(texts)} textos para predicción")

    # Predecir
    try:
        preds, probs = predict_texts(texts, tokenizer, model, thresholds)
    except Exception as e:
        print(f"Error en predicción: {e}")
        sys.exit(1)

    # Procesar resultados (tomar primera etiqueta como predicción principal)
    df["group_predicted"] = [p[0] for p in preds]

    # Guardar CSV con predicciones
    out_path = Path("predictions.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Predicciones guardadas en: {out_path}")

    # Calcular métricas
    print("\n" + "="*50)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*50)
    
    y_true = df["group"].values
    y_pred = df["group_predicted"].values

    # F1 ponderado (métrica principal)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    print(f"F1-SCORE PONDERADO: {f1_weighted:.4f}")

    # Otras métricas de F1
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")

    # Reporte detallado
    print("REPORTE DE CLASIFICACIÓN:")
    print("-" * 50)
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    # Matriz de confusión
    print("MATRIZ DE CONFUSIÓN:")
    print("-" * 30)
    cm = confusion_matrix(y_true, y_pred, labels=label_cols)
    
    # Imprimir matriz formateada
    print(f"{'':>15}", end="")
    for label in label_cols:
        print(f"{label[:10]:>12}", end="")
    print()
    
    for i, label in enumerate(label_cols):
        print(f"{label[:10]:>15}", end="")
        for j in range(len(label_cols)):
            print(f"{cm[i][j]:>12}", end="")
        print()
    
    print(f"Evaluación completada. Total de muestras: {len(df)}")
    
    return f1_weighted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluar modelo BioClinicalBERT con un CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
    python evaluate_model.py data/test_data.csv
    
El CSV debe contener las columnas: title, abstract, group
        """
    )
    parser.add_argument("csv_path", type=str, help="Ruta al archivo CSV con columnas: title, abstract, group")
    args = parser.parse_args()

    try:
        f1_score = evaluate(args.csv_path)
        print(f" Proceso completado exitosamente. F1 final: {f1_score:.4f}")
    except KeyboardInterrupt:
        print("Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"Error inesperado: {e}")
        sys.exit(1)