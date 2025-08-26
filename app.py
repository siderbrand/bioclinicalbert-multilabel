from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

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


app = FastAPI(
    title="BioBERT API",
    description="Predicción multilabel con BioClinicalBERT fine-tuned",
    version="2.2"
)

# CORS abierto para frontend (V0)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


label_cols = ["cardiovascular", "hepatorenal", "neurological", "oncological"]
class_map = {
    "cardiovascular": "cardio",
    "hepatorenal": "hepato",
    "neurological": "neuro",
    "oncological": "onco"
}

MODEL_DIR = Path("models/bioclinicalbert_finetuned_final")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = None
model = None
best_thresholds = None

class PredictRequest(BaseModel):
    texts: List[str]

@app.get("/")
def home():
    return {"message": "API funcionando"}

@app.get("/health")
def health():
    return {"status": "ok"}  

@app.post("/predict")
def predict(request: PredictRequest):
    global tokenizer, model, best_thresholds

    
    if model is None or tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            bert_model = AutoModel.from_pretrained(MODEL_DIR)

            model_tmp = BertForMultilabel(bert_model, num_labels=len(label_cols))
            state_dict = torch.load(MODEL_DIR / "pytorch_model.bin", map_location=device)
            model_tmp.load_state_dict(state_dict)

            model_tmp = model_tmp.to(device)
            model_tmp.eval()
            model = model_tmp

            with open(MODEL_DIR / "best_thresholds.json", "r", encoding="utf-8") as f:
                best_thresholds = json.load(f)

            print("Modelo cargado en /predict")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error cargando modelo: {e}")

    # Inference
    results = []
    for text in request.texts:
        enc = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        proba = {class_map[label_cols[i]]: float(probs[i]) for i in range(len(label_cols))}
        labels_short = [
            class_map[label_cols[i]]
            for i in range(len(label_cols))
            if probs[i] >= best_thresholds[label_cols[i]]
        ]

        results.append({
            "input": text,
            "proba": proba,
            "labels_short": labels_short
        })

    return results