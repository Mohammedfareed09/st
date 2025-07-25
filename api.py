from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os, sys, json

# Load fairseq-signals model from your local path
sys.path.append(os.path.join(os.getcwd(), "fairseq-signals"))
from fairseq_signals.models import build_model_from_checkpoint

# Load the ECG-FM model
ckpt_path = os.path.join("ckpts", "mimic_iv_ecg_finetuned.pt")
model = build_model_from_checkpoint(ckpt_path)
model.eval()

# Labels
label_names = [
    "Myocardial Infarction",
    "Atrial Fibrillation",
    "Bundle Branch Block",
    "Class_3", "Class_4", "Class_5", "Class_6",
    "Class_7", "Class_8", "Class_9", "Class_10",
    "Class_11", "Class_12", "Class_13", "Class_14",
    "Class_15", "Class_16"
]

# FastAPI setup
app = FastAPI()

class ECGInput(BaseModel):
    ecg: list[float]

def interpret(p):
    if p < 0.05:
        return "impossible"
    elif p < 0.2:
        return "good"
    elif p < 0.5:
        return "okay"
    elif p < 0.8:
        return "monitor"
    else:
        return "dangerous"

@app.post("/predict")
def predict_ecg(data: ECGInput):
    try:
        signal = np.array(data.ecg, dtype=np.float32)
        x = torch.from_numpy(signal).unsqueeze(0).unsqueeze(0)  # (1,1,L)
        x = x.repeat(1, 12, 1)  # (1,12,L)

        with torch.no_grad():
            out = model(source=x)
            probs = torch.sigmoid(out["out"])[0].tolist()

        results = []
        for name, p in zip(label_names, probs):
            results.append({
                "class": name,
                "probability": round(p, 3),
                "interpretation": interpret(p)
            })

        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
